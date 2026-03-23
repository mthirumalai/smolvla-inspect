#!/usr/bin/env python3
"""
Augment a LeRobot dataset by applying configurable image transforms.

Uses SAM-based scene segmentation to separate foreground from background,
then applies augmentations (background replacement, color jitter, blur, noise,
crop, cutout) per the YAML config. Writes augmented episodes as a new
LeRobot dataset.

Usage:
    python scripts/augment_dataset.py \
        --dataset mthirumalai/so101.pnp.1 \
        --config configs/augmentation.yaml \
        --output-repo my-org/so101.pnp.1.augmented \
        --output-dir ./augmented_data

    # Augment only specific episodes:
    python scripts/augment_dataset.py \
        --dataset mthirumalai/so101.pnp.1 \
        --config configs/augmentation.yaml \
        --output-repo augmented \
        --episodes 0 1 2

    # Override config values from CLI:
    python scripts/augment_dataset.py \
        --dataset mthirumalai/so101.pnp.1 \
        --config configs/augmentation.yaml \
        --output-repo augmented \
        --num-copies 5 \
        --seed 123
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from smolvla_inspect.augment.transforms import augment_image, _load_background_images
from smolvla_inspect.diagnostic.models import SceneSegmentation

# Keys managed by LeRobotDataset.add_frame() / save_episode() — never pass these in frames
_MANAGED_KEYS = {"episode_index", "frame_index", "timestamp", "index", "task_index"}


def load_config(config_path: str | Path) -> dict:
    """Load YAML augmentation config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def tensor_to_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor to (H, W, C) float32 numpy array."""
    if t.dim() == 3:
        return t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return t.cpu().numpy().astype(np.float32)


def img_to_hwc_uint8(img_float_hwc: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) float32 [0,1] to uint8 [0,255] for add_frame."""
    return (np.clip(img_float_hwc, 0.0, 1.0) * 255).astype(np.uint8)


class CachedSegmentationModels:
    """Cache OWL-ViT and SAM models to avoid reloading on every frame."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.owl_processor = None
        self.owl_model = None
        self.sam_predictor = None
        self._models_loaded = False

    def _load_models_once(self):
        """Load OWL-ViT and SAM models if not already loaded."""
        if self._models_loaded:
            return

        print(f"  Loading OWL-ViT v2 (google/owlv2-base-patch16-ensemble) on {self.device}...")
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.owl_model = self.owl_model.to(self.device)
            self.owl_model.eval()
        except ImportError:
            print("  WARNING: transformers not available. Object detection will be skipped.")
            return

        print(f"  Loading SAM vit_b on {self.device}...")
        try:
            from segment_anything import SamPredictor, sam_model_registry
            from smolvla_inspect.diagnostic.scene import _ensure_sam_checkpoint
            checkpoint_path = _ensure_sam_checkpoint()
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam = sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
        except ImportError:
            print("  WARNING: segment_anything not available. Will use bounding box fallbacks.")

        self._models_loaded = True
        print(f"  Models loaded successfully on {self.device}")

    def segment_frame(self, img_hwc: np.ndarray, task_objects: list[str]) -> SceneSegmentation:
        """Run detection + segmentation on a single frame using cached models."""
        from smolvla_inspect.diagnostic.scene import SceneSegmentation, DetectedObject
        import torch
        import numpy as np
        from PIL import Image

        # Load models once if not already loaded
        self._load_models_once()

        # Convert to uint8 for detection/segmentation
        img_uint8 = (img_hwc * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]

        # Skip if no models available
        if not self._models_loaded or self.owl_model is None:
            return SceneSegmentation(
                objects=[],
                background_mask=np.ones((h, w), dtype=bool),
                image_shape=(h, w),
            )

        # Run object detection with cached models
        detections = self._detect_objects_cached(img_uint8, task_objects)

        # Run segmentation with cached models
        segmentation = self._segment_scene_cached(img_uint8, detections)

        return segmentation

    def _detect_objects_cached(self, image: np.ndarray, object_queries: list[str]) -> list:
        """Run object detection using cached OWL-ViT model."""
        from smolvla_inspect.diagnostic.scene import DetectedObject
        import torch
        from PIL import Image

        if not object_queries or self.owl_model is None:
            return []

        # Convert numpy image to PIL
        pil_image = Image.fromarray(image)

        # Process inputs
        inputs = self.owl_processor(text=[object_queries], images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.owl_model(**inputs)

        # Post-process at the primary threshold
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.owl_processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=target_sizes,
        )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        for box, score, label_idx in zip(boxes, scores, labels):
            if label_idx < len(object_queries):
                detections.append(DetectedObject(
                    label=object_queries[label_idx],
                    box=tuple(box.astype(int).tolist()),
                    score=float(score),
                    mask=None
                ))

        return detections

    def _segment_scene_cached(self, image: np.ndarray, detections: list) -> SceneSegmentation:
        """Run scene segmentation using cached SAM model."""
        from smolvla_inspect.diagnostic.scene import SceneSegmentation, DetectedObject
        import numpy as np

        h, w = image.shape[:2]

        if not detections:
            return SceneSegmentation(
                objects=[],
                background_mask=np.ones((h, w), dtype=bool),
                image_shape=(h, w),
            )

        if self.sam_predictor is None:
            # Fallback to bounding box masks
            segmented_objects = []
            union_mask = np.zeros((h, w), dtype=bool)

            for det in detections:
                x1, y1, x2, y2 = det.box
                x1, x2 = max(0, int(x1)), min(w, int(x2))
                y1, y2 = max(0, int(y1)), min(h, int(y2))

                bbox_mask = np.zeros((h, w), dtype=bool)
                bbox_mask[y1:y2, x1:x2] = True
                union_mask |= bbox_mask

                segmented_objects.append(DetectedObject(
                    label=det.label,
                    box=det.box,
                    score=det.score,
                    mask=bbox_mask
                ))

            background_mask = ~union_mask
            return SceneSegmentation(
                objects=segmented_objects,
                background_mask=background_mask,
                image_shape=(h, w),
            )

        # Use SAM for proper segmentation
        self.sam_predictor.set_image(image)

        segmented_objects = []
        union_mask = np.zeros((h, w), dtype=bool)

        for det in detections:
            x1, y1, x2, y2 = det.box
            box_array = np.array([x1, y1, x2, y2])

            masks, scores, _ = self.sam_predictor.predict(box=box_array, multimask_output=True)

            if len(masks) > 0:
                best_mask = masks[np.argmax(scores)]
                union_mask |= best_mask

                segmented_objects.append(DetectedObject(
                    label=det.label,
                    box=det.box,
                    score=det.score,
                    mask=best_mask
                ))

        background_mask = ~union_mask
        return SceneSegmentation(
            objects=segmented_objects,
            background_mask=background_mask,
            image_shape=(h, w),
        )


def segment_frame(
    img_hwc: np.ndarray,
    task_objects: list[str],
    device: str,
) -> SceneSegmentation:
    """Run detection + segmentation on a single frame."""
    # This function is kept for backward compatibility but should use cached version
    from smolvla_inspect.diagnostic.scene import detect_objects, segment_scene

    # Convert to uint8 for detection/segmentation
    img_uint8 = (img_hwc * 255).astype(np.uint8)
    detections = detect_objects(img_uint8, task_objects, device=device)
    segmentation = segment_scene(img_uint8, detections, device=device)
    return segmentation


def get_episode_frame_indices(dataset, episode_idx: int) -> list[int]:
    """Get the global frame indices for all frames in an episode."""
    ep_meta = dataset.meta.episodes[episode_idx]
    from_idx = ep_meta["dataset_from_index"]
    to_idx = ep_meta["dataset_to_index"]
    return list(range(from_idx, to_idx))


def get_user_features(source_dataset) -> dict:
    """Extract user-defined features (excluding DEFAULT_FEATURES) from source dataset."""
    from lerobot.datasets.utils import DEFAULT_FEATURES
    features = {}
    for key, spec in source_dataset.features.items():
        if key not in DEFAULT_FEATURES:
            features[key] = dict(spec)
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Augment a LeRobot dataset with configurable image transforms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Source dataset repo_id (e.g., mthirumalai/so101.pnp.1)")
    parser.add_argument("--config", required=True, help="Path to augmentation YAML config")
    parser.add_argument("--output-repo", required=True, help="Output dataset repo_id")
    parser.add_argument("--output-dir", default=None, help="Local output directory (default: ~/.cache/huggingface/lerobot/<output-repo>)")
    parser.add_argument("--episodes", nargs="*", type=int, default=None, help="Episode indices to augment (default: all)")
    parser.add_argument("--num-copies", type=int, default=None, help="Override config num_copies")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--device", default=None, help="Override config device (auto/cpu/cuda/mps)")
    parser.add_argument("--task", default=None, help="Override task string for object detection")
    parser.add_argument("--include-originals", action="store_true", help="Also copy original (unaugmented) episodes into the output dataset")
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip SAM segmentation (no mask-aware augmentations)")
    parser.add_argument("--seg-every-n", type=int, default=None, help="Re-run segmentation every N frames (default: from config, or 1 = every frame)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing data")
    parser.add_argument("--push-to-hub", action="store_true", help="Push output dataset to HuggingFace Hub after creation")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.num_copies is not None:
        config["num_copies"] = args.num_copies
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    if args.task is not None:
        config["task_string"] = args.task

    seed = config.get("seed", 42)
    num_copies = config.get("num_copies", 3)
    device = config.get("device", "auto")

    # Resolve "auto" device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Load source dataset
    print(f"Loading source dataset: {args.dataset}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    source = LeRobotDataset(args.dataset)
    fps = source.fps
    print(f"  FPS: {fps}, Episodes: {source.meta.total_episodes}, Frames: {source.meta.total_frames}")

    # Determine which episodes to augment
    if args.episodes is not None:
        episode_indices = args.episodes
    elif config.get("episodes") is not None:
        episode_indices = config["episodes"]
    else:
        episode_indices = list(range(source.meta.total_episodes))

    print(f"  Augmenting {len(episode_indices)} episodes x {num_copies} copies = {len(episode_indices) * num_copies} new episodes")

    # Find image keys
    sample = source[0]
    image_keys = [k for k in sample.keys() if "image" in k.lower() and isinstance(sample.get(k), torch.Tensor) and sample[k].dim() >= 3]
    non_image_feature_keys = [k for k in source.features if k not in image_keys and "image" not in k.lower()]
    print(f"  Image keys: {image_keys}")

    # Get task string for segmentation
    task_string = config.get("task_string") or sample.get("task", "manipulate object")
    if isinstance(task_string, (int, float)):
        # Resolve task index to string
        try:
            tasks_df = source.meta.tasks
            row = tasks_df.loc[tasks_df["task_index"] == int(task_string)]
            if len(row) > 0:
                task_string = str(row.iloc[0].name)  # index is the task string
        except Exception:
            task_string = "manipulate object"
    print(f"  Task string: {task_string!r}")

    # Parse task objects for segmentation
    if not args.skip_segmentation:
        from smolvla_inspect.diagnostic.scene import parse_task_objects
        task_objects = parse_task_objects(task_string)
        # Keep all objects including robot gripper — we want to preserve them during augmentation
        print(f"  Task objects for segmentation: {task_objects}")
    else:
        task_objects = []
        print("  Skipping segmentation (no mask-aware augmentations)")

    # Load background image bank if needed
    bg_images = None
    bg_cfg = config.get("background_replacement", {})
    needs_bg_images = False

    if bg_cfg.get("enabled"):
        strategy = bg_cfg.get("strategy")
        if strategy == "image_bank":
            needs_bg_images = True
        elif strategy == "mix":
            # Check if image_bank is one of the mix strategies
            mix_entries = bg_cfg.get("mix", [])
            for entry in mix_entries:
                if entry.get("strategy") == "image_bank":
                    needs_bg_images = True
                    break

    if needs_bg_images:
        bg_dir = bg_cfg.get("image_bank", {}).get("directory")
        if bg_dir:
            print(f"  Loading background image bank from: {bg_dir}")
            bg_images = _load_background_images(bg_dir)
            print(f"  Loaded {len(bg_images)} background images")
        else:
            print("  WARNING: image_bank strategy selected but no directory specified. Falling back to noise.")
            config["background_replacement"]["strategy"] = "noise"

    # Summarize augmentation plan
    enabled = []
    for aug_name in ["background_replacement", "color_jitter", "gaussian_blur",
                     "random_crop", "foreground_cutout", "background_color_shift",
                     "noise_injection"]:
        if config.get(aug_name, {}).get("enabled", False):
            enabled.append(aug_name)
    print(f"  Enabled augmentations: {', '.join(enabled) or '(none)'}")

    if args.dry_run:
        print("\n  [DRY RUN] Would create dataset with:")
        total = len(episode_indices) * num_copies
        if args.include_originals:
            total += len(episode_indices)
        print(f"    {total} episodes")
        print(f"    Output: {args.output_repo}")
        return

    # Prepare output dataset
    output_root = args.output_dir
    if output_root:
        output_root = Path(output_root)
        # Clean existing output to avoid "directory exists" error
        if output_root.exists():
            print(f"  Removing existing output directory: {output_root}")
            shutil.rmtree(output_root)

    features = get_user_features(source)
    use_videos = config.get("use_videos", True)

    print(f"\nCreating output dataset: {args.output_repo}")
    output = LeRobotDataset.create(
        repo_id=args.output_repo,
        fps=fps,
        features=features,
        root=output_root,
        robot_type=getattr(source.meta, "robot_type", None),
        use_videos=use_videos,
        image_writer_threads=config.get("image_writer_threads", 0),
    )

    # Segmentation frequency: how often to re-run SAM (robot moves between frames)
    seg_every_n = args.seg_every_n if args.seg_every_n is not None else config.get("seg_every_n", 1)

    # Initialize cached segmentation models (load once, reuse for all frames)
    cached_segmentation = None
    if not args.skip_segmentation and task_objects:
        print(f"  Segmentation frequency: every {seg_every_n} frame(s)")
        cached_segmentation = CachedSegmentationModels(device)

    def get_frame_segmentation(
        sample: dict, frame_offset: int, prev_seg: SceneSegmentation | None,
    ) -> SceneSegmentation | None:
        """Get segmentation for a frame, re-running SAM every seg_every_n frames."""
        if args.skip_segmentation or not task_objects or cached_segmentation is None:
            return None
        if prev_seg is not None and frame_offset % seg_every_n != 0:
            return prev_seg
        img_tensor = sample[image_keys[0]]
        img_hwc = tensor_to_hwc(img_tensor)
        return cached_segmentation.segment_frame(img_hwc, task_objects)

    total_episodes_written = 0
    t_start = time.time()

    # Optionally copy originals first
    def _build_frame(sample, image_overrides=None):
        """Build a frame dict from a dataset sample, optionally with augmented images.

        Parameters
        ----------
        sample : dict from source[idx]
        image_overrides : dict mapping image_key -> (H,W,C) float32 augmented array, or None
        """
        frame = {}
        # Task string
        task = sample.get("task", task_string)
        if isinstance(task, (int, float)):
            task = task_string
        frame["task"] = task

        for key in source.features:
            if key in _MANAGED_KEYS:
                continue
            if key in image_keys:
                if image_overrides and key in image_overrides:
                    # Augmented image — already HWC float32
                    frame[key] = image_overrides[key]
                else:
                    # Original image — convert CHW tensor to HWC numpy
                    frame[key] = tensor_to_hwc(sample[key])
            elif key in sample:
                val = sample[key]
                if isinstance(val, torch.Tensor):
                    frame[key] = val.numpy()
                else:
                    frame[key] = val
        return frame

    if args.include_originals:
        print("\nCopying original episodes...")
        for ep_idx in episode_indices:
            frame_indices = get_episode_frame_indices(source, ep_idx)
            for fi in frame_indices:
                sample = source[fi]
                output.add_frame(_build_frame(sample))

            output.save_episode()
            total_episodes_written += 1
            print(f"  Copied episode {ep_idx} ({len(frame_indices)} frames)")

    # Generate augmented episodes
    print(f"\nGenerating {len(episode_indices) * num_copies} augmented episodes...")
    for copy_idx in range(num_copies):
        for ep_idx in episode_indices:
            copy_seed = seed + copy_idx * 10000 + ep_idx
            rng = np.random.RandomState(copy_seed)

            frame_indices = get_episode_frame_indices(source, ep_idx)
            seg = None  # will be computed on first frame

            for frame_offset, fi in enumerate(frame_indices):
                sample = source[fi]

                # Re-segment if needed (tracks moving objects across frames)
                seg = get_frame_segmentation(sample, frame_offset, seg)

                # Augment each image key
                img_overrides = {}
                for img_key in image_keys:
                    img_tensor = sample[img_key]
                    img_hwc = tensor_to_hwc(img_tensor)

                    # Get background mask sized to this image
                    bg_mask = None
                    if seg is not None:
                        from smolvla_inspect.diagnostic.counterfactual import _resize_mask
                        h, w = img_hwc.shape[:2]
                        bg_mask = _resize_mask(seg.background_mask, (h, w))

                    # Per-frame RNG so augmentation varies across frames
                    frame_rng = np.random.RandomState(copy_seed + frame_offset)

                    augmented = augment_image(img_hwc, bg_mask, config, frame_rng, bg_images)
                    img_overrides[img_key] = augmented  # HWC float32

                output.add_frame(_build_frame(sample, img_overrides))

            output.save_episode()
            total_episodes_written += 1
            elapsed = time.time() - t_start
            eps_rate = total_episodes_written / elapsed if elapsed > 0 else 0
            print(f"  [{total_episodes_written}/{len(episode_indices) * num_copies}] "
                  f"Copy {copy_idx}, episode {ep_idx} "
                  f"({len(frame_indices)} frames) — {eps_rate:.1f} ep/s")

    # Finalize
    print("\nFinalizing dataset...")
    output.finalize()

    elapsed = time.time() - t_start
    print(f"\nDone! Created {total_episodes_written} episodes in {elapsed:.1f}s")
    print(f"  Output: {output.root}")

    # Save augmentation config as metadata
    meta_path = output.root / "augmentation_config.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Config saved to: {meta_path}")

    # Save provenance
    provenance = {
        "source_dataset": args.dataset,
        "source_episodes": episode_indices,
        "num_copies": num_copies,
        "seed": seed,
        "total_augmented_episodes": total_episodes_written,
        "include_originals": args.include_originals,
        "enabled_augmentations": enabled,
    }
    prov_path = output.root / "augmentation_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Provenance saved to: {prov_path}")

    if args.push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        output.push_to_hub(private=False, push_videos=use_videos)
        print("  Pushed successfully!")


if __name__ == "__main__":
    main()

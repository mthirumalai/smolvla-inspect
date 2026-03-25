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

from smolvla_inspect.augment.transforms import (
    augment_image,
    _load_background_images,
    resolve_episode_consistency,
)
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


def segment_frame(
    img_hwc: np.ndarray,
    task_objects: list[str],
    device: str,
) -> SceneSegmentation:
    """Run detection + segmentation on a single frame."""
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
    parser.add_argument("--consistent-bg", action="store_true", default=None, help="Same background strategy/image within each episode (default: true)")
    parser.add_argument("--no-consistent-bg", dest="consistent_bg", action="store_false", help="Different background per frame (original behavior)")
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

    # Load background image bank if needed (for image_bank strategy or mix mode containing it)
    bg_images = None
    bg_cfg = config.get("background_replacement", {})
    needs_bg_images = (
        bg_cfg.get("strategy") == "image_bank"
        or (bg_cfg.get("strategy") == "mix" and any(
            e.get("strategy") == "image_bank" for e in bg_cfg.get("mix", [])
        ))
    )
    if bg_cfg.get("enabled") and needs_bg_images:
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
    if not args.skip_segmentation and task_objects:
        print(f"  Segmentation frequency: every {seg_every_n} frame(s)")

    # Consistent background: same strategy/image within each episode (default: true)
    consistent_bg = args.consistent_bg
    if consistent_bg is None:
        consistent_bg = config.get("background_replacement", {}).get("consistent", True)
    if consistent_bg:
        print("  Background consistency: per-episode (same background across all frames)")
    else:
        print("  Background consistency: per-frame (different background each frame)")

    def get_frame_segmentation(
        sample: dict, frame_offset: int, prev_segs: dict[str, SceneSegmentation] | None,
        image_key: str | None = None,
    ) -> dict[str, SceneSegmentation] | None:
        """Get segmentation for a frame, re-running SAM every seg_every_n frames.

        Returns a dict mapping image_key -> SceneSegmentation, one per camera.
        """
        if args.skip_segmentation or not task_objects:
            return None
        if prev_segs is not None and frame_offset % seg_every_n != 0:
            return prev_segs
        result = {}
        for img_key in image_keys:
            img_tensor = sample[img_key]
            img_hwc = tensor_to_hwc(img_tensor)
            result[img_key] = segment_frame(img_hwc, task_objects, device)
        return result

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

            # Pre-resolve background + color jitter for the whole episode
            episode_bg = None
            if consistent_bg:
                episode_bg = resolve_episode_consistency(
                    config, rng, bg_images,
                )

            frame_indices = get_episode_frame_indices(source, ep_idx)
            segs = None  # will be computed on first frame (per-camera dict)

            for frame_offset, fi in enumerate(frame_indices):
                sample = source[fi]

                # Re-segment if needed (per-camera, tracks moving objects)
                segs = get_frame_segmentation(sample, frame_offset, segs)

                # Augment each image key
                img_overrides = {}
                for img_key in image_keys:
                    img_tensor = sample[img_key]
                    img_hwc = tensor_to_hwc(img_tensor)

                    # Get background mask for this specific camera
                    bg_mask = None
                    if segs is not None and img_key in segs:
                        from smolvla_inspect.diagnostic.counterfactual import _resize_mask
                        h, w = img_hwc.shape[:2]
                        bg_mask = _resize_mask(segs[img_key].background_mask, (h, w))

                    # Per-frame RNG so non-background augmentation varies across frames
                    frame_rng = np.random.RandomState(copy_seed + frame_offset)

                    augmented = augment_image(img_hwc, bg_mask, config, frame_rng, bg_images, episode_bg)
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

"""Scene understanding primitives: object detection, segmentation, and dataset diversity."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .models import DatasetDiversityReport, DetectedObject, SceneSegmentation
from .registry import register_primitive

# ---------------------------------------------------------------------------
# 1. Parse task objects
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "pick", "up", "place", "put", "move", "push", "pull", "grab", "drop",
    "in", "on", "to", "from", "into", "onto", "the", "a", "an", "and",
    "it", "them", "then", "with", "near", "next", "beside", "over",
    "under", "above", "below", "at", "of", "is", "are", "was", "were",
}


@register_primitive(
    "scene.parse_task_objects",
    category="scene",
    cost="cheap",
    description="Extract object names from a task instruction string.",
)
def parse_task_objects(task_string: str) -> list[str]:
    """Extract object names from a task string using heuristic noun-chunk extraction.

    Strategy: split on common prepositions and verbs, filter noise, return
    cleaned noun phrases.  Always appends ``"robot gripper"`` as an implicit
    scene element.

    Example::

        >>> parse_task_objects("pick up the white lego block and place it in the stainless steel cup")
        ['white lego block', 'stainless steel cup', 'robot gripper']
    """
    text = task_string.lower().strip()

    # Build a regex that splits on any stop word (whole-word match)
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in sorted(_STOP_WORDS, key=len, reverse=True)) + r")\b"
    chunks = re.split(pattern, text)

    objects: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip().strip(",").strip()
        # Remove leading/trailing punctuation
        chunk = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", chunk)
        if not chunk or len(chunk) <= 1:
            continue
        # Skip if the chunk is just a pronoun or very short filler
        if chunk in {"it", "them", "this", "that", "its", "these", "those"}:
            continue
        if chunk not in objects:
            objects.append(chunk)

    # Always include the robot gripper as an implicit object
    if "robot gripper" not in objects:
        objects.append("robot gripper")

    print(f"  Parsed task objects: {objects}")
    return objects


# ---------------------------------------------------------------------------
# 2. Detect objects (OWL-ViT v2)
# ---------------------------------------------------------------------------

@register_primitive(
    "scene.detect_objects",
    category="scene",
    cost="moderate",
    requires_scene_models=True,
    description="Detect objects in an image using OWL-ViT v2.",
)
def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _per_class_nms(
    detections: list[DetectedObject],
    iou_threshold: float = 0.5,
    max_per_class: int = 3,
) -> list[DetectedObject]:
    """Per-class greedy NMS: suppress overlapping boxes, keep top-K per class."""
    # Group by label
    by_label: dict[str, list[DetectedObject]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)

    kept: list[DetectedObject] = []
    for label, dets in by_label.items():
        # Already sorted by score descending from caller
        survivors: list[DetectedObject] = []
        for det in dets:
            suppressed = False
            for survivor in survivors:
                if _box_iou(det.box, survivor.box) > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                survivors.append(det)
            if len(survivors) >= max_per_class:
                break
        kept.extend(survivors)

    # Re-sort by score
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept


_FALLBACK_CONFIDENCE = 0.03


def detect_objects(
    image: np.ndarray,
    object_queries: list[str],
    confidence_threshold: float = 0.1,
    nms_iou_threshold: float = 0.5,
    max_per_class: int = 3,
    device: str = "cpu",
    model_id: str = "google/owlv2-base-patch16-ensemble",
) -> list[DetectedObject]:
    """Run open-vocabulary object detection using OWL-ViT v2.

    If any query label has zero detections above *confidence_threshold*, a
    fallback pass at a lower threshold is used to recover the best-scoring
    detection for that label (provided it exceeds ``_FALLBACK_CONFIDENCE``).
    This ensures task-critical objects mentioned in the task string are not
    silently dropped.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) uint8 image.
    object_queries : list[str]
        Text queries describing objects to detect.
    confidence_threshold : float
        Minimum confidence score to keep a detection.
    nms_iou_threshold : float
        IoU threshold for per-class non-maximum suppression.
    max_per_class : int
        Maximum detections to keep per object class.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    list[DetectedObject]
        Detected objects with bounding boxes and scores (no masks).
    """
    if len(object_queries) == 0:
        print("  No object queries provided, skipping detection.")
        return []

    try:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError:
        print("  WARNING: transformers not available or OWL-ViT v2 not installed. "
              "Returning empty detections.")
        return []

    try:
        print(f"  Loading OWL-ViT v2 ({model_id}) on {device}...")
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
        model = model.to(device)
        model.eval()

        # Convert numpy image to PIL
        pil_image = Image.fromarray(image)

        # Process inputs — OWL-ViT expects a list of list of text queries
        inputs = processor(text=[object_queries], images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process at the primary threshold
        target_sizes = torch.tensor([pil_image.size[::-1]], device=device)  # (H, W)
        results = processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes,
        )[0]

        detections: list[DetectedObject] = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int).tolist()
            label = object_queries[label_idx] if label_idx < len(object_queries) else f"object_{label_idx}"
            detections.append(DetectedObject(
                label=label,
                box=(x1, y1, x2, y2),
                score=float(score),
                mask=None,
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.score, reverse=True)

        raw_count = len(detections)

        # Per-class NMS to remove overlapping duplicates
        detections = _per_class_nms(detections, iou_threshold=nms_iou_threshold,
                                     max_per_class=max_per_class)

        # --- Fallback for missing task objects ---
        # Re-scan at a lower threshold to recover any query labels that had
        # zero detections (e.g. a small cup scoring 0.07 vs threshold 0.1).
        detected_labels = {d.label for d in detections}
        missing = [q for q in object_queries if q not in detected_labels]

        if missing and _FALLBACK_CONFIDENCE < confidence_threshold:
            fallback_results = processor.post_process_object_detection(
                outputs, threshold=_FALLBACK_CONFIDENCE, target_sizes=target_sizes,
            )[0]
            fb_boxes = fallback_results["boxes"].cpu().numpy()
            fb_scores = fallback_results["scores"].cpu().numpy()
            fb_labels = fallback_results["labels"].cpu().numpy()

            for query in missing:
                query_idx = object_queries.index(query)
                # Find the best scoring detection for this query
                best_score = 0.0
                best_box = None
                for box, score, label_idx in zip(fb_boxes, fb_scores, fb_labels):
                    if int(label_idx) == query_idx and float(score) > best_score:
                        best_score = float(score)
                        best_box = box.astype(int).tolist()
                if best_box is not None:
                    x1, y1, x2, y2 = best_box
                    detections.append(DetectedObject(
                        label=query,
                        box=(x1, y1, x2, y2),
                        score=best_score,
                        mask=None,
                    ))
                    print(f"  Recovered '{query}' at fallback threshold "
                          f"(score={best_score:.3f})")
                else:
                    print(f"  WARNING: '{query}' not detected even at "
                          f"fallback threshold {_FALLBACK_CONFIDENCE}")

        det_summary = ", ".join(f"{d.label} ({d.score:.2f})" for d in detections)
        print(f"  Detected {raw_count} raw → {len(detections)} after NMS: {det_summary}")

    except Exception as e:
        print(f"  WARNING: OWL-ViT detection failed: {e}")
        detections = []

    finally:
        # Clean up model from GPU
        try:
            del model, processor
            if device != "cpu":
                torch.cuda.empty_cache()
        except NameError:
            pass

    return detections


# ---------------------------------------------------------------------------
# 3. Segment scene (SAM)
# ---------------------------------------------------------------------------

_SAM_CHECKPOINT_DIR = os.path.expanduser("~/.cache/smolvla_inspect")
_SAM_CHECKPOINT_PATH = os.path.join(_SAM_CHECKPOINT_DIR, "sam_vit_b.pth")
_SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def _ensure_sam_checkpoint() -> str:
    """Download SAM vit_b checkpoint if not already cached."""
    if os.path.exists(_SAM_CHECKPOINT_PATH):
        return _SAM_CHECKPOINT_PATH

    print(f"  Downloading SAM vit_b checkpoint to {_SAM_CHECKPOINT_PATH}...")
    os.makedirs(_SAM_CHECKPOINT_DIR, exist_ok=True)

    import urllib.request
    urllib.request.urlretrieve(_SAM_CHECKPOINT_URL, _SAM_CHECKPOINT_PATH)
    print("  SAM checkpoint downloaded.")
    return _SAM_CHECKPOINT_PATH


def _bbox_fallback_masks(
    detections: list[DetectedObject], h: int, w: int,
) -> SceneSegmentation:
    """Create rectangular masks from bounding boxes when SAM is unavailable."""
    union_mask = np.zeros((h, w), dtype=bool)
    objects_with_masks: list[DetectedObject] = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        mask = np.zeros((h, w), dtype=bool)
        mask[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = True
        objects_with_masks.append(DetectedObject(
            label=det.label, box=det.box, score=det.score, mask=mask,
        ))
        union_mask |= mask
    return SceneSegmentation(
        objects=objects_with_masks,
        background_mask=~union_mask,
        image_shape=(h, w),
    )


@register_primitive(
    "scene.segment_scene",
    category="scene",
    cost="moderate",
    requires_scene_models=True,
    description="Segment scene into object masks using SAM, guided by detection boxes.",
)
def segment_scene(
    image: np.ndarray,
    detections: list[DetectedObject],
    device: str = "cpu",
) -> SceneSegmentation:
    """Segment a scene using SAM, prompted by detected bounding boxes.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) uint8 image.
    detections : list[DetectedObject]
        Objects detected by :func:`detect_objects` (boxes used as SAM prompts).
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    SceneSegmentation
        Segmented scene with per-object masks and a background mask.
    """
    h, w = image.shape[:2]

    if not detections:
        print("  No detections to segment — returning full background mask.")
        return SceneSegmentation(
            objects=[],
            background_mask=np.ones((h, w), dtype=bool),
            image_shape=(h, w),
        )

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print("  WARNING: segment_anything not installed. Falling back to bounding-box masks.")
        return _bbox_fallback_masks(detections, h, w)

    predictor = None
    try:
        checkpoint_path = _ensure_sam_checkpoint()
        print(f"  Loading SAM vit_b on {device}...")
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam = sam.to(device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        segmented_objects: list[DetectedObject] = []
        union_mask = np.zeros((h, w), dtype=bool)

        for det in detections:
            x1, y1, x2, y2 = det.box
            box_array = np.array([x1, y1, x2, y2])

            masks, scores, _ = predictor.predict(
                box=box_array,
                multimask_output=True,
            )

            # Pick the highest-scoring mask
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(bool)

            segmented_objects.append(DetectedObject(
                label=det.label,
                box=det.box,
                score=det.score,
                mask=mask,
            ))
            union_mask |= mask

        background_mask = ~union_mask

        obj_summary = ", ".join(
            f"{o.label} ({o.mask.sum()} px)" for o in segmented_objects if o.mask is not None
        )
        print(f"  Segmented {len(segmented_objects)} objects: {obj_summary}")
        print(f"  Background: {background_mask.sum()} px")

    except Exception as e:
        print(f"  WARNING: SAM segmentation failed: {e}. Falling back to bounding-box masks.")
        fallback = _bbox_fallback_masks(detections, h, w)
        segmented_objects = fallback.objects
        background_mask = fallback.background_mask

    finally:
        # Clean up SAM from GPU
        try:
            del predictor, sam
            if device != "cpu":
                torch.cuda.empty_cache()
        except NameError:
            pass

    return SceneSegmentation(
        objects=segmented_objects,
        background_mask=background_mask,
        image_shape=(h, w),
    )


# ---------------------------------------------------------------------------
# 4. Task-string embedding helpers
# ---------------------------------------------------------------------------

_DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-512"


def _compute_text_embedding_spread(
    unique_tasks: list[str],
    device: str = "cpu",
    siglip_model_id: str = _DEFAULT_SIGLIP_MODEL,
) -> float:
    """Mean pairwise cosine distance of task strings using SigLIP's text encoder."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("  transformers not available, falling back to character distance.")
        return _character_set_spread(unique_tasks)

    try:
        print(f"  Computing SigLIP text embeddings ({siglip_model_id})...")
        tokenizer = AutoTokenizer.from_pretrained(siglip_model_id)
        model = AutoModel.from_pretrained(siglip_model_id).to(device).eval()

        inputs = tokenizer(
            unique_tasks, padding=True, truncation=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)

        embeddings = embeddings.cpu().numpy().astype(np.float32)
        # L2-normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        # Mean pairwise cosine distance
        n = len(unique_tasks)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = float(np.dot(embeddings[i], embeddings[j]))
                dists.append(1.0 - cos_sim)

        spread = float(np.mean(dists)) if dists else 0.0
        print(f"  Embedding spread: {spread:.4f} (over {len(dists)} pairs)")
        return spread

    except Exception as e:
        print(f"  SigLIP text embedding failed ({e}), falling back to character distance.")
        return _character_set_spread(unique_tasks)

    finally:
        try:
            del model, tokenizer
            if device != "cpu":
                torch.cuda.empty_cache()
        except NameError:
            pass


def _character_set_spread(unique_tasks: list[str]) -> float:
    """Fallback: mean pairwise character-set Jaccard distance."""
    dists = []
    for i in range(len(unique_tasks)):
        for j in range(i + 1, len(unique_tasks)):
            s1, s2 = set(unique_tasks[i].lower()), set(unique_tasks[j].lower())
            union = s1 | s2
            diff = s1 ^ s2
            dists.append(len(diff) / max(len(union), 1))
    return float(np.mean(dists)) if dists else 0.0


# ---------------------------------------------------------------------------
# 5. Analyze dataset diversity
# ---------------------------------------------------------------------------

@register_primitive(
    "scene.analyze_dataset_diversity",
    category="dataset",
    cost="expensive",
    requires_scene_models=True,
    description="Analyze spatial, lighting, and task diversity across dataset episodes.",
)
def analyze_dataset_diversity(
    dataset,
    image_key: str,
    task_string: str,
    num_episodes: int = 20,
    frames_per_episode: int = 3,
    device: str = "cpu",
) -> DatasetDiversityReport:
    """Compute dataset diversity statistics across sampled episodes.

    Analyzes object position variance, background diversity, lighting
    consistency, and task string diversity.

    Parameters
    ----------
    dataset
        A LeRobot-style dataset object.
    image_key : str
        Key to extract image tensors from samples.
    task_string : str
        Task instruction to parse for object queries.
    num_episodes : int
        Number of episodes to sample.
    frames_per_episode : int
        Number of evenly-spaced frames per episode.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    DatasetDiversityReport
    """
    from ..data import get_episode_frames

    # Determine available episodes
    try:
        total_episodes = len(dataset.meta.episodes)
    except (AttributeError, TypeError):
        try:
            total_episodes = len(dataset.episode_data_index["from"])
        except (AttributeError, KeyError, TypeError):
            total_episodes = max(1, len(dataset) // 200)

    sampled_episodes = min(num_episodes, total_episodes)
    if sampled_episodes <= 0:
        print("  No episodes to sample.")
        return DatasetDiversityReport(
            object_position_stats={},
            background_diversity_score=0.0,
            lighting_stats={"mean_brightness": 0.0, "contrast_variance": 0.0},
            task_string_diversity={"unique_count": 1, "embedding_spread": 0.0},
            num_episodes_sampled=0,
        )

    # Pick episode indices
    if sampled_episodes >= total_episodes:
        episode_indices = list(range(total_episodes))
    else:
        step = total_episodes / sampled_episodes
        episode_indices = [int(i * step) for i in range(sampled_episodes)]

    print(f"  Sampling {len(episode_indices)} episodes, {frames_per_episode} frames each...")

    # Parse object queries from the task string
    object_queries = parse_task_objects(task_string)

    # Collect per-object centroids, brightness values across all frames
    # object_label -> list of (cx, cy) centroids
    object_centroids: dict[str, list[tuple[float, float]]] = {q: [] for q in object_queries}
    brightness_values: list[float] = []
    contrast_values: list[float] = []

    for ep_idx in episode_indices:
        try:
            frames = get_episode_frames(dataset, ep_idx, frames_per_episode, image_key)
        except (ValueError, IndexError, KeyError) as e:
            print(f"  Skipping episode {ep_idx}: {e}")
            continue

        for frame_idx, img_tensor in frames:
            # Convert tensor (C, H, W) float [0,1] -> numpy (H, W, C) uint8
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                img_np = np.asarray(img_tensor, dtype=np.uint8)

            # Lighting statistics
            gray = np.mean(img_np, axis=2)
            brightness_values.append(float(np.mean(gray)))
            contrast_values.append(float(np.std(gray)))

            # Run detection on this frame
            try:
                dets = detect_objects(img_np, object_queries, confidence_threshold=0.1, device=device)
            except Exception as e:
                print(f"  Detection failed on frame {frame_idx}: {e}")
                continue

            # Collect centroids
            for det in dets:
                x1, y1, x2, y2 = det.box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if det.label in object_centroids:
                    object_centroids[det.label].append((cx, cy))

    # Compute per-object centroid statistics
    object_position_stats: dict[str, dict] = {}
    for label, centroids in object_centroids.items():
        if not centroids:
            object_position_stats[label] = {
                "count": 0,
                "mean_x": None, "mean_y": None,
                "std_x": None, "std_y": None,
            }
            continue
        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        object_position_stats[label] = {
            "count": len(centroids),
            "mean_x": float(np.mean(xs)),
            "mean_y": float(np.mean(ys)),
            "std_x": float(np.std(xs)),
            "std_y": float(np.std(ys)),
        }

    # Background diversity: variance of mean brightness across frames (normalised to 0-1)
    if len(brightness_values) > 1:
        bg_var = float(np.var(brightness_values))
        # Normalise: max possible variance of means in [0, 255] range is ~(127.5)^2
        background_diversity_score = min(1.0, bg_var / (127.5 ** 2))
    else:
        background_diversity_score = 0.0

    # Lighting stats
    lighting_stats = {
        "mean_brightness": float(np.mean(brightness_values)) if brightness_values else 0.0,
        "contrast_variance": float(np.var(contrast_values)) if len(contrast_values) > 1 else 0.0,
    }

    # Task string diversity (SigLIP text embeddings)
    task_string_diversity: dict = {"unique_count": 1, "embedding_spread": 0.0}
    try:
        tasks_df = dataset.meta.tasks
        unique_tasks = tasks_df["task"].unique().tolist() if hasattr(tasks_df["task"], "unique") else list(set(tasks_df["task"]))
        task_string_diversity["unique_count"] = len(unique_tasks)
        if len(unique_tasks) > 1:
            task_string_diversity["embedding_spread"] = _compute_text_embedding_spread(
                unique_tasks, device=device)
    except (AttributeError, KeyError, TypeError):
        pass

    print(f"  Diversity analysis complete: {len(episode_indices)} episodes, "
          f"{len(brightness_values)} frames analyzed.")

    return DatasetDiversityReport(
        object_position_stats=object_position_stats,
        background_diversity_score=background_diversity_score,
        lighting_stats=lighting_stats,
        task_string_diversity=task_string_diversity,
        num_episodes_sampled=len(episode_indices),
    )

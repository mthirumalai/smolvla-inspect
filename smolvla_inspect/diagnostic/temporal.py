"""Temporal attention trajectory analysis.

Tracks how the model's attention centroid moves across episode frames and
correlates the trajectory with ground-truth object positions from OWL-ViT.
"""

from __future__ import annotations

import numpy as np
import torch

from ..capture import SigLIPAttentionCapture
from ..data import (
    find_vision_encoder,
    get_episode_frames,
    build_policy_batch_from_sample,
    _resolve_task_string,
)
from ..heatmap import compute_patch_attention_scores
from .models import TemporalTrajectory
from .registry import register_primitive
from .scene import detect_objects, parse_task_objects


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def _compute_centroid(heatmap: np.ndarray) -> tuple[float, float]:
    """Compute the weighted center of mass of a (H, W) heatmap.

    Returns (cx, cy) normalised to [0, 1] where (0, 0) is the top-left
    corner and (1, 1) is the bottom-right corner.
    """
    h, w = heatmap.shape
    total = heatmap.sum()
    if total <= 0:
        return (0.5, 0.5)

    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * heatmap).sum() / total) / max(w - 1, 1)
    cy = float((ys * heatmap).sum() / total) / max(h - 1, 1)
    return (cx, cy)


# ---------------------------------------------------------------------------
# Main primitive
# ---------------------------------------------------------------------------

@register_primitive(
    "temporal.compute_trajectory",
    category="temporal",
    cost="expensive",
    description="Compute attention centroid trajectory across episode frames.",
)
def compute_temporal_trajectory(
    policy,
    dataset,
    episode_idx: int,
    image_key: str,
    device: str,
    num_frames: int = 20,
    signal_type: str = "attention",
    image_map=None,
) -> TemporalTrajectory:
    """Compute how the attention centroid moves across an episode.

    For each evenly-spaced frame the SigLIP last-layer attention is captured,
    reduced to a heatmap, and the weighted centroid is extracted.  Smoothness
    is measured as the mean inter-frame L2 displacement (normalised to [0, 1]
    image coordinates).  Object-tracking correlation compares the attention
    centroid trajectory to ground-truth object centroids from OWL-ViT.

    Parameters
    ----------
    policy
        SmolVLA policy (must be on *device*).
    dataset
        LeRobot-style dataset.
    episode_idx : int
        Which episode to analyse.
    image_key : str
        Dataset key for image tensors.
    device : str
        ``"cpu"`` or ``"cuda"``.
    num_frames : int
        Number of evenly-spaced frames to sample.
    signal_type : str
        Label stored in the result (e.g. ``"attention"``).
    image_map
        Optional image-key mapping passed through to batch building.

    Returns
    -------
    TemporalTrajectory
    """
    frames = get_episode_frames(dataset, episode_idx, num_frames, image_key)

    if len(frames) < 2:
        return TemporalTrajectory(
            frame_indices=[fi for fi, _ in frames],
            centroids=[(0.5, 0.5)] * len(frames),
            smoothness=0.0,
            object_tracking_correlation=0.0,
            signal_type=signal_type,
        )

    # Set up SigLIP attention capture
    vision_encoder = find_vision_encoder(policy)
    capture = SigLIPAttentionCapture()
    capture.register_hooks(vision_encoder)

    img_size = getattr(
        getattr(vision_encoder, "config", None), "image_size", 512,
    )
    patch_size = getattr(
        getattr(vision_encoder, "config", None), "patch_size", 16,
    )
    grid_h = img_size // patch_size
    grid_w = img_size // patch_size

    frame_indices: list[int] = []
    centroids: list[tuple[float, float]] = []

    try:
        for frame_idx, img_tensor in frames:
            # Get the full sample from the dataset for proper batch building
            sample = dataset[frame_idx]

            policy.reset()
            capture.reset_maps()

            batch, _ = build_policy_batch_from_sample(
                sample, policy, device, dataset=dataset, image_map=image_map,
            )

            with torch.no_grad():
                policy.select_action(batch)

            # Extract last-layer attention, average across heads
            last_attn = capture.get_last_layer_attention()
            if last_attn is None:
                centroids.append((0.5, 0.5))
                frame_indices.append(frame_idx)
                continue

            scores = compute_patch_attention_scores(last_attn)

            # Convert to a 2-D heatmap (grid_h x grid_w)
            if isinstance(scores, torch.Tensor):
                scores_np = scores.float().cpu().numpy()
            else:
                scores_np = np.asarray(scores, dtype=np.float32)
            heatmap = scores_np.reshape(grid_h, grid_w)
            hmin, hmax = heatmap.min(), heatmap.max()
            if hmax > hmin:
                heatmap = (heatmap - hmin) / (hmax - hmin)

            centroids.append(_compute_centroid(heatmap))
            frame_indices.append(frame_idx)
    finally:
        capture.clear()

    # Smoothness: mean L2 distance between consecutive centroids
    # Centroids are already normalised to [0, 1] so values are comparable.
    displacements: list[float] = []
    for i in range(1, len(centroids)):
        dx = centroids[i][0] - centroids[i - 1][0]
        dy = centroids[i][1] - centroids[i - 1][1]
        displacements.append(float(np.sqrt(dx * dx + dy * dy)))
    smoothness = float(np.mean(displacements)) if displacements else 0.0

    # Object-tracking correlation
    correlation = _compute_object_tracking_correlation(
        frames, centroids, dataset, device,
    )

    return TemporalTrajectory(
        frame_indices=frame_indices,
        centroids=centroids,
        smoothness=smoothness,
        object_tracking_correlation=correlation,
        signal_type=signal_type,
    )


# ---------------------------------------------------------------------------
# Object-tracking correlation helper
# ---------------------------------------------------------------------------

def _compute_object_tracking_correlation(
    frames: list[tuple[int, torch.Tensor]],
    attention_centroids: list[tuple[float, float]],
    dataset,
    device: str,
) -> float:
    """Correlate attention centroids with OWL-ViT object centroids.

    Runs OWL-ViT on each frame, averages detected object centroids, and
    computes the Pearson correlation between the attention x-trajectory and
    the object x-trajectory (and likewise for y), returning the mean.

    Returns 0.0 if detection fails or there are fewer than 3 data points.
    """
    if len(frames) < 3:
        return 0.0

    # Determine object queries from the task string
    try:
        # Get a real sample from the dataset to resolve the task string
        first_frame_idx = frames[0][0]
        real_sample = dataset[first_frame_idx]
        task_str = _resolve_task_string(real_sample, dataset=dataset)
        object_queries = parse_task_objects(task_str)
        # Remove generic 'robot gripper' — unlikely to be consistently detected
        object_queries = [q for q in object_queries if q != "robot gripper"]
    except Exception:
        return 0.0

    if not object_queries:
        return 0.0

    object_centroids: list[tuple[float, float] | None] = []

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

        h, w = img_np.shape[:2]

        try:
            dets = detect_objects(
                img_np, object_queries,
                confidence_threshold=0.1, device=device,
            )
        except Exception:
            object_centroids.append(None)
            continue

        if not dets:
            object_centroids.append(None)
            continue

        # Average centroid of all detected objects, normalised to [0, 1]
        cxs, cys = [], []
        for det in dets:
            x1, y1, x2, y2 = det.box
            cxs.append((x1 + x2) / 2.0 / max(w - 1, 1))
            cys.append((y1 + y2) / 2.0 / max(h - 1, 1))
        object_centroids.append((float(np.mean(cxs)), float(np.mean(cys))))

    # Build parallel arrays of valid data points
    attn_x, attn_y = [], []
    obj_x, obj_y = [], []
    for ac, oc in zip(attention_centroids, object_centroids):
        if oc is None:
            continue
        attn_x.append(ac[0])
        attn_y.append(ac[1])
        obj_x.append(oc[0])
        obj_y.append(oc[1])

    if len(attn_x) < 3:
        return 0.0

    # Pearson correlation for x and y separately, then average
    try:
        corr_x = float(np.corrcoef(attn_x, obj_x)[0, 1])
        corr_y = float(np.corrcoef(attn_y, obj_y)[0, 1])
    except Exception:
        return 0.0

    # Handle NaN from constant arrays
    if np.isnan(corr_x):
        corr_x = 0.0
    if np.isnan(corr_y):
        corr_y = 0.0

    return (corr_x + corr_y) / 2.0

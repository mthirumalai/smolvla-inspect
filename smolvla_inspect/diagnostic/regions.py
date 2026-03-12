"""Region attribution scoring -- pure numpy, no GPU needed.

Takes heatmaps + segmentation masks, returns per-region attribution scores.
"""

from __future__ import annotations

import numpy as np

from .models import SceneSegmentation
from .registry import register_primitive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_heatmap(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize *heatmap* to *target_shape* (H, W) using bilinear-style zoom.

    Uses ``scipy.ndimage.zoom`` when available, otherwise falls back to a
    simple nearest-neighbour resize via ``np.repeat`` / slicing so the module
    stays lightweight.
    """
    if heatmap.shape == target_shape:
        return heatmap
    try:
        from scipy.ndimage import zoom

        zoom_factors = (
            target_shape[0] / heatmap.shape[0],
            target_shape[1] / heatmap.shape[1],
        )
        return zoom(heatmap, zoom_factors, order=1)
    except ImportError:  # pragma: no cover -- scipy optional at this layer
        # Nearest-neighbour fallback
        row_idx = (np.arange(target_shape[0]) * heatmap.shape[0] / target_shape[0]).astype(int)
        col_idx = (np.arange(target_shape[1]) * heatmap.shape[1] / target_shape[1]).astype(int)
        return heatmap[np.ix_(row_idx, col_idx)]


def _build_region_masks(
    segmentation: SceneSegmentation,
) -> list[tuple[str, np.ndarray]]:
    """Return a list of ``(label, mask)`` pairs covering every region.

    Each detected object contributes one entry; the background is always last.
    If an object has no mask we skip it (shouldn't happen in practice).
    """
    pairs: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for obj in segmentation.objects:
        if obj.label in seen:
            continue
        seen.add(obj.label)
        if obj.mask is not None:
            pairs.append((obj.label, obj.mask.astype(float)))
    pairs.append(("background", segmentation.background_mask.astype(float)))
    return pairs


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@register_primitive(
    "regions.attribute_to_regions",
    category="composite",
    cost="cheap",
    description="Compute per-region attribution shares from a heatmap and segmentation.",
)
def attribute_to_regions(
    heatmap: np.ndarray,
    segmentation: SceneSegmentation,
    signal_name: str = "",
) -> dict[str, float]:
    """Per-region share of *heatmap* attribution.

    Parameters
    ----------
    heatmap:
        (H, W) float array -- any saliency/attention/GradCAM map.
    segmentation:
        Scene segmentation providing region masks.
    signal_name:
        Optional label used for logging / debugging only.

    Returns
    -------
    dict mapping region name -> attribution fraction (sums to ~1.0).
    """
    heatmap = _resize_heatmap(heatmap, segmentation.image_shape)
    total = float(np.sum(heatmap))

    region_masks = _build_region_masks(segmentation)

    # Near-zero total -> equal distribution
    if total < 1e-12:
        n = len(region_masks)
        return {label: 1.0 / n for label, _ in region_masks}

    result: dict[str, float] = {}
    for label, mask in region_masks:
        result[label] = float(np.sum(heatmap * mask) / total)
    return result


@register_primitive(
    "regions.attention_gradcam_divergence",
    category="composite",
    cost="cheap",
    description=(
        "Per-region divergence between attention and GradCAM attribution. "
        "Positive means looking-but-not-using; negative means using-without-looking."
    ),
)
def attention_gradcam_divergence(
    attention_heatmap: np.ndarray,
    gradcam_heatmap: np.ndarray,
    segmentation: SceneSegmentation,
) -> dict[str, float]:
    """Per-region attention_share minus gradcam_share."""
    attn_shares = attribute_to_regions(attention_heatmap, segmentation, signal_name="attention")
    gc_shares = attribute_to_regions(gradcam_heatmap, segmentation, signal_name="gradcam")

    regions = set(attn_shares) | set(gc_shares)
    return {r: attn_shares.get(r, 0.0) - gc_shares.get(r, 0.0) for r in regions}


@register_primitive(
    "regions.foreground_ratio",
    category="composite",
    cost="cheap",
    description="Fraction of total heatmap attribution falling on foreground (non-background) regions.",
)
def foreground_ratio(
    heatmap: np.ndarray,
    segmentation: SceneSegmentation,
) -> float:
    """Total foreground attribution divided by total attribution."""
    shares = attribute_to_regions(heatmap, segmentation, signal_name="foreground_ratio")
    bg = shares.get("background", 0.0)
    return 1.0 - bg


@register_primitive(
    "regions.spatial_prior_ratio",
    category="composite",
    cost="cheap",
    description="Cosine similarity between a heatmap and a positional baseline heatmap.",
)
def spatial_prior_ratio(
    heatmap: np.ndarray,
    baseline_heatmap: np.ndarray,
) -> float:
    """Cosine similarity between flattened *heatmap* and *baseline_heatmap*.

    Returns a float in [0, 1] indicating what fraction of the signal is
    explained by positional bias.
    """
    a = heatmap.flatten().astype(float)
    b = baseline_heatmap.flatten().astype(float)

    # Resize if shapes differ
    if a.shape != b.shape:
        baseline_resized = _resize_heatmap(baseline_heatmap, heatmap.shape)
        b = baseline_resized.flatten().astype(float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [0, 1]
    return max(0.0, min(1.0, cosine))

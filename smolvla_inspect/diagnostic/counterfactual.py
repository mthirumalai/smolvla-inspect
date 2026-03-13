"""Counterfactual perturbation primitives.

Modify images and re-run the model forward pass to measure how action
predictions change.  Each primitive follows the same pattern:

1. Run forward pass on the original sample to get baseline actions.
2. Modify the image in a cloned sample.
3. Run forward pass on the modified sample to get perturbed actions.
4. Compare via L2 norm of the action delta and per-dimension deltas.
5. Produce a side-by-side visual comparison image.
6. Return a :class:`CounterfactualResult`.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ..data import build_policy_batch_from_sample
from ..gradient import _patch_eager_attention_bool_mask
from .models import CounterfactualResult, SceneSegmentation
from .registry import register_primitive
from .semantic_probe import compute_patch_text_similarity


# ---------------------------------------------------------------------------
# Helpers: tensor <-> numpy conversions
# ---------------------------------------------------------------------------

def _tensor_to_hwc(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] tensor -> (H, W, C) float [0,1] numpy."""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _hwc_to_tensor(arr: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """(H, W, C) float [0,1] numpy -> (C, H, W) float tensor."""
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def _make_comparison(
    original_img: np.ndarray,
    modified_img: np.ndarray,
    mask: np.ndarray | None = None,
    zoom_padding: int = 40,
) -> np.ndarray:
    """Create annotated side-by-side comparison. Both are (H, W, 3) uint8 numpy arrays.

    If *mask* is provided, draws a highlight rectangle around the modified region
    and appends a zoomed inset of original vs modified below the main comparison.
    """
    h, w = original_img.shape[:2]
    top = np.concatenate([original_img.copy(), modified_img.copy()], axis=1)

    if mask is None or not mask.any():
        return top

    # Find the bounding box of the mask
    ys, xs = np.where(mask[:h, :w] if mask.shape[0] >= h else mask)
    if len(ys) == 0:
        return top

    y1, y2 = max(int(ys.min()) - zoom_padding, 0), min(int(ys.max()) + zoom_padding, h)
    x1, x2 = max(int(xs.min()) - zoom_padding, 0), min(int(xs.max()) + zoom_padding, w)

    # Draw rectangles on both halves (green = original, red = modified)
    _draw_rect(top, y1, x1, y2, x2, color=(0, 255, 0), thickness=2)
    _draw_rect(top, y1, x1 + w, y2, x2 + w, color=(255, 0, 0), thickness=2)

    # Build zoomed inset row — scale crop to a readable height (min 120px)
    crop_h = y2 - y1
    crop_w = x2 - x1
    if crop_h > 0 and crop_w > 0:
        scale = max(1, 120 // crop_h)
        orig_crop = original_img[y1:y2, x1:x2]
        mod_crop = modified_img[y1:y2, x1:x2]
        orig_zoom = np.repeat(np.repeat(orig_crop, scale, axis=0), scale, axis=1)
        mod_zoom = np.repeat(np.repeat(mod_crop, scale, axis=0), scale, axis=1)

        # Pad zoomed crops so they match the top width
        zoom_row = np.concatenate([orig_zoom, mod_zoom], axis=1)
        pad_w = top.shape[1] - zoom_row.shape[1]
        if pad_w > 0:
            zoom_row = np.pad(zoom_row, ((0, 0), (0, pad_w), (0, 0)), constant_values=32)
        elif pad_w < 0:
            zoom_row = zoom_row[:, :top.shape[1]]

        # Add a 2px separator
        sep = np.full((2, top.shape[1], 3), 128, dtype=np.uint8)
        top = np.concatenate([top, sep, zoom_row], axis=0)

    return top


def _draw_rect(img: np.ndarray, y1: int, x1: int, y2: int, x2: int,
               color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    """Draw a rectangle on an image (in-place). No OpenCV needed."""
    h, w = img.shape[:2]
    for t in range(thickness):
        # Top and bottom edges
        if 0 <= y1 + t < h:
            img[y1 + t, max(x1, 0):min(x2, w)] = color
        if 0 <= y2 - t < h:
            img[y2 - t, max(x1, 0):min(x2, w)] = color
        # Left and right edges
        if 0 <= x1 + t < w:
            img[max(y1, 0):min(y2, h), x1 + t] = color
        if 0 <= x2 - t < w:
            img[max(y1, 0):min(y2, h), x2 - t] = color


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float [0,1] (H, W, C) image to uint8."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)


_ACTION_DIM_LABELS = [
    "x", "y", "z", "roll", "pitch", "yaw",
    "gripper",
]


def _make_action_delta_chart(
    baseline: np.ndarray,
    modified: np.ndarray,
    original_task: str,
    replacement_task: str,
    width: int = 640,
    height: int = 400,
) -> np.ndarray:
    """Render a per-dimension action delta bar chart as a uint8 (H, W, 3) image.

    Used for language-only perturbations where a side-by-side image comparison
    would show two identical frames.
    """
    delta = modified - baseline
    n_dims = len(delta)
    delta_l2 = float(np.linalg.norm(delta))

    # Canvas
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Layout constants
    margin_top = 70
    margin_bottom = 55
    margin_left = 60
    margin_right = 20
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    # Bar geometry
    bar_w = max(2, chart_w // max(n_dims, 1) - 2)
    spacing = chart_w // max(n_dims, 1)

    # Scale
    max_abs = max(abs(delta.max()), abs(delta.min()), 0.01)
    zero_y = margin_top + chart_h // 2

    # Draw zero line
    img[zero_y, margin_left:margin_left + chart_w] = (180, 180, 180)

    # Draw bars
    for i in range(n_dims):
        x = margin_left + i * spacing + (spacing - bar_w) // 2
        val = delta[i]
        bar_h = int(abs(val) / max_abs * (chart_h // 2))
        bar_h = max(bar_h, 1)

        if val >= 0:
            y1, y2 = zero_y - bar_h, zero_y
            color = (66, 133, 244)  # blue
        else:
            y1, y2 = zero_y, zero_y + bar_h
            color = (219, 68, 55)  # red

        img[y1:y2, x:x + bar_w] = color

        # Dim label below chart
        label = _ACTION_DIM_LABELS[i] if i < len(_ACTION_DIM_LABELS) else str(i)
        _draw_text_simple(img, label, x + bar_w // 2 - 3 * len(label),
                          height - margin_bottom + 5, color=(80, 80, 80), scale=1)

    # Y-axis labels
    _draw_text_simple(img, f"+{max_abs:.3f}", margin_left - 55, margin_top, color=(80, 80, 80), scale=1)
    _draw_text_simple(img, "0", margin_left - 15, zero_y - 4, color=(80, 80, 80), scale=1)
    _draw_text_simple(img, f"-{max_abs:.3f}", margin_left - 55, margin_top + chart_h - 8, color=(80, 80, 80), scale=1)

    # Title and task labels
    _draw_text_simple(img, f"Task String Swap — Action Delta (L2={delta_l2:.4f})",
                      margin_left, 10, color=(40, 40, 40), scale=1)
    orig_label = f"Original: \"{_truncate(original_task, 60)}\""
    repl_label = f"Replaced: \"{_truncate(replacement_task, 60)}\""
    _draw_text_simple(img, orig_label, margin_left, 28, color=(80, 80, 80), scale=1)
    _draw_text_simple(img, repl_label, margin_left, 44, color=(219, 68, 55), scale=1)

    return img


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def _draw_text_simple(img: np.ndarray, text: str, x: int, y: int,
                      color: tuple = (0, 0, 0), scale: int = 1):
    """Draw text onto image using PIL (fallback: skip silently)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12 * scale)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((x, y), text, fill=color, font=font)
        img[:] = np.array(pil_img)
    except ImportError:
        pass  # No PIL — skip text rendering


# ---------------------------------------------------------------------------
# Helpers: forward pass and mask resizing
# ---------------------------------------------------------------------------

def _resize_heatmap(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a heatmap to ``target_shape``."""
    if heatmap.shape == target_shape:
        return heatmap
    try:
        from scipy.ndimage import zoom

        zoom_factors = (
            target_shape[0] / heatmap.shape[0],
            target_shape[1] / heatmap.shape[1],
        )
        return zoom(heatmap, zoom_factors, order=1)
    except ImportError:
        row_idx = (np.arange(target_shape[0]) * heatmap.shape[0] / target_shape[0]).astype(int)
        col_idx = (np.arange(target_shape[1]) * heatmap.shape[1] / target_shape[1]).astype(int)
        return heatmap[np.ix_(row_idx, col_idx)]


def _mask_share(heatmap: np.ndarray | None, mask: np.ndarray) -> float:
    """Return the fraction of heatmap mass falling inside ``mask``."""
    if heatmap is None:
        return 0.0
    hm = _resize_heatmap(np.asarray(heatmap, dtype=np.float32), mask.shape)
    total = float(np.sum(hm))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(hm * mask.astype(np.float32)) / total)


def _normalize_similarity_map(similarity_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(similarity_map, dtype=np.float32)
    if not valid_mask.any():
        return out
    valid = similarity_map[valid_mask].astype(np.float32)
    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmax > vmin:
        out[valid_mask] = (valid - vmin) / (vmax - vmin)
    return out


def _semantic_region_mean(similarity_map: np.ndarray | None, region_mask: np.ndarray,
                          grid_shape: tuple[int, int], valid_mask: np.ndarray) -> float | None:
    if similarity_map is None:
        return None
    region_grid = _resize_mask(region_mask, grid_shape) & valid_mask
    if not region_grid.any():
        return None
    return float(similarity_map[region_grid].mean())


def _semantic_region_peak(similarity_map: np.ndarray | None, region_mask: np.ndarray,
                          grid_shape: tuple[int, int], valid_mask: np.ndarray) -> float | None:
    if similarity_map is None:
        return None
    region_grid = _resize_mask(region_mask, grid_shape) & valid_mask
    if not region_grid.any():
        return None
    return float(similarity_map[region_grid].max())


def _shift_mask(mask: np.ndarray, shift_pixels: tuple[int, int]) -> np.ndarray:
    """Translate a binary mask by ``(dx, dy)`` with clipping."""
    dx, dy = shift_pixels
    h, w = mask.shape
    shifted = np.zeros_like(mask, dtype=bool)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return shifted

    ny = ys + dy
    nx = xs + dx
    valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    shifted[ny[valid], nx[valid]] = True
    return shifted

def _get_actions(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    image_map=None,
) -> np.ndarray:
    """Run forward pass and return action predictions as a numpy array.

    Returns the first batch element, first timestep action vector.
    """
    batch, _ = build_policy_batch_from_sample(
        sample, policy, device, image_key_for_grad=None,
        dataset=dataset, image_map=image_map,
    )
    with torch.no_grad(), _patch_eager_attention_bool_mask(policy):
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        bsize = state.shape[0]
        actions_shape = (
            bsize,
            policy.model.config.chunk_size,
            policy.model.config.max_action_dim,
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        noise = torch.randn(
            actions_shape, device=device, generator=gen, dtype=torch.float32,
        )
        actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )

    return actions[0, 0].cpu().numpy()  # first batch, first timestep


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to *target_shape* ``(H, W)``.

    Uses ``scipy.ndimage.zoom`` when available, otherwise nearest-neighbour.
    """
    if mask.shape[:2] == target_shape:
        return mask
    try:
        from scipy.ndimage import zoom

        zoom_factors = (
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1],
        )
        resized = zoom(mask.astype(float), zoom_factors, order=0)
        return resized > 0.5
    except ImportError:
        row_idx = (np.arange(target_shape[0]) * mask.shape[0] / target_shape[0]).astype(int)
        col_idx = (np.arange(target_shape[1]) * mask.shape[1] / target_shape[1]).astype(int)
        return mask[np.ix_(row_idx, col_idx)]


def _clone_sample(sample: dict, image_key: str) -> dict:
    """Shallow-copy the sample dict and deep-copy the image tensor."""
    new_sample = dict(sample)
    new_sample[image_key] = sample[image_key].clone()
    return new_sample


def _compute_result(
    baseline_actions: np.ndarray,
    modified_actions: np.ndarray,
    original_img_hwc: np.ndarray,
    modified_img_hwc: np.ndarray,
    hypothesis_id: str,
    test_type: str,
    affected_mask: np.ndarray | None = None,
    metrics: dict | None = None,
) -> CounterfactualResult:
    """Compare baseline and modified actions, build a CounterfactualResult."""
    delta = modified_actions - baseline_actions
    delta_l2 = float(np.linalg.norm(delta))
    delta_per_dim = delta.tolist()

    comparison = _make_comparison(
        _to_uint8(original_img_hwc),
        _to_uint8(modified_img_hwc),
        mask=affected_mask,
    )

    return CounterfactualResult(
        hypothesis_id=hypothesis_id,
        test_type=test_type,
        action_delta_l2=delta_l2,
        action_delta_per_dim=delta_per_dim,
        gradcam_shift=0.0,
        attribution_shift_per_region={},
        confirmed=delta_l2 > 0.01,
        metrics=metrics or {},
        visual_comparison=comparison,
    )


# ---------------------------------------------------------------------------
# 1. Background substitution
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.background_substitution",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Replace background pixels and measure action change.",
)
def background_substitution(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    segmentation: SceneSegmentation,
    replacement: str = "gray",
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Replace background pixels with gray / noise / blur and measure action shift.

    Parameters
    ----------
    replacement : str
        One of ``"gray"``, ``"noise"``, or ``"blur"``.
    """
    # Baseline actions
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    # Prepare modified sample
    img_tensor = sample[image_key]  # (C, H, W) float [0, 1]
    img_hwc = _tensor_to_hwc(img_tensor)  # (H, W, C)
    h, w = img_hwc.shape[:2]

    bg_mask = _resize_mask(segmentation.background_mask, (h, w))

    modified_hwc = img_hwc.copy()

    if replacement == "gray":
        modified_hwc[bg_mask] = 0.5
    elif replacement == "noise":
        rng = np.random.RandomState(noise_seed)
        noise = rng.rand(h, w, img_hwc.shape[2]).astype(np.float32)
        modified_hwc[bg_mask] = noise[bg_mask]
    elif replacement == "blur":
        try:
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(img_hwc, sigma=(20, 20, 0))
        except ImportError:
            # Simple box-blur fallback
            kernel_size = 41
            pad = kernel_size // 2
            padded = np.pad(img_hwc, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            blurred = np.zeros_like(img_hwc)
            for c in range(img_hwc.shape[2]):
                cumsum = np.cumsum(np.cumsum(padded[:, :, c], axis=0), axis=1)
                blurred[:, :, c] = (
                    cumsum[kernel_size:, kernel_size:]
                    - cumsum[:-kernel_size, kernel_size:]
                    - cumsum[kernel_size:, :-kernel_size]
                    + cumsum[:-kernel_size, :-kernel_size]
                ) / (kernel_size * kernel_size)
            blurred = blurred[:h, :w]
        modified_hwc[bg_mask] = blurred[bg_mask]
    else:
        raise ValueError(f"Unknown replacement mode: {replacement!r}")

    modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

    # Build modified sample and run forward
    new_sample = _clone_sample(sample, image_key)
    new_sample[image_key] = _hwc_to_tensor(modified_hwc, device=str(img_tensor.device))

    policy.reset()
    modified_actions = _get_actions(policy, new_sample, dataset, image_key, device, image_map)

    return _compute_result(
        baseline_actions, modified_actions,
        img_hwc, modified_hwc,
        hypothesis_id=f"background_substitution_{replacement}",
        test_type="background_substitution",
        affected_mask=bg_mask,
        metrics={"replacement": replacement},
    )


# ---------------------------------------------------------------------------
# 2. Object relocation
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.object_relocation",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Move an object in the image and measure action change.",
)
def object_relocation(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    segmentation: SceneSegmentation,
    target_object: str,
    shift_pixels: tuple[int, int] = (100, -80),
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Relocate *target_object* by *shift_pixels* ``(dx, dy)`` and measure action shift."""
    from ..gradient import compute_gradcam_map

    # Baseline actions
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    img_tensor = sample[image_key]  # (C, H, W) float [0, 1]
    img_hwc = _tensor_to_hwc(img_tensor)  # (H, W, C)
    h, w = img_hwc.shape[:2]

    obj_mask = segmentation.get_mask(target_object)
    if obj_mask is None:
        print(f"  WARNING: No mask found for object '{target_object}', returning zero-delta result.")
        comparison = _make_comparison(_to_uint8(img_hwc), _to_uint8(img_hwc))
        return CounterfactualResult(
            hypothesis_id=f"object_relocation_{target_object}",
            test_type="object_relocation",
            action_delta_l2=0.0,
            action_delta_per_dim=[0.0] * len(baseline_actions),
            gradcam_shift=0.0,
            attribution_shift_per_region={},
            confirmed=False,
            metrics={"target_object": target_object, "shift_pixels": list(shift_pixels)},
            visual_comparison=comparison,
        )

    obj_mask = _resize_mask(obj_mask, (h, w))
    dx, dy = shift_pixels

    modified_hwc = img_hwc.copy()

    # Fill the original object region with the mean of neighbouring pixels (or gray)
    ys, xs = np.where(obj_mask)
    if len(ys) > 0:
        # Compute mean of pixels just outside the mask (dilation - mask)
        try:
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(obj_mask, iterations=5)
            border = dilated & ~obj_mask
            if border.any():
                fill_color = img_hwc[border].mean(axis=0)
            else:
                fill_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        except ImportError:
            fill_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        modified_hwc[obj_mask] = fill_color

        # Extract object pixels and paste at new location
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()

        for oy, ox in zip(ys, xs):
            ny = oy + dy
            nx = ox + dx
            if 0 <= ny < h and 0 <= nx < w:
                modified_hwc[ny, nx] = img_hwc[oy, ox]

    modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

    new_sample = _clone_sample(sample, image_key)
    new_sample[image_key] = _hwc_to_tensor(modified_hwc, device=str(img_tensor.device))

    policy.reset()
    modified_actions = _get_actions(policy, new_sample, dataset, image_key, device, image_map)

    # Probe whether causal focus follows the moved object or remains on the old anchor.
    original_cam = compute_gradcam_map(
        policy, sample, dataset, image_key, device, image_map=image_map,
    )
    modified_cam = compute_gradcam_map(
        policy, new_sample, dataset, image_key, device, image_map=image_map,
    )
    moved_mask = _shift_mask(obj_mask, (dx, dy))
    old_anchor_mask = obj_mask & ~moved_mask
    moved_target_mask = moved_mask & ~obj_mask
    if not old_anchor_mask.any():
        old_anchor_mask = obj_mask
    if not moved_target_mask.any():
        moved_target_mask = moved_mask

    original_target_share = _mask_share(original_cam, obj_mask)
    old_anchor_share = _mask_share(modified_cam, old_anchor_mask)
    moved_target_share = _mask_share(modified_cam, moved_target_mask)
    focus_total = old_anchor_share + moved_target_share + 1e-8
    metrics = {
        "target_object": target_object,
        "shift_pixels": [dx, dy],
        "original_target_gradcam_share": original_target_share,
        "modified_old_anchor_share": old_anchor_share,
        "modified_new_object_share": moved_target_share,
        "focus_follow_ratio": moved_target_share / focus_total,
        "anchor_retention_ratio": old_anchor_share / focus_total,
        "focus_shift_gap": moved_target_share - old_anchor_share,
    }

    original_semantics = compute_patch_text_similarity(
        policy, sample, dataset, image_key, device, [target_object], image_map=image_map,
    )
    modified_semantics = compute_patch_text_similarity(
        policy, new_sample, dataset, image_key, device, [target_object], image_map=image_map,
    )
    if original_semantics is not None and modified_semantics is not None:
        orig_map = original_semantics["similarity_maps"].get(target_object)
        mod_map = modified_semantics["similarity_maps"].get(target_object)
        if orig_map is not None and mod_map is not None:
            orig_valid = original_semantics["valid_patch_mask"]
            mod_valid = modified_semantics["valid_patch_mask"]
            orig_grid = original_semantics["grid_size"]
            mod_grid = modified_semantics["grid_size"]
            orig_peak = _semantic_region_peak(orig_map, obj_mask, orig_grid, orig_valid)
            old_anchor_sem = _semantic_region_mean(mod_map, old_anchor_mask, mod_grid, mod_valid)
            moved_target_sem = _semantic_region_mean(mod_map, moved_target_mask, mod_grid, mod_valid)
            mod_norm = _normalize_similarity_map(mod_map, mod_valid)
            old_anchor_norm = _semantic_region_mean(mod_norm, old_anchor_mask, mod_grid, mod_valid) or 0.0
            moved_target_norm = _semantic_region_mean(mod_norm, moved_target_mask, mod_grid, mod_valid) or 0.0
            semantic_total = old_anchor_norm + moved_target_norm + 1e-8
            metrics.update({
                "original_target_semantic_peak": orig_peak if orig_peak is not None else 0.0,
                "modified_old_anchor_target_semantic": (
                    old_anchor_sem if old_anchor_sem is not None else 0.0
                ),
                "modified_new_object_target_semantic": (
                    moved_target_sem if moved_target_sem is not None else 0.0
                ),
                "semantic_follow_ratio": moved_target_norm / semantic_total,
                "semantic_anchor_ratio": old_anchor_norm / semantic_total,
                "semantic_shift_gap": moved_target_norm - old_anchor_norm,
            })

    return _compute_result(
        baseline_actions, modified_actions,
        img_hwc, modified_hwc,
        hypothesis_id=f"object_relocation_{target_object}",
        test_type="object_relocation",
        affected_mask=obj_mask | moved_mask,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# 3. Lighting perturbation
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.lighting_perturbation",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Shift brightness and contrast of the image and measure action change.",
)
def lighting_perturbation(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    brightness_delta: float = 0.3,
    contrast_delta: float = 0.3,
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Apply brightness and contrast shifts and measure action change.

    Parameters
    ----------
    brightness_delta : float
        Additive brightness shift (applied before contrast).
    contrast_delta : float
        Multiplicative contrast factor: ``image * (1 + contrast_delta)``.
    """
    # Baseline actions
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    img_tensor = sample[image_key]  # (C, H, W) float [0, 1]
    img_hwc = _tensor_to_hwc(img_tensor)  # (H, W, C)

    # Apply brightness and contrast shift
    modified_hwc = img_hwc.copy()
    modified_hwc = modified_hwc + brightness_delta
    modified_hwc = modified_hwc * (1.0 + contrast_delta)
    modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

    new_sample = _clone_sample(sample, image_key)
    new_sample[image_key] = _hwc_to_tensor(modified_hwc, device=str(img_tensor.device))

    policy.reset()
    modified_actions = _get_actions(policy, new_sample, dataset, image_key, device, image_map)

    return _compute_result(
        baseline_actions, modified_actions,
        img_hwc, modified_hwc,
        hypothesis_id=f"lighting_b{brightness_delta}_c{contrast_delta}",
        test_type="lighting_perturbation",
    )


# ---------------------------------------------------------------------------
# 4. Object recolor
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.object_recolor",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Shift the hue of a target object and measure action change.",
)
def object_recolor(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    segmentation: SceneSegmentation,
    target_object: str,
    hue_shift: float = 0.3,
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Shift the hue of *target_object* by *hue_shift* and measure action change.

    The hue channel is in ``[0, 1]`` and wraps around.
    """
    # Baseline actions
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    img_tensor = sample[image_key]  # (C, H, W) float [0, 1]
    img_hwc = _tensor_to_hwc(img_tensor)  # (H, W, C)
    h, w = img_hwc.shape[:2]

    obj_mask = segmentation.get_mask(target_object)
    if obj_mask is None:
        print(f"  WARNING: No mask found for object '{target_object}', returning zero-delta result.")
        comparison = _make_comparison(_to_uint8(img_hwc), _to_uint8(img_hwc))
        return CounterfactualResult(
            hypothesis_id=f"object_recolor_{target_object}",
            test_type="object_recolor",
            action_delta_l2=0.0,
            action_delta_per_dim=[0.0] * len(baseline_actions),
            gradcam_shift=0.0,
            attribution_shift_per_region={},
            confirmed=False,
            metrics={"target_object": target_object, "hue_shift": hue_shift},
            visual_comparison=comparison,
        )

    obj_mask = _resize_mask(obj_mask, (h, w))

    modified_hwc = img_hwc.copy()

    # Convert masked region to HSV, shift hue, convert back.
    # If the object has low saturation (near-gray), also boost saturation
    # so the color change is actually visible and tests appearance sensitivity.
    _MIN_SATURATION = 0.5
    try:
        import matplotlib.colors as mcolors

        masked_rgb = modified_hwc[obj_mask]  # (N, 3) float [0, 1]
        hsv = mcolors.rgb_to_hsv(masked_rgb)
        hsv[:, 0] = (hsv[:, 0] + hue_shift) % 1.0
        # Ensure enough saturation for the hue change to be visible
        hsv[:, 1] = np.maximum(hsv[:, 1], _MIN_SATURATION)
        # Ensure enough brightness for the color to be visible
        hsv[:, 2] = np.maximum(hsv[:, 2], 0.4)
        rgb_shifted = mcolors.hsv_to_rgb(hsv)
        modified_hwc[obj_mask] = rgb_shifted
    except ImportError:
        import colorsys

        ys, xs = np.where(obj_mask)
        for y, x in zip(ys, xs):
            r, g, b = modified_hwc[y, x, 0], modified_hwc[y, x, 1], modified_hwc[y, x, 2]
            h_val, s_val, v_val = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            h_val = (h_val + hue_shift) % 1.0
            s_val = max(s_val, _MIN_SATURATION)
            v_val = max(v_val, 0.4)
            r2, g2, b2 = colorsys.hsv_to_rgb(h_val, s_val, v_val)
            modified_hwc[y, x] = [r2, g2, b2]

    modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

    new_sample = _clone_sample(sample, image_key)
    new_sample[image_key] = _hwc_to_tensor(modified_hwc, device=str(img_tensor.device))

    policy.reset()
    modified_actions = _get_actions(policy, new_sample, dataset, image_key, device, image_map)

    return _compute_result(
        baseline_actions, modified_actions,
        img_hwc, modified_hwc,
        hypothesis_id=f"object_recolor_{target_object}",
        test_type="object_recolor",
        affected_mask=obj_mask,
        metrics={"target_object": target_object, "hue_shift": hue_shift},
    )


# ---------------------------------------------------------------------------
# 5. Distractor insertion
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.distractor_insertion",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Insert a distractor object and measure action change.",
)
def distractor_insertion(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    segmentation: SceneSegmentation,
    position: tuple[int, int] = (100, 100),  # (x, y) center
    distractor_size: int = 80,
    distractor_source: str = "synthetic",  # "synthetic" or "noise"
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Insert a distractor at *position* and measure action shift.

    Parameters
    ----------
    position : tuple[int, int]
        ``(x, y)`` center of the distractor in pixel coordinates.
    distractor_size : int
        Approximate diameter (pixels) of the distractor.
    distractor_source : str
        ``"synthetic"`` generates a coloured ellipse; ``"noise"`` fills with
        random noise.
    """
    # Baseline actions
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    img_tensor = sample[image_key]  # (C, H, W) float [0, 1]
    img_hwc = _tensor_to_hwc(img_tensor)  # (H, W, C)
    h, w = img_hwc.shape[:2]

    rng = np.random.RandomState(noise_seed)

    # Build a binary mask for the distractor region (ellipse)
    cx, cy = position
    radius = distractor_size // 2
    yy, xx = np.ogrid[:h, :w]
    dist_sq = ((xx - cx).astype(np.float64)) ** 2 + ((yy - cy).astype(np.float64)) ** 2
    distractor_mask = dist_sq <= (radius ** 2)

    modified_hwc = img_hwc.copy()

    if distractor_source == "synthetic":
        # Generate a random solid colour and draw a filled ellipse
        color = rng.rand(3).astype(np.float32)
        modified_hwc[distractor_mask] = color
    elif distractor_source == "noise":
        noise_patch = rng.rand(h, w, img_hwc.shape[2]).astype(np.float32)
        modified_hwc[distractor_mask] = noise_patch[distractor_mask]
    else:
        raise ValueError(f"Unknown distractor_source: {distractor_source!r}")

    modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

    new_sample = _clone_sample(sample, image_key)
    new_sample[image_key] = _hwc_to_tensor(modified_hwc, device=str(img_tensor.device))

    policy.reset()
    modified_actions = _get_actions(policy, new_sample, dataset, image_key, device, image_map)

    return _compute_result(
        baseline_actions, modified_actions,
        img_hwc, modified_hwc,
        hypothesis_id=f"distractor_insertion_{distractor_source}",
        test_type="distractor_insertion",
        affected_mask=distractor_mask,
        metrics={
            "position": [cx, cy],
            "distractor_size": distractor_size,
            "distractor_source": distractor_source,
        },
    )


# ---------------------------------------------------------------------------
# 6. Task string swap
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.task_string_swap",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Replace task instruction and measure action change.",
)
def task_string_swap(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    replacement_task: str = "do nothing",
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Replace the task instruction with *replacement_task* and measure action shift.

    The image is kept identical; only the language conditioning changes.
    """
    # Baseline actions (original task string)
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    # Build a batch with the replacement task string
    batch, _ = build_policy_batch_from_sample(
        sample, policy, device, image_key_for_grad=None,
        dataset=dataset, image_map=image_map,
        task_override=replacement_task,
    )

    policy.reset()
    with torch.no_grad(), _patch_eager_attention_bool_mask(policy):
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        bsize = state.shape[0]
        actions_shape = (
            bsize,
            policy.model.config.chunk_size,
            policy.model.config.max_action_dim,
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(noise_seed)
        noise = torch.randn(
            actions_shape, device=device, generator=gen, dtype=torch.float32,
        )
        actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )

    modified_actions = actions[0, 0].cpu().numpy()

    # Resolve original task string for labelling
    from ..data import _resolve_task_string
    original_task = _resolve_task_string(sample, dataset)

    # Visual: per-dimension action delta bar chart (no image comparison —
    # the perturbation is purely linguistic)
    comparison = _make_action_delta_chart(
        baseline_actions, modified_actions,
        original_task, replacement_task,
    )

    delta = modified_actions - baseline_actions
    delta_l2 = float(np.linalg.norm(delta))

    return CounterfactualResult(
        hypothesis_id="task_string_swap",
        test_type="task_string_swap",
        action_delta_l2=delta_l2,
        action_delta_per_dim=delta.tolist(),
        gradcam_shift=0.0,
        attribution_shift_per_region={},
        confirmed=delta_l2 > 0.01,
        visual_comparison=comparison,
    )


# ---------------------------------------------------------------------------
# 7. Temporal consistency
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.temporal_consistency",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Apply perturbation across multiple frames and check action sequence coherence.",
)
def temporal_consistency(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    segmentation: SceneSegmentation,
    perturbation_type: str = "background_substitution",
    num_frames: int = 5,
    episode_idx: int = 0,
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Apply the same perturbation across episode frames and measure coherence.

    For each frame the baseline and perturbed actions are computed.  The
    standard deviation of per-frame action deltas across the episode indicates
    how consistently the model responds to the same perturbation — high
    variance suggests unstable or inconsistent behaviour.

    Parameters
    ----------
    perturbation_type : str
        Currently only ``"background_substitution"`` is supported.
    num_frames : int
        Number of evenly-spaced frames to sample from the episode.
    episode_idx : int
        Episode to sample frames from.
    """
    from ..data import get_episode_frames

    frames = get_episode_frames(dataset, episode_idx, num_frames, image_key)
    bg_mask_raw = segmentation.background_mask

    rng = np.random.RandomState(noise_seed)

    per_frame_deltas: list[np.ndarray] = []
    first_img_hwc: np.ndarray | None = None
    first_modified_hwc: np.ndarray | None = None
    first_affected_mask: np.ndarray | None = None

    for frame_idx, frame_img_tensor in frames:
        # Build a sample dict for this frame by copying the original and
        # replacing the image tensor.
        frame_sample = dict(sample)
        frame_sample[image_key] = frame_img_tensor

        # Baseline
        policy.reset()
        baseline_actions = _get_actions(
            policy, frame_sample, dataset, image_key, device, image_map,
        )

        # Perturbed
        img_hwc = _tensor_to_hwc(frame_img_tensor)
        h, w = img_hwc.shape[:2]
        bg_mask = _resize_mask(bg_mask_raw, (h, w))

        modified_hwc = img_hwc.copy()

        if perturbation_type == "background_substitution":
            modified_hwc[bg_mask] = 0.5
        else:
            raise ValueError(
                f"Unsupported perturbation_type for temporal_consistency: "
                f"{perturbation_type!r}"
            )

        modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

        perturbed_sample = _clone_sample(frame_sample, image_key)
        perturbed_sample[image_key] = _hwc_to_tensor(
            modified_hwc, device=str(frame_img_tensor.device),
        )

        policy.reset()
        modified_actions = _get_actions(
            policy, perturbed_sample, dataset, image_key, device, image_map,
        )

        delta = modified_actions - baseline_actions
        per_frame_deltas.append(delta)

        # Keep the first frame's visuals for the comparison image
        if first_img_hwc is None:
            first_img_hwc = img_hwc
            first_modified_hwc = modified_hwc
            first_affected_mask = bg_mask

    # Aggregate: mean L2 delta and std of deltas across frames
    deltas_array = np.stack(per_frame_deltas, axis=0)  # (num_frames, action_dim)
    mean_delta = deltas_array.mean(axis=0)
    mean_delta_l2 = float(np.linalg.norm(mean_delta))
    std_across_frames = deltas_array.std(axis=0)  # per-dim std

    comparison = _make_comparison(
        _to_uint8(first_img_hwc),
        _to_uint8(first_modified_hwc),
        mask=first_affected_mask,
    )

    return CounterfactualResult(
        hypothesis_id=f"temporal_consistency_{perturbation_type}",
        test_type="temporal_consistency",
        action_delta_l2=mean_delta_l2,
        action_delta_per_dim=std_across_frames.tolist(),
        gradcam_shift=0.0,
        attribution_shift_per_region={},
        confirmed=mean_delta_l2 > 0.01,
        visual_comparison=comparison,
    )


# ---------------------------------------------------------------------------
# 8. Occlusion targeted
# ---------------------------------------------------------------------------

@register_primitive(
    "counterfactual.occlusion_targeted",
    category="counterfactual",
    cost="expensive",
    requires_gpu=True,
    requires_model=True,
    description="Completely occlude a target object and measure action change.",
)
def occlusion_targeted(
    policy,
    sample: dict,
    dataset,
    image_key: str,
    device: str | torch.device,
    segmentation: SceneSegmentation,
    target_object: str,
    fill: str = "gray",  # "gray" or "noise"
    noise_seed: int = 42,
    image_map=None,
) -> CounterfactualResult:
    """Completely occlude *target_object* with a uniform fill and measure action shift.

    Unlike ``object_recolor`` (which changes appearance while preserving
    shape), this primitive removes **all** visual information about the
    object, testing whether the model truly relies on it.

    Parameters
    ----------
    fill : str
        ``"gray"`` fills the masked region with mid-gray (0.5);
        ``"noise"`` fills with random noise.
    """
    # Baseline actions
    policy.reset()
    baseline_actions = _get_actions(policy, sample, dataset, image_key, device, image_map)

    img_tensor = sample[image_key]  # (C, H, W) float [0, 1]
    img_hwc = _tensor_to_hwc(img_tensor)  # (H, W, C)
    h, w = img_hwc.shape[:2]

    obj_mask = segmentation.get_mask(target_object)
    if obj_mask is None:
        print(
            f"  WARNING: No mask found for object '{target_object}', "
            f"returning zero-delta result."
        )
        comparison = _make_comparison(_to_uint8(img_hwc), _to_uint8(img_hwc))
        return CounterfactualResult(
            hypothesis_id=f"occlusion_targeted_{target_object}",
            test_type="occlusion_targeted",
            action_delta_l2=0.0,
            action_delta_per_dim=[0.0] * len(baseline_actions),
            gradcam_shift=0.0,
            attribution_shift_per_region={},
            confirmed=False,
            metrics={"target_object": target_object, "fill": fill},
            visual_comparison=comparison,
        )

    obj_mask = _resize_mask(obj_mask, (h, w))

    modified_hwc = img_hwc.copy()

    if fill == "gray":
        modified_hwc[obj_mask] = 0.5
    elif fill == "noise":
        rng = np.random.RandomState(noise_seed)
        noise_patch = rng.rand(h, w, img_hwc.shape[2]).astype(np.float32)
        modified_hwc[obj_mask] = noise_patch[obj_mask]
    else:
        raise ValueError(f"Unknown fill mode: {fill!r}")

    modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

    new_sample = _clone_sample(sample, image_key)
    new_sample[image_key] = _hwc_to_tensor(modified_hwc, device=str(img_tensor.device))

    policy.reset()
    modified_actions = _get_actions(policy, new_sample, dataset, image_key, device, image_map)

    metrics = {"target_object": target_object, "fill": fill}
    original_semantics = compute_patch_text_similarity(
        policy, sample, dataset, image_key, device, [target_object], image_map=image_map,
    )
    occluded_semantics = compute_patch_text_similarity(
        policy, new_sample, dataset, image_key, device, [target_object], image_map=image_map,
    )
    if original_semantics is not None and occluded_semantics is not None:
        orig_map = original_semantics["similarity_maps"].get(target_object)
        occ_map = occluded_semantics["similarity_maps"].get(target_object)
        if orig_map is not None and occ_map is not None:
            orig_valid = original_semantics["valid_patch_mask"]
            occ_valid = occluded_semantics["valid_patch_mask"]
            orig_grid = original_semantics["grid_size"]
            occ_grid = occluded_semantics["grid_size"]
            orig_peak = _semantic_region_peak(orig_map, obj_mask, orig_grid, orig_valid)
            occ_peak = _semantic_region_peak(occ_map, obj_mask, occ_grid, occ_valid)
            if orig_peak is not None and occ_peak is not None:
                metrics.update({
                    "original_target_semantic_peak": orig_peak,
                    "occluded_target_semantic_peak": occ_peak,
                    "occlusion_target_semantic_drop": orig_peak - occ_peak,
                })

    return _compute_result(
        baseline_actions, modified_actions,
        img_hwc, modified_hwc,
        hypothesis_id=f"occlusion_targeted_{target_object}",
        test_type="occlusion_targeted",
        affected_mask=obj_mask,
        metrics=metrics,
    )

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


def _make_comparison(original_img: np.ndarray, modified_img: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison. Both are (H, W, 3) uint8 numpy arrays."""
    return np.concatenate([original_img, modified_img], axis=1)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float [0,1] (H, W, C) image to uint8."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers: forward pass and mask resizing
# ---------------------------------------------------------------------------

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
) -> CounterfactualResult:
    """Compare baseline and modified actions, build a CounterfactualResult."""
    delta = modified_actions - baseline_actions
    delta_l2 = float(np.linalg.norm(delta))
    delta_per_dim = delta.tolist()

    comparison = _make_comparison(
        _to_uint8(original_img_hwc),
        _to_uint8(modified_img_hwc),
    )

    return CounterfactualResult(
        hypothesis_id=hypothesis_id,
        test_type=test_type,
        action_delta_l2=delta_l2,
        action_delta_per_dim=delta_per_dim,
        gradcam_shift=0.0,
        attribution_shift_per_region={},
        confirmed=delta_l2 > 0.01,
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

    return _compute_result(
        baseline_actions, modified_actions,
        img_hwc, modified_hwc,
        hypothesis_id=f"object_relocation_{target_object}",
        test_type="object_relocation",
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
            visual_comparison=comparison,
        )

    obj_mask = _resize_mask(obj_mask, (h, w))

    modified_hwc = img_hwc.copy()

    # Convert masked region to HSV, shift hue, convert back
    # Use colorsys per-pixel if matplotlib is not available
    try:
        import matplotlib.colors as mcolors

        # Extract masked pixels as (N, 3)
        masked_rgb = modified_hwc[obj_mask]  # (N, 3) float [0, 1]
        hsv = mcolors.rgb_to_hsv(masked_rgb)
        hsv[:, 0] = (hsv[:, 0] + hue_shift) % 1.0
        rgb_shifted = mcolors.hsv_to_rgb(hsv)
        modified_hwc[obj_mask] = rgb_shifted
    except ImportError:
        import colorsys

        ys, xs = np.where(obj_mask)
        for y, x in zip(ys, xs):
            r, g, b = modified_hwc[y, x, 0], modified_hwc[y, x, 1], modified_hwc[y, x, 2]
            h_val, s_val, v_val = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            h_val = (h_val + hue_shift) % 1.0
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
    )

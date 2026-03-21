"""
Image augmentation transforms for training data generation.

All transforms operate on (H, W, C) float32 numpy arrays in [0, 1] range.
Each transform accepts a config dict and an optional background mask (H, W) bool.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) float RGB [0,1] to HSV [0,1]."""
    out = np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = colorsys.rgb_to_hsv(img[y, x, 0], img[y, x, 1], img[y, x, 2])
    return out


def _hsv_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) float HSV [0,1] to RGB [0,1]."""
    out = np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = colorsys.hsv_to_rgb(img[y, x, 0], img[y, x, 1], img[y, x, 2])
    return out


def _fast_rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Vectorized RGB→HSV. Input/output (H,W,3) float32 in [0,1]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    # Value
    v = maxc

    # Saturation
    s = np.where(maxc > 0, diff / (maxc + 1e-10), 0.0)

    # Hue
    h = np.zeros_like(maxc)
    mask_r = (maxc == r) & (diff > 0)
    mask_g = (maxc == g) & (diff > 0) & ~mask_r
    mask_b = (diff > 0) & ~mask_r & ~mask_g

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (diff[mask_r] + 1e-10)) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (diff[mask_g] + 1e-10)) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (diff[mask_b] + 1e-10)) + 4.0
    h = h / 6.0  # normalize to [0, 1]

    return np.stack([h, s, v], axis=-1).astype(np.float32)


def _fast_hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorized HSV→RGB. Input/output (H,W,3) float32 in [0,1]."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _load_background_images(directory: str | Path) -> list[np.ndarray]:
    """Load all images from a directory as float32 HWC arrays."""
    from PIL import Image

    bg_dir = Path(directory)
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for p in sorted(bg_dir.glob(ext)):
            img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
            images.append(img)
    if not images:
        raise FileNotFoundError(f"No background images found in {directory}")
    return images


# ---------------------------------------------------------------------------
# Background replacement
# ---------------------------------------------------------------------------

def apply_background_replacement(
    img: np.ndarray,
    bg_mask: np.ndarray,
    cfg: dict,
    rng: np.random.RandomState,
    bg_images: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Replace background pixels according to the configured strategy.

    Parameters
    ----------
    img : (H, W, C) float32 in [0, 1]
    bg_mask : (H, W) bool — True where background
    cfg : background_replacement config dict
    rng : seeded RNG
    bg_images : pre-loaded background images (for "image_bank" strategy)
    """
    if not cfg.get("enabled", False):
        return img

    strategy = cfg.get("strategy", "noise")

    # Mix mode: randomly pick a strategy per frame based on weights
    if strategy == "mix":
        mix_entries = cfg.get("mix", [])
        if not mix_entries:
            raise ValueError("strategy 'mix' requires a 'mix' list with strategy/weight entries")
        strategies = [e["strategy"] for e in mix_entries]
        weights = np.array([e.get("weight", 1.0) for e in mix_entries], dtype=np.float64)
        weights /= weights.sum()
        chosen = strategies[rng.choice(len(strategies), p=weights)]
        # Build a single-strategy config and recurse
        single_cfg = {**cfg, "strategy": chosen}
        return apply_background_replacement(img, bg_mask, single_cfg, rng, bg_images)

    out = img.copy()
    h, w = img.shape[:2]

    if strategy == "gray":
        value = cfg.get("gray", {}).get("value", 0.5)
        out[bg_mask] = value

    elif strategy == "noise":
        params = cfg.get("noise", {})
        lo, hi = params.get("low", 0.0), params.get("high", 1.0)
        noise = rng.uniform(lo, hi, (h, w, img.shape[2])).astype(np.float32)
        out[bg_mask] = noise[bg_mask]

    elif strategy == "blur":
        sigma = cfg.get("blur", {}).get("sigma", 20.0)
        try:
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(img, sigma=(sigma, sigma, 0))
        except ImportError:
            # Box-blur fallback
            kernel_size = int(sigma * 2) | 1  # ensure odd
            pad = kernel_size // 2
            padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            blurred = np.zeros_like(img)
            for c in range(img.shape[2]):
                cumsum = np.cumsum(np.cumsum(padded[:, :, c], axis=0), axis=1)
                blurred[:, :, c] = (
                    cumsum[kernel_size:, kernel_size:]
                    - cumsum[:-kernel_size, kernel_size:]
                    - cumsum[kernel_size:, :-kernel_size]
                    + cumsum[:-kernel_size, :-kernel_size]
                ) / (kernel_size * kernel_size)
            blurred = blurred[:h, :w]
        out[bg_mask] = blurred[bg_mask]

    elif strategy == "image_bank":
        if bg_images is None or len(bg_images) == 0:
            raise ValueError("image_bank strategy requires bg_images to be loaded")
        params = cfg.get("image_bank", {})
        resize_mode = params.get("resize_mode", "crop")
        chosen = bg_images[rng.randint(len(bg_images))]
        bg_patch = _resize_bg_image(chosen, h, w, resize_mode, rng)
        out[bg_mask] = bg_patch[bg_mask]

    else:
        raise ValueError(f"Unknown background replacement strategy: {strategy!r}")

    return np.clip(out, 0.0, 1.0)


def _resize_bg_image(
    bg_img: np.ndarray, target_h: int, target_w: int,
    mode: str, rng: np.random.RandomState,
) -> np.ndarray:
    """Resize a background image to target dimensions."""
    from PIL import Image

    bh, bw = bg_img.shape[:2]

    if mode == "crop":
        # Random crop that covers target aspect ratio, then resize
        target_ratio = target_w / target_h
        bg_ratio = bw / bh
        if bg_ratio > target_ratio:
            # bg is wider — crop width
            new_bw = int(bh * target_ratio)
            x_start = rng.randint(0, max(bw - new_bw, 1))
            bg_img = bg_img[:, x_start:x_start + new_bw]
        else:
            # bg is taller — crop height
            new_bh = int(bw / target_ratio)
            y_start = rng.randint(0, max(bh - new_bh, 1))
            bg_img = bg_img[y_start:y_start + new_bh, :]

    # Resize to target
    pil_img = Image.fromarray((bg_img * 255).astype(np.uint8))
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil_img).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Color jitter
# ---------------------------------------------------------------------------

def apply_color_jitter(
    img: np.ndarray,
    bg_mask: np.ndarray | None,
    cfg: dict,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply color jitter (brightness, contrast, saturation, hue).

    Parameters
    ----------
    img : (H, W, C) float32 in [0, 1]
    bg_mask : (H, W) bool or None
    cfg : color_jitter config dict
    rng : seeded RNG
    """
    if not cfg.get("enabled", False):
        return img

    out = img.copy()
    target = cfg.get("target", "full")

    # Sample jitter parameters
    brightness = rng.uniform(-cfg.get("brightness", 0.0), cfg.get("brightness", 0.0))
    contrast = rng.uniform(max(0, 1 - cfg.get("contrast", 0.0)), 1 + cfg.get("contrast", 0.0))
    sat_factor = rng.uniform(max(0, 1 - cfg.get("saturation", 0.0)), 1 + cfg.get("saturation", 0.0))
    hue_delta = rng.uniform(-cfg.get("hue", 0.0), cfg.get("hue", 0.0))

    # Build mask for which pixels to affect
    if target == "foreground" and bg_mask is not None:
        mask = ~bg_mask
    elif target == "background" and bg_mask is not None:
        mask = bg_mask
    else:
        mask = np.ones(img.shape[:2], dtype=bool)

    # Apply brightness
    region = out.copy()
    region[mask] = np.clip(region[mask] + brightness, 0, 1)

    # Apply contrast (relative to mean)
    mean = region[mask].mean() if mask.any() else 0.5
    region[mask] = np.clip((region[mask] - mean) * contrast + mean, 0, 1)

    # Apply saturation + hue in HSV space
    if abs(sat_factor - 1.0) > 1e-4 or abs(hue_delta) > 1e-4:
        hsv = _fast_rgb_to_hsv(region)
        hsv[mask, 0] = (hsv[mask, 0] + hue_delta) % 1.0
        hsv[mask, 1] = np.clip(hsv[mask, 1] * sat_factor, 0, 1)
        region = _fast_hsv_to_rgb(hsv)

    out[mask] = region[mask]
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

def apply_gaussian_blur(
    img: np.ndarray,
    cfg: dict,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply random Gaussian blur to the full image."""
    if not cfg.get("enabled", False):
        return img

    if rng.rand() > cfg.get("probability", 0.5):
        return img

    k_range = cfg.get("kernel_size_range", [5, 15])
    sigma_range = cfg.get("sigma_range", [0.5, 2.0])

    sigma = rng.uniform(sigma_range[0], sigma_range[1])

    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=(sigma, sigma, 0)).astype(np.float32)
    except ImportError:
        kernel_size = rng.randint(k_range[0] // 2, k_range[1] // 2 + 1) * 2 + 1
        pad = kernel_size // 2
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        blurred = np.zeros_like(img)
        h, w = img.shape[:2]
        for c in range(img.shape[2]):
            cumsum = np.cumsum(np.cumsum(padded[:, :, c], axis=0), axis=1)
            blurred[:, :, c] = (
                cumsum[kernel_size:, kernel_size:]
                - cumsum[:-kernel_size, kernel_size:]
                - cumsum[kernel_size:, :-kernel_size]
                + cumsum[:-kernel_size, :-kernel_size]
            ) / (kernel_size * kernel_size)
        return blurred[:h, :w].astype(np.float32)


# ---------------------------------------------------------------------------
# Random crop + resize
# ---------------------------------------------------------------------------

def apply_random_crop(
    img: np.ndarray,
    cfg: dict,
    rng: np.random.RandomState,
    bg_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random crop and resize back to original size. Returns (img, updated_bg_mask)."""
    if not cfg.get("enabled", False):
        return img, bg_mask

    from PIL import Image

    h, w = img.shape[:2]
    scale_range = cfg.get("scale_range", [0.8, 1.0])
    ratio_range = cfg.get("ratio_range", [0.9, 1.1])

    area = h * w
    target_area = rng.uniform(scale_range[0], scale_range[1]) * area
    aspect = rng.uniform(ratio_range[0], ratio_range[1])

    crop_h = int(np.sqrt(target_area / aspect))
    crop_w = int(np.sqrt(target_area * aspect))
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    y0 = rng.randint(0, max(h - crop_h, 1))
    x0 = rng.randint(0, max(w - crop_w, 1))

    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
    pil_img = pil_img.resize((w, h), Image.BILINEAR)
    out = np.array(pil_img).astype(np.float32) / 255.0

    # Also crop/resize the mask if provided
    out_mask = None
    if bg_mask is not None:
        mask_cropped = bg_mask[y0:y0 + crop_h, x0:x0 + crop_w]
        mask_pil = Image.fromarray(mask_cropped.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        out_mask = np.array(mask_pil) > 127

    return out, out_mask


# ---------------------------------------------------------------------------
# Foreground cutout
# ---------------------------------------------------------------------------

def apply_foreground_cutout(
    img: np.ndarray,
    bg_mask: np.ndarray | None,
    cfg: dict,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Randomly zero out a fraction of foreground pixels."""
    if not cfg.get("enabled", False):
        return img
    if bg_mask is None:
        return img

    if rng.rand() > cfg.get("probability", 0.3):
        return img

    fg_mask = ~bg_mask
    fg_pixels = np.argwhere(fg_mask)
    if len(fg_pixels) == 0:
        return img

    max_frac = cfg.get("max_fraction", 0.4)
    n_cut = int(len(fg_pixels) * rng.uniform(0, max_frac))
    if n_cut == 0:
        return img

    indices = rng.choice(len(fg_pixels), n_cut, replace=False)
    out = img.copy()
    coords = fg_pixels[indices]
    out[coords[:, 0], coords[:, 1]] = 0.0
    return out


# ---------------------------------------------------------------------------
# Background color shift
# ---------------------------------------------------------------------------

def apply_background_color_shift(
    img: np.ndarray,
    bg_mask: np.ndarray | None,
    cfg: dict,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Shift hue/saturation/value of background pixels."""
    if not cfg.get("enabled", False):
        return img
    if bg_mask is None or not bg_mask.any():
        return img

    hue_range = cfg.get("hue_range", [-0.15, 0.15])
    sat_range = cfg.get("saturation_range", [0.7, 1.3])
    val_range = cfg.get("value_range", [0.7, 1.3])

    hue_shift = rng.uniform(hue_range[0], hue_range[1])
    sat_mult = rng.uniform(sat_range[0], sat_range[1])
    val_mult = rng.uniform(val_range[0], val_range[1])

    hsv = _fast_rgb_to_hsv(img)
    out_hsv = hsv.copy()
    out_hsv[bg_mask, 0] = (hsv[bg_mask, 0] + hue_shift) % 1.0
    out_hsv[bg_mask, 1] = np.clip(hsv[bg_mask, 1] * sat_mult, 0, 1)
    out_hsv[bg_mask, 2] = np.clip(hsv[bg_mask, 2] * val_mult, 0, 1)

    out = _fast_hsv_to_rgb(out_hsv)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def apply_noise_injection(
    img: np.ndarray,
    cfg: dict,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Add sensor noise (Gaussian or salt-and-pepper)."""
    if not cfg.get("enabled", False):
        return img

    if rng.rand() > cfg.get("probability", 0.5):
        return img

    noise_type = cfg.get("type", "gaussian")

    if noise_type == "gaussian":
        std = cfg.get("gaussian", {}).get("std", 0.02)
        noise = rng.randn(*img.shape).astype(np.float32) * std
        return np.clip(img + noise, 0.0, 1.0)

    elif noise_type == "salt_pepper":
        amount = cfg.get("salt_pepper", {}).get("amount", 0.01)
        out = img.copy()
        n_pixels = int(amount * img.shape[0] * img.shape[1])
        # Salt
        coords = (rng.randint(0, img.shape[0], n_pixels),
                  rng.randint(0, img.shape[1], n_pixels))
        out[coords[0], coords[1]] = 1.0
        # Pepper
        coords = (rng.randint(0, img.shape[0], n_pixels),
                  rng.randint(0, img.shape[1], n_pixels))
        out[coords[0], coords[1]] = 0.0
        return out

    return img


# ---------------------------------------------------------------------------
# Compose pipeline
# ---------------------------------------------------------------------------

def augment_image(
    img: np.ndarray,
    bg_mask: np.ndarray | None,
    config: dict,
    rng: np.random.RandomState,
    bg_images: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Apply the full augmentation pipeline to a single image.

    Parameters
    ----------
    img : (H, W, C) float32 in [0, 1]
    bg_mask : (H, W) bool or None
    config : full augmentation config dict
    rng : seeded RNG instance
    bg_images : pre-loaded background bank images

    Returns
    -------
    Augmented image (H, W, C) float32 in [0, 1]
    """
    # 1. Random crop (must happen first — changes spatial layout + mask)
    img, bg_mask = apply_random_crop(
        img, config.get("random_crop", {}), rng, bg_mask,
    )

    # 2. Background replacement
    if bg_mask is not None:
        img = apply_background_replacement(
            img, bg_mask, config.get("background_replacement", {}), rng, bg_images,
        )

    # 3. Background color shift
    img = apply_background_color_shift(
        img, bg_mask, config.get("background_color_shift", {}), rng,
    )

    # 4. Color jitter
    img = apply_color_jitter(
        img, bg_mask, config.get("color_jitter", {}), rng,
    )

    # 5. Foreground cutout
    img = apply_foreground_cutout(
        img, bg_mask, config.get("foreground_cutout", {}), rng,
    )

    # 6. Gaussian blur
    img = apply_gaussian_blur(img, config.get("gaussian_blur", {}), rng)

    # 7. Noise injection (last — additive noise on final image)
    img = apply_noise_injection(img, config.get("noise_injection", {}), rng)

    return img

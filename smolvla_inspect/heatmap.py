"""
Attention-to-heatmap conversion and positional baseline computation.

Imports ``resize_with_pad`` from ``_compat``.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from ._compat import resize_with_pad


def compute_padding_patches(input_hw, target_size, patch_size):
    """Number of pure-padding patch rows (top) and columns (left)
    produced by resize_with_pad for the given input dimensions."""
    if input_hw is None:
        return (0, 0)
    in_h, in_w = input_hw
    ratio = max(in_w / target_size, in_h / target_size)
    resized_h = int(in_h / ratio)
    resized_w = int(in_w / ratio)
    pad_h = max(0, target_size - resized_h)
    pad_w = max(0, target_size - resized_w)
    return (pad_h // patch_size, pad_w // patch_size)


def attention_to_heatmap(attn_weights, grid_size, image_size, content_crop=None,
                         threshold_pct=0.0):
    """
    Convert attention weights from patch-space to pixel-space heatmap.

    Args:
        attn_weights: Tensor of shape (num_patches,) or (H_patches, W_patches)
                      representing per-patch attention scores
        grid_size: (H_patches, W_patches) — the patch grid dimensions
        image_size: (H_pixels, W_pixels) — the original image dimensions
        content_crop: Optional (crop_h, crop_w) to remove padding patches
        threshold_pct: Percentile threshold (0.0–1.0). Values below this
                       percentile are zeroed, then the remainder is
                       re-normalised to [0, 1].  0.0 = no thresholding.

    Returns:
        heatmap: numpy array of shape (H_pixels, W_pixels) normalized to [0, 1]
    """
    h_patches, w_patches = grid_size
    h_img, w_img = image_size

    # Reshape to 2D grid if flat
    if attn_weights.dim() == 1:
        expected = h_patches * w_patches
        if attn_weights.shape[0] != expected:
            # Try to infer square grid
            side = int(math.sqrt(attn_weights.shape[0]))
            if side * side == attn_weights.shape[0]:
                h_patches, w_patches = side, side
            else:
                # Truncate or pad
                attn_weights = attn_weights[:expected]

        attn_2d = attn_weights.reshape(h_patches, w_patches)
    else:
        attn_2d = attn_weights

    # Crop out pure-padding patches (top rows, left columns) before upsampling
    if content_crop is not None:
        crop_h, crop_w = content_crop
        if crop_h > 0 or crop_w > 0:
            attn_2d = attn_2d[crop_h:, crop_w:]

    # Upsample to image resolution using bilinear interpolation
    attn_2d = attn_2d.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    heatmap = F.interpolate(attn_2d, size=(h_img, w_img), mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze().numpy()

    # Normalize to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)

    # Percentile thresholding: zero out diffuse low-attention noise
    if threshold_pct > 0:
        thresh_val = np.percentile(heatmap, threshold_pct * 100)
        heatmap = np.where(heatmap >= thresh_val, heatmap, 0.0)
        hmax = heatmap.max()
        if hmax > 0:
            heatmap = heatmap / hmax

    return heatmap


def compute_patch_attention_scores(attn_weights, method="mean"):
    """
    Reduce full attention matrix to per-patch importance scores.

    SigLIP is a pure patch encoder with no CLS token, so we use
    column-wise mean attention (how much attention each patch receives
    from all other patches) as the default aggregation.

    Args:
        attn_weights: Tensor, typically (num_patches, num_patches) or
                      (heads, num_patches, num_patches)
        method: How to aggregate:
            - "mean": Average attention each patch receives from all others

    Returns:
        scores: Tensor of shape (num_patches,)
    """
    # If multi-head, average across heads first
    if attn_weights.dim() == 3:
        attn_weights = attn_weights.mean(dim=0)  # (patches, patches)

    if attn_weights.dim() == 1:
        return attn_weights  # Already reduced

    # Column-wise mean = how much attention each patch receives
    scores = attn_weights.mean(dim=0)

    return scores


def compute_attention_rollout(all_layer_attentions):
    """
    Combine attention across all encoder layers using attention rollout.

    The method accounts for residual connections by mixing each layer's
    attention with an identity matrix (the residual stream keeps a copy of
    each token unchanged):

        rollout = I
        for A in layers:
            A_hat = 0.5 * I + 0.5 * A   # residual connection
            rollout = rollout @ A_hat
        normalize each row to sum to 1

    Args:
        all_layer_attentions: list of (layer_idx, attn_weights) tuples.
            Each attn_weights tensor has shape (batch, heads, patches, patches)
            or (heads, patches, patches) or (patches, patches).

    Returns:
        rollout: Tensor of shape (patches, patches) — the accumulated
                 attention from input patches to output patches.
    """
    matrices = []
    for _layer_idx, attn in sorted(all_layer_attentions, key=lambda x: x[0]):
        # Reduce to (patches, patches)
        while attn.dim() > 3:
            attn = attn[0]  # drop batch
        if attn.dim() == 3:
            attn = attn.mean(dim=0)  # average heads
        matrices.append(attn.float())

    if not matrices:
        return None

    n = matrices[0].shape[0]
    eye = torch.eye(n)
    rollout = eye.clone()

    for A in matrices:
        A_hat = 0.5 * eye + 0.5 * A
        rollout = rollout @ A_hat

    # Row-normalise so each row sums to 1
    rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-8)

    return rollout


def compute_positional_baseline(vision_encoder, attn_capture, device, method,
                                input_hw=None):
    """
    Compute the attention pattern produced by a content-free (mean-gray)
    image.  This captures the fixed positional component of attention so
    it can be subtracted from real frames to reveal the content-dependent
    signal.

    Args:
        vision_encoder: The SigLIP vision encoder module.
        attn_capture: A ``SigLIPAttentionCapture`` instance with hooks
            already registered on *vision_encoder*.
        device: Torch device.
        method: ``"last-layer"`` or ``"rollout"`` — same aggregation used
            for real frames so the baseline is comparable.
        input_hw: Optional ``(H, W)`` of the original input images.  When
            provided the baseline gray image is built at this aspect ratio
            then preprocessed with ``resize_with_pad`` so padding patches
            match those in real frames.

    Returns:
        ``(baseline_scores, per_head_baseline)`` where *baseline_scores* has
        shape ``(num_patches,)`` and *per_head_baseline* has shape
        ``(heads, num_patches)`` (or *None* if unavailable).
    """
    import torch

    # Build a mean-gray image at the encoder's expected resolution
    img_size = getattr(
        getattr(vision_encoder, "config", None), "image_size", None,
    ) or getattr(vision_encoder, "image_size", 384)

    if input_hw is not None:
        # Build gray at original aspect ratio, then resize_with_pad
        # so padding patches match the real frames exactly.
        in_h, in_w = input_hw
        gray_content = torch.full((1, 3, in_h, in_w), 0.5, device=device)
        gray = resize_with_pad(gray_content, img_size, img_size, pad_value=0)
        gray = gray * 2.0 - 1.0  # normalize to [-1, 1] matching SigLIP
    else:
        gray = torch.full((1, 3, img_size, img_size), 0.5, device=device)
    try:
        enc_dtype = next(vision_encoder.parameters()).dtype
        gray = gray.to(enc_dtype)
    except StopIteration:
        pass

    patch_size = getattr(vision_encoder, "patch_size", None) or getattr(
        getattr(vision_encoder, "config", None), "patch_size", 14,
    )
    n_patches_h = img_size // patch_size
    n_patches_w = img_size // patch_size
    patch_mask = torch.ones(1, n_patches_h, n_patches_w, dtype=torch.bool, device=device)

    # Forward pass through the vision encoder
    attn_capture.reset_maps()
    with torch.no_grad():
        try:
            if hasattr(vision_encoder, "embeddings") and hasattr(vision_encoder, "encoder"):
                embeddings = vision_encoder.embeddings(gray, patch_mask)
                vision_encoder.encoder(embeddings)
            elif hasattr(vision_encoder, "forward"):
                vision_encoder(gray)
            else:
                vision_encoder(pixel_values=gray, patch_attention_mask=patch_mask)
        except Exception as e:
            print(f"  WARNING: Baseline forward pass failed ({e}), skipping correction")
            return None, None

    # Capture per-head baseline from the last layer before aggregation
    per_head_baseline = None
    last_layer_attn = attn_capture.get_last_layer_attention()
    if last_layer_attn is not None:
        head_attn = last_layer_attn
        while head_attn.dim() > 3:
            head_attn = head_attn[0]
        # head_attn: (heads, patches, patches)
        per_head_baseline = head_attn.mean(dim=-2)  # (heads, patches)

    # Reduce captured attention using the same method as real frames
    if method == "rollout":
        all_layers = attn_capture.get_all_layer_attentions()
        rollout_mat = compute_attention_rollout(all_layers)
        if rollout_mat is None:
            return None, per_head_baseline
        baseline_scores = rollout_mat.mean(dim=0)
    else:
        # "last-layer" or "all-layers" both use last-layer for the summary
        if last_layer_attn is None:
            return None, None
        attn = last_layer_attn
        while attn.dim() > 3:
            attn = attn[0]
        if attn.dim() == 3:
            attn = attn.mean(dim=0)
        baseline_scores = compute_patch_attention_scores(attn, method="mean")

    attn_capture.reset_maps()
    return baseline_scores, per_head_baseline

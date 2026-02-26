"""
Visualization functions — overlays, grids, per-head attention maps.

Imports from ``heatmap``.
"""

import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from .heatmap import attention_to_heatmap

_CYAN_CMAP = LinearSegmentedColormap.from_list("cyan", ["black", "cyan", "white"])


def overlay_heatmap(image_np, heatmap, alpha=0.5, colormap="jet"):
    """
    Overlay a heatmap on an image.

    Args:
        image_np: numpy array (H, W, 3) in [0, 255] uint8 or [0, 1] float
        heatmap: numpy array (H, W) in [0, 1]
        alpha: blend factor (0 = only image, 1 = only heatmap)
        colormap: matplotlib colormap name

    Returns:
        blended: numpy array (H, W, 3) uint8
    """
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) float in [0, 1]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    blended = (
        (1 - alpha) * image_np.astype(np.float32) +
        alpha * heatmap_colored.astype(np.float32)
    ).astype(np.uint8)

    return blended


def _resize_heatmap(heatmap, h, w):
    """Resize a heatmap to (h, w) if necessary."""
    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        return np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
        ) / 255.0
    return heatmap


def _frame_to_np(frame):
    """Convert a frame tensor or array to uint8 numpy (H, W, 3)."""
    if isinstance(frame, torch.Tensor):
        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
    else:
        frame_np = np.array(frame)
    return frame_np


def create_visualization_grid(frames, heatmaps, actions=None,
                               cross_attn_heatmaps=None,
                               saliency_maps=None, gradcam_maps=None,
                               episode_idx=0, output_path="attention_grid.png"):
    """
    Create a grid visualization showing original frames, heatmaps, and overlays.

    Layout per frame (dynamic rows):
      Row 1: Original image
      Row 2: Vision encoder self-attention heatmap (colorized)
      Row 3: Overlay (image + self-attention heatmap blended)
      Row 4-5 (optional): Cross-attention heatmap + co-attention overlay
      +1 row (optional): Saliency overlay (|dA/dpx|)
      +1 row (optional): GradCAM overlay (SigLIP last layer)
    """
    n_frames = len(frames)
    has_cross = cross_attn_heatmaps is not None and len(cross_attn_heatmaps) == n_frames
    has_saliency = saliency_maps is not None and len(saliency_maps) == n_frames
    has_gradcam = gradcam_maps is not None and len(gradcam_maps) == n_frames

    # Dynamic row count: base 3 + 2 cross + 1 saliency + 1 gradcam = 7 max
    n_rows = 3
    if has_cross:
        n_rows += 2
    if has_saliency:
        n_rows += 1
    if has_gradcam:
        n_rows += 1

    fig = plt.figure(figsize=(4 * n_frames, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_frames, hspace=0.3, wspace=0.05)

    for i, (frame, heatmap) in enumerate(zip(frames, heatmaps)):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]

        heatmap_resized = _resize_heatmap(heatmap, h, w)
        overlay = overlay_heatmap(frame_np, heatmap_resized, alpha=0.45)

        row = 0

        # Precompute optional maps needed by multiple rows
        cross_hm = None
        co_overlay = None
        if has_cross:
            cross_hm = _resize_heatmap(cross_attn_heatmaps[i], h, w)
            co_attn = heatmap_resized * cross_hm
            co_attn = co_attn / (co_attn.max() + 1e-8)
            co_overlay = frame_np.copy()
            co_overlay = (0.5 * co_overlay.astype(np.float32)
                          + 0.5 * _CYAN_CMAP(co_attn)[:, :, :3] * 255)
            co_overlay = np.clip(co_overlay, 0, 255).astype(np.uint8)

        sal_overlay = None
        if has_saliency:
            sal_hm = _resize_heatmap(saliency_maps[i], h, w)
            sal_overlay = overlay_heatmap(frame_np, sal_hm, alpha=0.45, colormap="inferno")

        gc_overlay = None
        if has_gradcam:
            gc_hm = _resize_heatmap(gradcam_maps[i], h, w)
            gc_overlay = overlay_heatmap(frame_np, gc_hm, alpha=0.45, colormap="magma")

        # --- Top rows: raw maps ---

        # Original
        ax = fig.add_subplot(gs[row, i])
        ax.imshow(frame_np)
        ax.set_title(f"Frame {i}", fontsize=10)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Original", fontsize=11, rotation=0, labelpad=60, va="center")
        row += 1

        # SigLIP self-attention heatmap
        ax = fig.add_subplot(gs[row, i])
        ax.imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("SigLIP\nself-attn", fontsize=11, rotation=0, labelpad=60, va="center")
        row += 1

        if has_cross:
            # Action cross-attention heatmap
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(cross_hm, cmap="Greens", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Action\ncross-attn", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

        if has_saliency:
            # Saliency overlay
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(sal_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Saliency\n|dA/dpx|", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

        # --- Bottom rows: interpretable overlays (most important) ---

        # Self-attention overlay
        ax = fig.add_subplot(gs[row, i])
        ax.imshow(overlay)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Self-attn\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")
        row += 1

        if has_cross:
            # Co-attention overlay (self-attn × cross-attn)
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(co_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Co-attention\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

        if has_gradcam:
            # GradCAM overlay (bottommost)
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(gc_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("GradCAM\nSigLIP L-1", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

    # --- Build legend string dynamically ---
    legend_parts = [
        "Original",
        "SigLIP self-attn heatmap",
    ]
    if has_cross:
        legend_parts.append("Action cross-attn heatmap")
    if has_saliency:
        legend_parts.append("Saliency |dA/dpx|")
    legend_parts.append("Self-attn overlay")
    if has_cross:
        legend_parts.append("Co-attention (self \u00d7 cross)")
    if has_gradcam:
        legend_parts.append("GradCAM SigLIP L-1")

    legend = "  |  ".join(f"Row {j+1}: {lbl}" for j, lbl in enumerate(legend_parts))

    extra_lines = []
    if has_cross:
        extra_lines.append("Co-attention: self-attn \u00d7 cross-attn \u2014 bright regions are both visually salient and action-relevant")
    if has_saliency or has_gradcam:
        extra_lines.append("Gradient rows show which image regions causally influence the predicted action")

    title = f"SmolVLA Attention \u2014 Episode {episode_idx}\n\n{legend}"
    if extra_lines:
        title += "\n\n" + "\n".join(extra_lines)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved grid: {output_path}")


def save_individual_frames(frames, heatmaps, output_dir, episode_idx=0):
    """Save each frame's overlay as a separate high-res PNG."""
    os.makedirs(output_dir, exist_ok=True)

    for i, (frame, heatmap) in enumerate(zip(frames, heatmaps)):
        if isinstance(frame, torch.Tensor):
            frame_np = frame.permute(1, 2, 0).numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        else:
            frame_np = np.array(frame)

        h, w = frame_np.shape[:2]
        if heatmap.shape[0] != h or heatmap.shape[1] != w:
            heatmap_resized = np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
            ) / 255.0
        else:
            heatmap_resized = heatmap

        overlay = overlay_heatmap(frame_np, heatmap_resized, alpha=0.45)

        # Save overlay
        path = os.path.join(output_dir, f"ep{episode_idx:03d}_frame{i:04d}_overlay.png")
        Image.fromarray(overlay).save(path)

        # Save raw heatmap
        path_hm = os.path.join(output_dir, f"ep{episode_idx:03d}_frame{i:04d}_heatmap.png")
        fig_hm, ax_hm = plt.subplots(figsize=(6, 6))
        ax_hm.imshow(heatmap_resized, cmap="jet")
        ax_hm.axis("off")
        fig_hm.savefig(path_hm, dpi=100, bbox_inches="tight")
        plt.close(fig_hm)

    print(f"  Saved {len(frames)} individual frames to {output_dir}/")


def create_per_head_grid(frame, attn_weights, grid_size, image_size,
                         output_path="per_head_attention.png", content_crop=None,
                         baseline_per_head=None, threshold_pct=0.0):
    """
    Visualise each attention head's pattern individually for a single
    frame.  Useful for identifying specialised heads (e.g. one tracking
    the gripper, another tracking the object).

    Args:
        frame: image tensor (C, H, W) or numpy (H, W, 3)
        attn_weights: Tensor of shape ``(heads, patches, patches)`` —
            one attention matrix per head (not yet averaged).
        grid_size: ``(H_patches, W_patches)``
        image_size: ``(H_pixels, W_pixels)``
        output_path: where to save the PNG
        baseline_per_head: Optional tensor ``(heads, patches)`` — per-head
            positional baseline to subtract before heatmap creation.
        threshold_pct: Percentile threshold passed to ``attention_to_heatmap``.
    """
    if isinstance(frame, torch.Tensor):
        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
    else:
        frame_np = np.array(frame)

    n_heads = attn_weights.shape[0]
    cols = min(n_heads, 8)
    rows = math.ceil(n_heads / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for h in range(n_heads):
        r, c = divmod(h, cols)
        head_attn = attn_weights[h]                      # (patches, patches)
        scores = head_attn.mean(dim=0)                    # per-patch importance
        if baseline_per_head is not None and h < baseline_per_head.shape[0]:
            scores = torch.clamp(scores - baseline_per_head[h], min=0)
        hmap = attention_to_heatmap(scores, grid_size, image_size,
                                    content_crop=content_crop,
                                    threshold_pct=threshold_pct)

        blended = overlay_heatmap(frame_np.copy(), hmap, alpha=0.5)
        axes[r, c].imshow(blended)
        axes[r, c].set_title(f"Head {h}", fontsize=9)
        axes[r, c].axis("off")

    # Turn off unused subplots
    for idx in range(n_heads, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("SigLIP vision encoder per-head self-attention", fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved per-head grid: {output_path}")

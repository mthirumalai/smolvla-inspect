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
                               episode_idx=0, output_path="attention_grid.png"):
    """
    Create a grid visualization showing original frames, heatmaps, and overlays.

    Layout per frame:
      Row 1: Original image
      Row 2: Vision encoder self-attention heatmap (colorized)
      Row 3: Overlay (image + self-attention heatmap blended)

    When *cross_attn_heatmaps* is provided two extra rows are added:
      Row 4: Action→Vision cross-attention heatmap
      Row 5: Dual-color overlay (self-attn blue, cross-attn red)
    """
    n_frames = len(frames)
    has_cross = cross_attn_heatmaps is not None and len(cross_attn_heatmaps) == n_frames
    n_rows = 5 if has_cross else 3

    fig = plt.figure(figsize=(4 * n_frames, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_frames, hspace=0.3, wspace=0.05)

    for i, (frame, heatmap) in enumerate(zip(frames, heatmaps)):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]

        heatmap_resized = _resize_heatmap(heatmap, h, w)
        overlay = overlay_heatmap(frame_np, heatmap_resized, alpha=0.45)

        # Row 1: Original
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(frame_np)
        ax1.set_title(f"Frame {i}", fontsize=10)
        ax1.axis("off")
        if i == 0:
            ax1.set_ylabel("Original", fontsize=11, rotation=0, labelpad=60, va="center")

        # Row 2: Self-attention heatmap
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
        ax2.axis("off")
        if i == 0:
            ax2.set_ylabel("SigLIP\nself-attn", fontsize=11, rotation=0, labelpad=60, va="center")

        # Row 3: Self-attention overlay
        ax3 = fig.add_subplot(gs[2, i])
        ax3.imshow(overlay)
        ax3.axis("off")
        if i == 0:
            ax3.set_ylabel("Self-attn\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")

        if has_cross:
            cross_hm = _resize_heatmap(cross_attn_heatmaps[i], h, w)

            # Row 4: Cross-attention heatmap
            ax4 = fig.add_subplot(gs[3, i])
            ax4.imshow(cross_hm, cmap="Greens", vmin=0, vmax=1)
            ax4.axis("off")
            if i == 0:
                ax4.set_ylabel("Action\ncross-attn", fontsize=11, rotation=0, labelpad=60, va="center")

            # Row 5: Co-attention overlay (self-attn × cross-attn)
            co_attn = heatmap_resized * cross_hm          # element-wise product
            co_attn = co_attn / (co_attn.max() + 1e-8)   # renormalize to [0, 1]
            co_overlay = frame_np.copy()
            co_overlay = (0.5 * co_overlay.astype(np.float32)
                          + 0.5 * _CYAN_CMAP(co_attn)[:, :, :3] * 255)
            co_overlay = np.clip(co_overlay, 0, 255).astype(np.uint8)

            ax5 = fig.add_subplot(gs[4, i])
            ax5.imshow(co_overlay)
            ax5.axis("off")
            if i == 0:
                ax5.set_ylabel("Co-attention\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")

    if has_cross:
        legend = ("Row 1: Original  |  Row 2: SigLIP self-attn heatmap  |  Row 3: Self-attn overlay  |  "
                  "Row 4: Action cross-attn heatmap  |  Row 5: Co-attention (self × cross)")
        dual_legend = "Co-attention: self-attn × cross-attn — bright regions are both visually salient and action-relevant"
    else:
        legend = "Row 1: Original  |  Row 2: SigLIP self-attn heatmap  |  Row 3: Overlay (heatmap on frame)"
        dual_legend = None

    if dual_legend:
        title = (
            f"SmolVLA Attention — Episode {episode_idx}\n\n"
            f"{legend}\n\n"
            f"{dual_legend}"
        )
    else:
        title = (
            f"SmolVLA Attention — Episode {episode_idx}\n\n"
            f"{legend}"
        )
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

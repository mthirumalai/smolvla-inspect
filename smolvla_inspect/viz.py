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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

from .heatmap import attention_to_heatmap

_CYAN_CMAP = LinearSegmentedColormap.from_list("cyan", ["black", "cyan", "white"])


def _adaptive_figsize(cell_w, cell_h, n_cols, n_rows, max_w=48.0, max_h=60.0, base_dpi=150):
    """Compute figure size and DPI that fit within max dimensions."""
    w, h = cell_w * n_cols, cell_h * n_rows
    scale = min(1.0, max_w / w, max_h / h)
    return (w * scale, h * scale), max(base_dpi, int(base_dpi / scale))


def _spatial_entropy(heatmap):
    """Normalized spatial entropy of a heatmap. 0 = focused, 1 = uniform."""
    flat = heatmap.flatten()
    flat = flat / (flat.sum() + 1e-12)
    flat = flat[flat > 0]
    H = -(flat * np.log(flat)).sum()
    return float(H / (np.log(len(heatmap.flatten())) + 1e-12))


def _shared_normalize(heatmaps):
    """Re-normalize list of heatmaps to shared [0,1] range. Returns (normed, vmin, vmax)."""
    valid = [h for h in heatmaps if h is not None]
    if not valid:
        return heatmaps, 0.0, 1.0
    vmin = min(h.min() for h in valid)
    vmax = max(h.max() for h in valid)
    rng = vmax - vmin
    if rng < 1e-12:
        return heatmaps, float(vmin), float(vmax)
    normed = []
    for h in heatmaps:
        if h is None:
            normed.append(None)
        else:
            normed.append((h - vmin) / rng)
    return normed, float(vmin), float(vmax)


def _compute_centroid(heatmap):
    """Weighted center of mass. Returns (cy, cx) in pixel coordinates."""
    h, w = heatmap.shape
    total = heatmap.sum() + 1e-12
    ys = np.arange(h)
    xs = np.arange(w)
    cy = (heatmap.sum(axis=1) * ys).sum() / total
    cx = (heatmap.sum(axis=0) * xs).sum() / total
    return float(cy), float(cx)


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
                               connector_gradcam_maps=None,
                               language_diff_maps=None,
                               episode_idx=0, output_path="attention_grid.png",
                               smooth_n=1):
    """
    Create a grid visualization showing original frames, heatmaps, and overlays.

    Layout per frame (dynamic rows):
      Raw maps section:
        Row: Original image
        Row: Vision encoder self-attention heatmap (colorized)
        Row (optional): Cross-attention heatmap
        Row (optional): Saliency overlay
      --- separator ---
      Interpretable overlays section:
        Row: Self-attention overlay
        Row (optional): Co-attention overlay
        Row (optional): GradCAM overlay
        Row (optional): Connector GradCAM overlay
        Row (optional): Language-conditional diff overlay

    Features shared normalization across frames, colorbars for heatmap rows,
    and a separator between raw maps and interpretable overlays.
    """
    n_frames = len(frames)
    has_cross = cross_attn_heatmaps is not None and len(cross_attn_heatmaps) == n_frames
    has_saliency = saliency_maps is not None and len(saliency_maps) == n_frames
    has_gradcam = gradcam_maps is not None and len(gradcam_maps) == n_frames
    has_connector = connector_gradcam_maps is not None and len(connector_gradcam_maps) == n_frames
    has_lang_diff = language_diff_maps is not None and len(language_diff_maps) == n_frames

    # --- Pass 1: resize all heatmaps and collect per-row-type lists ---
    frame_nps = []
    all_self_attn_hm = []
    all_cross_hm = []
    all_sal_hm = []
    all_gc_hm = []
    all_conn_hm = []
    all_diff_hm = []

    for i, (frame, heatmap) in enumerate(zip(frames, heatmaps)):
        frame_np = _frame_to_np(frame)
        frame_nps.append(frame_np)
        h, w = frame_np.shape[:2]
        all_self_attn_hm.append(_resize_heatmap(heatmap, h, w))
        if has_cross:
            all_cross_hm.append(_resize_heatmap(cross_attn_heatmaps[i], h, w))
        if has_saliency:
            all_sal_hm.append(_resize_heatmap(saliency_maps[i], h, w))
        if has_gradcam:
            all_gc_hm.append(_resize_heatmap(gradcam_maps[i], h, w))
        if has_connector:
            all_conn_hm.append(_resize_heatmap(connector_gradcam_maps[i], h, w))
        if has_lang_diff:
            if language_diff_maps[i] is not None:
                all_diff_hm.append(_resize_heatmap(language_diff_maps[i]["diff_cam"], h, w))
            else:
                all_diff_hm.append(None)

    # Shared normalization per row type
    all_self_attn_hm, sa_vmin, sa_vmax = _shared_normalize(all_self_attn_hm)
    if has_cross:
        all_cross_hm, cr_vmin, cr_vmax = _shared_normalize(all_cross_hm)
    if has_saliency:
        all_sal_hm, sal_vmin, sal_vmax = _shared_normalize(all_sal_hm)
    if has_gradcam:
        all_gc_hm, gc_vmin, gc_vmax = _shared_normalize(all_gc_hm)
    if has_connector:
        all_conn_hm, conn_vmin, conn_vmax = _shared_normalize(all_conn_hm)
    # Lang diff is symmetric [-1,1], no shared normalize needed

    # --- Build row layout with separator ---
    # Raw rows: original, self-attn, [cross-attn], [saliency]
    raw_row_count = 2
    if has_cross:
        raw_row_count += 1
    if has_saliency:
        raw_row_count += 1

    # Overlay rows: self-attn overlay, [co-attn], [gradcam], [connector], [lang-diff]
    overlay_row_count = 1
    if has_cross:
        overlay_row_count += 1
    if has_gradcam:
        overlay_row_count += 1
    if has_connector:
        overlay_row_count += 1
    if has_lang_diff:
        overlay_row_count += 1

    # Total rows = raw + 1 separator + overlay
    n_content_rows = raw_row_count + overlay_row_count
    n_rows = n_content_rows + 1  # +1 for separator
    height_ratios = [1.0] * raw_row_count + [0.05] + [1.0] * overlay_row_count
    n_cols = n_frames + 1  # +1 for colorbar column
    width_ratios = [1.0] * n_frames + [0.05]

    figsize, dpi = _adaptive_figsize(4, 4, n_cols, n_content_rows)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.05,
                           height_ratios=height_ratios, width_ratios=width_ratios)

    # Track which row types have colorbars: (row_index, cmap_name, vmin, vmax, label)
    colorbar_info = {}

    # --- Pass 2: Plot ---
    for i in range(n_frames):
        frame_np = frame_nps[i]
        h, w = frame_np.shape[:2]
        heatmap_resized = all_self_attn_hm[i]

        row = 0

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
        colorbar_info.setdefault(row, ("jet", 0, 1, "Self-attn"))
        row += 1

        if has_cross:
            cross_hm = all_cross_hm[i]
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(cross_hm, cmap="Greens", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Action\ncross-attn", fontsize=11, rotation=0, labelpad=60, va="center")
            colorbar_info.setdefault(row, ("Greens", 0, 1, "Cross-attn"))
            row += 1

        if has_saliency:
            sal_hm = all_sal_hm[i]
            sal_label = f"SmoothGrad\nN={smooth_n}" if smooth_n > 1 else "Saliency\n|dA/dpx|"
            sal_overlay = overlay_heatmap(frame_np, sal_hm, alpha=0.45, colormap="inferno")
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(sal_hm, cmap="inferno", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(sal_label, fontsize=11, rotation=0, labelpad=60, va="center")
            colorbar_info.setdefault(row, ("inferno", 0, 1, "Saliency"))
            row += 1

        # --- Separator row ---
        sep_row = row
        for ci in range(n_frames):
            ax_sep = fig.add_subplot(gs[sep_row, ci])
            ax_sep.axhline(y=0.5, color="#DBE4E8", linewidth=2)
            ax_sep.set_xlim(0, 1)
            ax_sep.set_ylim(0, 1)
            ax_sep.axis("off")
        row += 1

        # --- Overlay rows ---

        # Self-attention overlay
        overlay = overlay_heatmap(frame_np, heatmap_resized, alpha=0.45)
        ax = fig.add_subplot(gs[row, i])
        ax.imshow(overlay)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Self-attn\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")
        row += 1

        if has_cross:
            cross_hm = all_cross_hm[i]
            co_attn = np.sqrt(heatmap_resized * cross_hm)
            co_attn = co_attn / (co_attn.max() + 1e-8)
            co_overlay = frame_np.copy()
            co_overlay = (0.5 * co_overlay.astype(np.float32)
                          + 0.5 * _CYAN_CMAP(co_attn)[:, :, :3] * 255)
            co_overlay = np.clip(co_overlay, 0, 255).astype(np.uint8)
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(co_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Co-attention\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

        if has_gradcam:
            gc_hm = all_gc_hm[i]
            gc_overlay = overlay_heatmap(frame_np, gc_hm, alpha=0.45, colormap="magma")
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(gc_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("GradCAM\nSigLIP L-1", fontsize=11, rotation=0, labelpad=60, va="center")
            colorbar_info.setdefault(row, ("magma", 0, 1, "GradCAM"))
            row += 1

        if has_connector:
            conn_hm = all_conn_hm[i]
            conn_overlay = overlay_heatmap(frame_np, conn_hm, alpha=0.45, colormap="magma")
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(conn_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("GradCAM\nConnector", fontsize=11, rotation=0, labelpad=60, va="center")
            colorbar_info.setdefault(row, ("magma", 0, 1, "Connector"))
            row += 1

        if has_lang_diff:
            ax = fig.add_subplot(gs[row, i])
            diff_hm = all_diff_hm[i] if i < len(all_diff_hm) else None
            if diff_hm is not None:
                cmap_rdbu = plt.get_cmap("RdBu_r")
                diff_colored = cmap_rdbu((diff_hm + 1) / 2)[:, :, :3]
                diff_colored = (diff_colored * 255).astype(np.uint8)
                lang_diff_overlay = (
                    0.55 * frame_np.astype(np.float32) +
                    0.45 * diff_colored.astype(np.float32)
                ).astype(np.uint8)
                ax.imshow(lang_diff_overlay)
            else:
                ax.imshow(frame_np)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Lang-cond\ndiff", fontsize=11, rotation=0, labelpad=60, va="center")
            colorbar_info.setdefault(row, ("RdBu_r", -1, 1, "Lang diff"))
            row += 1

    # --- Add colorbars in the extra column ---
    for row_idx, (cmap_name, vmin, vmax, label) in colorbar_info.items():
        cbar_ax = fig.add_subplot(gs[row_idx, n_frames])
        if cmap_name == "RdBu_r":
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar_ax.tick_params(labelsize=7)

    # --- Build legend string dynamically ---
    legend_parts = [
        "Original",
        "SigLIP self-attn heatmap",
    ]
    if has_cross:
        legend_parts.append("Action cross-attn heatmap")
    if has_saliency:
        legend_parts.append(f"SmoothGrad N={smooth_n}" if smooth_n > 1 else "Saliency |dA/dpx|")
    legend_parts.append("--- separator ---")
    legend_parts.append("Self-attn overlay")
    if has_cross:
        legend_parts.append("Co-attention \u221a(self \u00d7 cross)")
    if has_gradcam:
        legend_parts.append("GradCAM SigLIP L-1")
    if has_connector:
        legend_parts.append("GradCAM Connector")
    if has_lang_diff:
        legend_parts.append("Lang-conditional diff")

    legend = "  |  ".join(f"Row {j+1}: {lbl}" for j, lbl in enumerate(legend_parts))

    extra_lines = []
    if has_cross:
        extra_lines.append("Co-attention: \u221a(self-attn \u00d7 cross-attn) \u2014 geometric mean, bright = both visually salient and action-relevant")
    if has_saliency or has_gradcam:
        extra_lines.append("Gradient rows show which image regions causally influence the predicted action")
    if has_connector:
        extra_lines.append("Connector GradCAM: attribution on post-pixel-shuffle tokens (8\u00d78)")
    if has_lang_diff:
        extra_lines.append("Lang-cond diff: red = original stronger, blue = alt stronger")

    title = f"SmolVLA Attention \u2014 Episode {episode_idx}\n\n{legend}"
    if extra_lines:
        title += "\n\n" + "\n".join(extra_lines)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
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

    Each cell shows the head's overlay with spatial entropy in the title.
    The most focused head (lowest entropy) gets a green border, the most
    diffuse head (highest entropy) gets a red border.

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
    frame_np = _frame_to_np(frame)

    n_heads = attn_weights.shape[0]
    cols = min(n_heads, 8)
    rows = math.ceil(n_heads / cols)

    figsize, dpi = _adaptive_figsize(3, 3, cols, rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    entropies = {}
    for h in range(n_heads):
        r, c = divmod(h, cols)
        head_attn = attn_weights[h]                      # (patches, patches)
        scores = head_attn.mean(dim=0)                    # per-patch importance
        if baseline_per_head is not None and h < baseline_per_head.shape[0]:
            scores = torch.clamp(scores - baseline_per_head[h], min=0)
        hmap = attention_to_heatmap(scores, grid_size, image_size,
                                    content_crop=content_crop,
                                    threshold_pct=threshold_pct)

        ent = _spatial_entropy(hmap)
        entropies[h] = ent

        blended = overlay_heatmap(frame_np.copy(), hmap, alpha=0.5)
        axes[r, c].imshow(blended)
        axes[r, c].set_title(f"Head {h} (H={ent:.2f})", fontsize=9)
        axes[r, c].axis("off")

    # Highlight most focused (green) and most diffuse (red) heads
    if entropies:
        min_h = min(entropies, key=entropies.get)
        max_h = max(entropies, key=entropies.get)
        for head_idx, color in [(min_h, "lime"), (max_h, "red")]:
            r, c = divmod(head_idx, cols)
            for spine in axes[r, c].spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)

    # Turn off unused subplots
    for idx in range(n_heads, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("SigLIP vision encoder per-head self-attention", fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved per-head grid: {output_path}")


# ---------------------------------------------------------------------------
# F3: Per-denoising-step cross-attention grid
# ---------------------------------------------------------------------------

def create_per_step_cross_attn_grid(frames, per_step_maps, episode_idx=0,
                                     output_path="per_step_cross_attn.png"):
    """
    Visualize cross-attention across denoising steps.

    Includes a thumbnail reference row at the top, per-cell spatial entropy
    annotations, and an additional centroid trajectory figure.

    Args:
        frames: list of image tensors (C, H, W)
        per_step_maps: list (per frame) of lists (per step) of Tensors
            ``(n_vision_tokens,)``.
        output_path: where to save the PNG.
    """
    n_frames = len(frames)
    # Determine number of steps from first frame
    n_steps = len(per_step_maps[0]) if per_step_maps and per_step_maps[0] else 0
    if n_steps == 0:
        print("  No per-step cross-attention data to visualize")
        return

    # Layout: thumbnail row + n_steps content rows
    n_rows = 1 + n_steps
    height_ratios = [0.5] + [1.0] * n_steps
    figsize, dpi = _adaptive_figsize(3, 3, n_frames, n_steps + 1)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_frames, hspace=0.3, wspace=0.05,
                           height_ratios=height_ratios)

    # Thumbnail reference row
    frame_nps = []
    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        frame_nps.append(frame_np)
        ax = fig.add_subplot(gs[0, fi])
        ax.imshow(frame_np)
        ax.set_title(f"Frame {fi}", fontsize=9)
        ax.axis("off")
        if fi == 0:
            ax.set_ylabel("Original", fontsize=9, rotation=0, labelpad=50, va="center")

    # Collect per-frame, per-step heatmaps for trajectory figure
    all_step_heatmaps = []  # [frame][step] = hm (h, w)

    for fi in range(n_frames):
        frame_np = frame_nps[fi]
        h, w = frame_np.shape[:2]

        step_maps = per_step_maps[fi] if fi < len(per_step_maps) else []
        frame_hms = []
        for si in range(n_steps):
            ax = fig.add_subplot(gs[si + 1, fi])
            if si < len(step_maps):
                scores = step_maps[si]
                n_vis = scores.shape[0]
                side = int(n_vis ** 0.5)
                hm_small = scores.reshape(side, side).detach().float().cpu().numpy()
                hm_small = hm_small / (hm_small.max() + 1e-8)
                hm = _resize_heatmap(hm_small, h, w)
                blended = overlay_heatmap(frame_np, hm, alpha=0.45, colormap="Greens")
                ax.imshow(blended)

                # Entropy annotation
                ent = _spatial_entropy(hm)
                ax.text(w - 5, h - 10, f"H={ent:.2f}", fontsize=7,
                        color="white", backgroundcolor="black", alpha=0.8,
                        ha="right")
                frame_hms.append(hm)
            else:
                ax.imshow(frame_np)
                frame_hms.append(None)

            ax.axis("off")
            if fi == 0:
                t_val = 1.0 - si * 0.1 if n_steps == 10 else si
                ax.set_ylabel(f"Step {si}\n(t={t_val:.1f})", fontsize=9,
                              rotation=0, labelpad=50, va="center")

        all_step_heatmaps.append(frame_hms)

    fig.suptitle(f"Per-denoising-step cross-attention \u2014 Episode {episode_idx}",
                 fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved per-step cross-attn grid: {output_path}")

    # --- Centroid trajectory figure ---
    _create_centroid_trajectory(frame_nps, all_step_heatmaps, n_steps, episode_idx, output_path)


def _create_centroid_trajectory(frame_nps, all_step_heatmaps, n_steps, episode_idx, base_path):
    """Create centroid trajectory overlaid on original frames."""
    n_frames = len(frame_nps)
    traj_path = base_path.replace(".png", "_trajectory.png")

    figsize, dpi = _adaptive_figsize(4, 4, n_frames, 1)
    fig, axes = plt.subplots(1, n_frames, figsize=figsize)
    if n_frames == 1:
        axes = [axes]

    cmap_traj = plt.get_cmap("coolwarm")

    for fi in range(n_frames):
        frame_np = frame_nps[fi]
        axes[fi].imshow(frame_np)
        axes[fi].set_title(f"Frame {fi}", fontsize=9)
        axes[fi].axis("off")

        centroids = []
        for si in range(n_steps):
            hm = all_step_heatmaps[fi][si]
            if hm is not None:
                cy, cx = _compute_centroid(hm)
                centroids.append((cx, cy))

        if len(centroids) >= 2:
            for j in range(len(centroids) - 1):
                t = j / max(len(centroids) - 1, 1)
                color = cmap_traj(t)
                axes[fi].plot([centroids[j][0], centroids[j + 1][0]],
                              [centroids[j][1], centroids[j + 1][1]],
                              color=color, linewidth=2, alpha=0.8)
            # Start and end markers
            axes[fi].plot(centroids[0][0], centroids[0][1], "o",
                          color="blue", markersize=8, zorder=5)
            axes[fi].plot(centroids[-1][0], centroids[-1][1], "o",
                          color="red", markersize=8, zorder=5)

    fig.suptitle(f"Attention centroid trajectory (blue=t=1.0 \u2192 red=t=0.1) \u2014 Episode {episode_idx}",
                 fontsize=12, fontweight="bold")
    plt.savefig(traj_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved centroid trajectory: {traj_path}")


# ---------------------------------------------------------------------------
# F2: VLM layer GradCAM grid
# ---------------------------------------------------------------------------

def create_vlm_layer_grid(frames, vlm_layer_results, layer_indices,
                           episode_idx=0, output_path="vlm_layers.png",
                           lang_tokens=None):
    """
    Visualize GradCAM at multiple VLM intermediate layers.

    Includes a thumbnail reference row at the top. If ``lang_tokens`` is
    provided and ``lang_cam`` data exists, also saves a separate language
    token attribution bar chart figure.

    Args:
        frames: list of image tensors
        vlm_layer_results: dict mapping layer_index to list of per-frame dicts
            with ``"vision_cam"``, ``"lang_cam"``, ``"state_cam"``.
        layer_indices: sorted list of layer indices
        lang_tokens: optional list of token strings for bar chart labels
        output_path: where to save the PNG.
    """
    n_frames = len(frames)
    n_layers = len(layer_indices)
    if n_layers == 0:
        return

    # Layout: thumbnail row + n_layers content rows
    n_rows = 1 + n_layers
    height_ratios = [0.5] + [1.0] * n_layers
    figsize, dpi = _adaptive_figsize(3, 3, n_frames, n_layers + 1)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_frames, hspace=0.3, wspace=0.05,
                           height_ratios=height_ratios)

    # Thumbnail reference row
    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        ax = fig.add_subplot(gs[0, fi])
        ax.imshow(frame_np)
        ax.set_title(f"Frame {fi}", fontsize=9)
        ax.axis("off")
        if fi == 0:
            ax.set_ylabel("Original", fontsize=9, rotation=0, labelpad=50, va="center")

    for li_idx, li in enumerate(layer_indices):
        layer_data = vlm_layer_results.get(li, [])
        for fi, frame in enumerate(frames):
            ax = fig.add_subplot(gs[li_idx + 1, fi])
            frame_np = _frame_to_np(frame)
            h, w = frame_np.shape[:2]

            if fi < len(layer_data) and layer_data[fi] is not None:
                vcam = layer_data[fi]["vision_cam"]
                vcam_resized = _resize_heatmap(vcam, h, w)
                blended = overlay_heatmap(frame_np, vcam_resized, alpha=0.45, colormap="magma")
                ax.imshow(blended)

                # Annotate state attribution if available
                state_cam = layer_data[fi].get("state_cam")
                if state_cam is not None:
                    ax.text(5, h - 10, f"state={state_cam:.2f}", fontsize=7,
                            color="white", backgroundcolor="black", alpha=0.7)
            else:
                ax.imshow(frame_np)

            ax.axis("off")
            if fi == 0:
                ax.set_ylabel(f"Layer {li + 1}", fontsize=10, rotation=0,
                              labelpad=50, va="center")

    fig.suptitle(f"VLM intermediate layer GradCAM \u2014 Episode {episode_idx}",
                 fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved VLM layer grid: {output_path}")

    # --- Language token attribution bar charts ---
    if lang_tokens is not None:
        _create_lang_token_bar_charts(vlm_layer_results, layer_indices,
                                       n_frames, lang_tokens, episode_idx,
                                       output_path)


def _create_lang_token_bar_charts(vlm_layer_results, layer_indices, n_frames,
                                   lang_tokens, episode_idx, base_path):
    """Create horizontal bar charts of language token attribution per layer."""
    # Check if any layer has lang_cam data
    has_lang_data = False
    for li in layer_indices:
        layer_data = vlm_layer_results.get(li, [])
        for fd in layer_data:
            if fd is not None and fd.get("lang_cam") is not None:
                has_lang_data = True
                break
        if has_lang_data:
            break

    if not has_lang_data:
        return

    n_layers = len(layer_indices)
    lang_path = base_path.replace(".png", "_lang_tokens.png")

    figsize, dpi = _adaptive_figsize(6, 3, 1, n_layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, squeeze=False)

    for li_idx, li in enumerate(layer_indices):
        ax = axes[li_idx, 0]
        layer_data = vlm_layer_results.get(li, [])

        # Average lang_cam across frames
        valid_cams = []
        for fi in range(n_frames):
            if fi < len(layer_data) and layer_data[fi] is not None:
                lc = layer_data[fi].get("lang_cam")
                if lc is not None:
                    valid_cams.append(lc)

        if not valid_cams:
            ax.set_title(f"Layer {li + 1} (no data)", fontsize=9)
            ax.axis("off")
            continue

        avg_cam = np.mean(valid_cams, axis=0)
        n_tok = min(len(avg_cam), len(lang_tokens))
        tokens = lang_tokens[:n_tok]
        values = avg_cam[:n_tok]

        y_pos = np.arange(n_tok)
        ax.barh(y_pos, values, color="#3B3BD3", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"Layer {li + 1}", fontsize=10)
        ax.set_xlabel("Attribution", fontsize=8)

    fig.suptitle(f"Language token attribution (frame-averaged) \u2014 Episode {episode_idx}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(lang_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved lang token bar charts: {lang_path}")


# ---------------------------------------------------------------------------
# F6: Per-action-dimension GradCAM grid
# ---------------------------------------------------------------------------

def create_per_action_dim_grid(frames, per_dim_maps, action_dim_names=None,
                                episode_idx=0, output_path="per_action_dim.png",
                                magnitudes=None):
    """
    Visualize GradCAM for each action dimension.

    Args:
        frames: list of image tensors
        per_dim_maps: list (per frame) of lists (per action dim) of numpy
            arrays ``(grid_h, grid_w)``, or *None* for failed frames.
        action_dim_names: optional list of dimension name strings.
        magnitudes: optional list (per frame) of lists (per dim) of float
            raw GradCAM magnitudes.
        output_path: where to save the PNG.
    """
    n_frames = len(frames)
    # Determine number of dims from first non-None entry
    n_dims = 0
    for m in per_dim_maps:
        if m is not None:
            n_dims = len(m)
            break
    if n_dims == 0:
        print("  No per-action-dim data to visualize")
        return

    if action_dim_names is None:
        action_dim_names = [f"dim_{d}" for d in range(n_dims)]

    # Layout: thumbnail reference row + n_dims content rows
    n_rows = 1 + n_dims
    height_ratios = [0.5] + [1.0] * n_dims
    figsize, dpi = _adaptive_figsize(3, 3, n_frames, n_dims + 1)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_frames, hspace=0.3, wspace=0.05,
                           height_ratios=height_ratios)

    # Thumbnail reference row
    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        ax = fig.add_subplot(gs[0, fi])
        ax.imshow(frame_np)
        ax.set_title(f"Frame {fi}", fontsize=9)
        ax.axis("off")
        if fi == 0:
            ax.set_ylabel("Original", fontsize=9, rotation=0, labelpad=60, va="center")

    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]

        dim_maps = per_dim_maps[fi] if fi < len(per_dim_maps) and per_dim_maps[fi] is not None else None
        frame_mags = None
        if magnitudes is not None and fi < len(magnitudes):
            frame_mags = magnitudes[fi]

        for di in range(n_dims):
            ax = fig.add_subplot(gs[di + 1, fi])
            if dim_maps is not None and di < len(dim_maps):
                hm = _resize_heatmap(dim_maps[di], h, w)
                blended = overlay_heatmap(frame_np, hm, alpha=0.45, colormap="magma")
                ax.imshow(blended)
                # Magnitude annotation
                if frame_mags is not None and di < len(frame_mags):
                    mag_val = frame_mags[di]
                    ax.text(5, h - 10, f"mag={mag_val:.2e}", fontsize=7,
                            color="white", backgroundcolor="black", alpha=0.8)
            else:
                ax.imshow(frame_np)
            ax.axis("off")
            if fi == 0:
                name = action_dim_names[di] if di < len(action_dim_names) else f"dim_{di}"
                ax.set_ylabel(name, fontsize=9, rotation=0, labelpad=60, va="center")

    fig.suptitle(f"Per-action-dimension GradCAM \u2014 Episode {episode_idx}",
                 fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved per-action-dim grid: {output_path}")


# ---------------------------------------------------------------------------
# F5: Language-conditional comparison grid
# ---------------------------------------------------------------------------

def create_language_diff_grid(frames, lang_diff_results, episode_idx=0,
                               output_path="language_diff.png"):
    """
    Four-row grid: thumbnail reference, original GradCAM, alternative GradCAM, difference.

    Args:
        frames: list of image tensors
        lang_diff_results: list (per frame) of dicts with
            ``"original_cam"``, ``"alt_cam"``, ``"diff_cam"``,
            ``"original_task"``, ``"alt_task"``, or *None*.
        output_path: where to save the PNG.
    """
    n_frames = len(frames)
    n_content_rows = 3  # original, alt, diff
    n_rows = 1 + n_content_rows  # +thumbnail
    height_ratios = [0.5] + [1.0] * n_content_rows

    figsize, dpi = _adaptive_figsize(3, 3, n_frames, n_content_rows + 1)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_frames, hspace=0.3, wspace=0.05,
                           height_ratios=height_ratios)

    row_labels = ["Original\nGradCAM", "Alt task\nGradCAM", "Difference\n(orig \u2212 alt)"]
    row_cmaps = ["magma", "magma", "RdBu_r"]

    # Extract task strings from first non-None result
    orig_task = alt_task = ""
    for r in lang_diff_results:
        if r is not None:
            orig_task = r.get("original_task", "")
            alt_task = r.get("alt_task", "")
            break

    # Thumbnail reference row
    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        ax = fig.add_subplot(gs[0, fi])
        ax.imshow(frame_np)
        ax.set_title(f"Frame {fi}", fontsize=9)
        ax.axis("off")
        if fi == 0:
            ax.set_ylabel("Original", fontsize=9, rotation=0, labelpad=60, va="center")

    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]
        result = lang_diff_results[fi] if fi < len(lang_diff_results) else None

        cam_keys = ["original_cam", "alt_cam", "diff_cam"]
        for ri in range(n_content_rows):
            ax = fig.add_subplot(gs[ri + 1, fi])
            if result is not None and cam_keys[ri] in result:
                hm = _resize_heatmap(result[cam_keys[ri]], h, w)
                if cam_keys[ri] == "diff_cam":
                    # Symmetric [-1, 1] → [0, 1] for RdBu_r
                    cmap_rdbu = plt.get_cmap("RdBu_r")
                    diff_colored = cmap_rdbu((hm + 1) / 2)[:, :, :3]
                    diff_colored = (diff_colored * 255).astype(np.uint8)
                    blended = (
                        0.55 * frame_np.astype(np.float32) +
                        0.45 * diff_colored.astype(np.float32)
                    ).astype(np.uint8)
                else:
                    blended = overlay_heatmap(frame_np, hm, alpha=0.45, colormap=row_cmaps[ri])
                ax.imshow(blended)
            else:
                ax.imshow(frame_np)
            ax.axis("off")
            if fi == 0:
                ax.set_ylabel(row_labels[ri], fontsize=10, rotation=0,
                              labelpad=60, va="center")

    title = f"Language-conditional comparison \u2014 Episode {episode_idx}"
    if orig_task:
        title += f"\nOriginal: \"{orig_task}\""
    if alt_task:
        title += f"  |  Alt: \"{alt_task}\""
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved language diff grid: {output_path}")


# ---------------------------------------------------------------------------
# Vision vs. State stacked bar chart
# ---------------------------------------------------------------------------

def create_vision_vs_state_chart(vision_vs_state_results, episode_idx=0,
                                  output_path="vision_vs_state.png"):
    """
    Stacked bar chart showing vision vs. state attribution share per frame.

    Args:
        vision_vs_state_results: list of dicts with ``"vision_share"`` or *None*.
        episode_idx: episode index for the title.
        output_path: where to save the PNG.
    """
    n_frames = len(vision_vs_state_results)
    vision_shares = []
    state_shares = []
    frame_labels = []

    for fi, r in enumerate(vision_vs_state_results):
        if r is not None:
            vision_shares.append(r["vision_share"])
            state_shares.append(1.0 - r["vision_share"])
        else:
            vision_shares.append(0.0)
            state_shares.append(0.0)
        frame_labels.append(f"Frame {fi}")

    fig, ax = plt.subplots(figsize=(max(6, n_frames * 1.2), 4))
    x = np.arange(n_frames)
    bar_width = 0.6

    ax.bar(x, vision_shares, bar_width, label="Vision", color="#3B3BD3", alpha=0.85)
    ax.bar(x, state_shares, bar_width, bottom=vision_shares, label="State", color="#E87722", alpha=0.85)

    # Average vision share line
    valid_shares = [v for v, r in zip(vision_shares, vision_vs_state_results) if r is not None]
    if valid_shares:
        avg_v = np.mean(valid_shares)
        ax.axhline(y=avg_v, color="#3B3BD3", linestyle="--", linewidth=1.5,
                    alpha=0.7, label=f"Avg vision={avg_v:.0%}")

    ax.set_xticks(x)
    ax.set_xticklabels(frame_labels, fontsize=9)
    ax.set_ylabel("Share", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Vision vs. State Attribution \u2014 Episode {episode_idx}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved vision vs state chart: {output_path}")

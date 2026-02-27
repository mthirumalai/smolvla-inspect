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
                               connector_gradcam_maps=None,
                               language_diff_maps=None,
                               episode_idx=0, output_path="attention_grid.png",
                               smooth_n=1):
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
    has_connector = connector_gradcam_maps is not None and len(connector_gradcam_maps) == n_frames
    has_lang_diff = language_diff_maps is not None and len(language_diff_maps) == n_frames

    # Dynamic row count
    n_rows = 3
    if has_cross:
        n_rows += 2
    if has_saliency:
        n_rows += 1
    if has_gradcam:
        n_rows += 1
    if has_connector:
        n_rows += 1
    if has_lang_diff:
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

        conn_overlay = None
        if has_connector:
            conn_hm = _resize_heatmap(connector_gradcam_maps[i], h, w)
            conn_overlay = overlay_heatmap(frame_np, conn_hm, alpha=0.45, colormap="magma")

        lang_diff_overlay = None
        if has_lang_diff and language_diff_maps[i] is not None:
            diff_hm = _resize_heatmap(language_diff_maps[i]["diff_cam"], h, w)
            lang_diff_overlay = overlay_heatmap(frame_np, diff_hm, alpha=0.45, colormap="RdBu_r")

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
            sal_label = f"SmoothGrad\nN={smooth_n}" if smooth_n > 1 else "Saliency\n|dA/dpx|"
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(sal_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(sal_label, fontsize=11, rotation=0, labelpad=60, va="center")
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
            # GradCAM overlay
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(gc_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("GradCAM\nSigLIP L-1", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

        if has_connector:
            # Connector GradCAM overlay
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(conn_overlay)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("GradCAM\nConnector", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

        if has_lang_diff:
            # Language-conditional diff overlay
            ax = fig.add_subplot(gs[row, i])
            if lang_diff_overlay is not None:
                ax.imshow(lang_diff_overlay)
            else:
                ax.imshow(frame_np)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Lang-cond\ndiff", fontsize=11, rotation=0, labelpad=60, va="center")
            row += 1

    # --- Build legend string dynamically ---
    legend_parts = [
        "Original",
        "SigLIP self-attn heatmap",
    ]
    if has_cross:
        legend_parts.append("Action cross-attn heatmap")
    if has_saliency:
        legend_parts.append(f"SmoothGrad N={smooth_n}" if smooth_n > 1 else "Saliency |dA/dpx|")
    legend_parts.append("Self-attn overlay")
    if has_cross:
        legend_parts.append("Co-attention (self \u00d7 cross)")
    if has_gradcam:
        legend_parts.append("GradCAM SigLIP L-1")
    if has_connector:
        legend_parts.append("GradCAM Connector")
    if has_lang_diff:
        legend_parts.append("Lang-conditional diff")

    legend = "  |  ".join(f"Row {j+1}: {lbl}" for j, lbl in enumerate(legend_parts))

    extra_lines = []
    if has_cross:
        extra_lines.append("Co-attention: self-attn \u00d7 cross-attn \u2014 bright regions are both visually salient and action-relevant")
    if has_saliency or has_gradcam:
        extra_lines.append("Gradient rows show which image regions causally influence the predicted action")
    if has_connector:
        extra_lines.append("Connector GradCAM: attribution on post-pixel-shuffle tokens (8\u00d78)")
    if has_lang_diff:
        extra_lines.append("Lang-cond diff: regions more important for original task vs. alternative")

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


# ---------------------------------------------------------------------------
# F3: Per-denoising-step cross-attention grid
# ---------------------------------------------------------------------------

def create_per_step_cross_attn_grid(frames, per_step_maps, episode_idx=0,
                                     output_path="per_step_cross_attn.png"):
    """
    Visualize cross-attention across denoising steps.

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

    fig, axes = plt.subplots(n_steps, n_frames, figsize=(3 * n_frames, 3 * n_steps))
    if n_steps == 1:
        axes = axes[np.newaxis, :] if n_frames > 1 else np.array([[axes]])
    if n_frames == 1:
        axes = axes[:, np.newaxis] if n_steps > 1 else np.array([[axes]])

    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]

        step_maps = per_step_maps[fi] if fi < len(per_step_maps) else []
        for si in range(n_steps):
            ax = axes[si, fi]
            if si < len(step_maps):
                scores = step_maps[si]
                n_vis = scores.shape[0]
                side = int(n_vis ** 0.5)
                hm_small = scores.reshape(side, side).detach().float().cpu().numpy()
                hm_small = hm_small / (hm_small.max() + 1e-8)
                hm = _resize_heatmap(hm_small, h, w)
                blended = overlay_heatmap(frame_np, hm, alpha=0.45, colormap="Greens")
                ax.imshow(blended)
            else:
                ax.imshow(frame_np)

            ax.axis("off")
            if fi == 0:
                t_val = 1.0 - si * 0.1 if n_steps == 10 else si
                ax.set_ylabel(f"Step {si}\n(t={t_val:.1f})", fontsize=9,
                              rotation=0, labelpad=50, va="center")
            if si == 0:
                ax.set_title(f"Frame {fi}", fontsize=9)

    fig.suptitle(f"Per-denoising-step cross-attention \u2014 Episode {episode_idx}",
                 fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved per-step cross-attn grid: {output_path}")


# ---------------------------------------------------------------------------
# F2: VLM layer GradCAM grid
# ---------------------------------------------------------------------------

def create_vlm_layer_grid(frames, vlm_layer_results, layer_indices,
                           episode_idx=0, output_path="vlm_layers.png",
                           lang_tokens=None):
    """
    Visualize GradCAM at multiple VLM intermediate layers.

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

    fig, axes = plt.subplots(n_layers, n_frames, figsize=(3 * n_frames, 3 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :] if n_frames > 1 else np.array([[axes]])
    if n_frames == 1:
        axes = axes[:, np.newaxis] if n_layers > 1 else np.array([[axes]])

    for li_idx, li in enumerate(layer_indices):
        layer_data = vlm_layer_results.get(li, [])
        for fi, frame in enumerate(frames):
            ax = axes[li_idx, fi]
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
            if li_idx == 0:
                ax.set_title(f"Frame {fi}", fontsize=9)

    fig.suptitle(f"VLM intermediate layer GradCAM \u2014 Episode {episode_idx}",
                 fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved VLM layer grid: {output_path}")


# ---------------------------------------------------------------------------
# F6: Per-action-dimension GradCAM grid
# ---------------------------------------------------------------------------

def create_per_action_dim_grid(frames, per_dim_maps, action_dim_names=None,
                                episode_idx=0, output_path="per_action_dim.png"):
    """
    Visualize GradCAM for each action dimension.

    Args:
        frames: list of image tensors
        per_dim_maps: list (per frame) of lists (per action dim) of numpy
            arrays ``(grid_h, grid_w)``, or *None* for failed frames.
        action_dim_names: optional list of dimension name strings.
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

    fig, axes = plt.subplots(n_dims, n_frames, figsize=(3 * n_frames, 3 * n_dims))
    if n_dims == 1:
        axes = axes[np.newaxis, :] if n_frames > 1 else np.array([[axes]])
    if n_frames == 1:
        axes = axes[:, np.newaxis] if n_dims > 1 else np.array([[axes]])

    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]

        dim_maps = per_dim_maps[fi] if fi < len(per_dim_maps) and per_dim_maps[fi] is not None else None

        for di in range(n_dims):
            ax = axes[di, fi]
            if dim_maps is not None and di < len(dim_maps):
                hm = _resize_heatmap(dim_maps[di], h, w)
                blended = overlay_heatmap(frame_np, hm, alpha=0.45, colormap="magma")
                ax.imshow(blended)
            else:
                ax.imshow(frame_np)
            ax.axis("off")
            if fi == 0:
                name = action_dim_names[di] if di < len(action_dim_names) else f"dim_{di}"
                ax.set_ylabel(name, fontsize=9, rotation=0, labelpad=60, va="center")
            if di == 0:
                ax.set_title(f"Frame {fi}", fontsize=9)

    fig.suptitle(f"Per-action-dimension GradCAM \u2014 Episode {episode_idx}",
                 fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved per-action-dim grid: {output_path}")


# ---------------------------------------------------------------------------
# F5: Language-conditional comparison grid
# ---------------------------------------------------------------------------

def create_language_diff_grid(frames, lang_diff_results, episode_idx=0,
                               output_path="language_diff.png"):
    """
    Three-row grid: original GradCAM, alternative GradCAM, difference.

    Args:
        frames: list of image tensors
        lang_diff_results: list (per frame) of dicts with
            ``"original_cam"``, ``"alt_cam"``, ``"diff_cam"``,
            ``"original_task"``, ``"alt_task"``, or *None*.
        output_path: where to save the PNG.
    """
    n_frames = len(frames)
    n_rows = 3  # original, alt, diff

    fig, axes = plt.subplots(n_rows, n_frames, figsize=(3 * n_frames, 3 * n_rows))
    if n_frames == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Original\nGradCAM", "Alt task\nGradCAM", "Difference\n(orig - alt)"]
    row_cmaps = ["magma", "magma", "RdBu_r"]

    # Extract task strings from first non-None result
    orig_task = alt_task = ""
    for r in lang_diff_results:
        if r is not None:
            orig_task = r.get("original_task", "")
            alt_task = r.get("alt_task", "")
            break

    for fi, frame in enumerate(frames):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]
        result = lang_diff_results[fi] if fi < len(lang_diff_results) else None

        cam_keys = ["original_cam", "alt_cam", "diff_cam"]
        for ri in range(n_rows):
            ax = axes[ri, fi]
            if result is not None and cam_keys[ri] in result:
                hm = _resize_heatmap(result[cam_keys[ri]], h, w)
                blended = overlay_heatmap(frame_np, hm, alpha=0.45, colormap=row_cmaps[ri])
                ax.imshow(blended)
            else:
                ax.imshow(frame_np)
            ax.axis("off")
            if fi == 0:
                ax.set_ylabel(row_labels[ri], fontsize=10, rotation=0,
                              labelpad=60, va="center")
            if ri == 0:
                ax.set_title(f"Frame {fi}", fontsize=9)

    title = f"Language-conditional comparison \u2014 Episode {episode_idx}"
    if orig_task:
        title += f"\nOriginal: \"{orig_task}\""
    if alt_task:
        title += f"  |  Alt: \"{alt_task}\""
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved language diff grid: {output_path}")

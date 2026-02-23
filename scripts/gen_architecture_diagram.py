#!/usr/bin/env python3
"""
Generate the "How it works" architecture + hooks + attention-to-heatmap diagram.
Saves to assets/how_it_works_architecture.png (3Blue1Brown-style).
"""
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 3Blue1Brown-style palette
BG = "#fafafa"
BLUE_DARK = "#1a237e"
BLUE_MID = "#3949ab"
BLUE_LIGHT = "#e8eaf6"
PURPLE = "#5e35b1"
PURPLE_LIGHT = "#ede7f6"
GRAY = "#e0e0e0"
HOOK_COLOR = "#b71c1c"
ARROW_COLOR = "#455a64"  # soft slate so arrows don't compete with boxes

# Generous spacing so arrows sit in clear "lanes" between blocks
GAP = 1.05   # vertical space reserved between blocks (arrow runs in the middle)
ARROW_INSET = 0.22  # arrow starts/ends this far from box edge (no overlap)


def draw_box(ax, x_center, y_center, w, h, text, box_kw, fontsize=10):
    """Draw a rounded box centered at (x_center, y_center). Returns (top, bottom) y."""
    left = x_center - w / 2
    bottom = y_center - h / 2
    rect = FancyBboxPatch((left, bottom), w, h, **box_kw)
    ax.add_patch(rect)
    ax.text(x_center, y_center, text, ha="center", va="center", fontsize=fontsize, color=BLUE_DARK)
    return y_center + h / 2, y_center - h / 2


def draw_arrow_down(ax, x, y_from, y_to, color=ARROW_COLOR):
    """Draw a clean vertical arrow from y_from down to y_to. Arrow stays clear of both y values."""
    y_start = y_from - ARROW_INSET
    y_end = y_to + ARROW_INSET
    arrow = FancyArrowPatch(
        (x, y_start),
        (x, y_end),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2,
        color=color,
        connectionstyle="arc3,rad=0",
        zorder=1,
    )
    ax.add_patch(arrow)


def main():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)
    for ax in (ax_left, ax_right):
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 17)

    box_kw = dict(
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE_MID,
        linewidth=1.4,
        boxstyle="round,pad=0.35,rounding_size=0.2",
    )
    cx, w = 5, 6.5

    # ---- Left: Architecture + hooks ----
    ax = ax_left
    y = 16.0

    # Camera image
    _, bottom = draw_box(ax, cx, y, w, 0.9, "Camera image (H×W×3)", box_kw, 11)
    y_next = bottom - GAP
    draw_arrow_down(ax, cx, bottom, y_next)
    y = y_next - GAP

    # Patch embedding
    _, bottom = draw_box(ax, cx, y, w, 0.75, "Patch embedding\n(image → patches)", box_kw, 10)
    y_next = bottom - GAP
    draw_arrow_down(ax, cx, bottom, y_next)
    y = y_next - GAP

    # Vision encoder section label
    ax.text(cx, y + 0.35, "Vision encoder (Transformer)", ha="center", va="center",
            fontsize=10, color=BLUE_DARK, fontweight="bold")
    y -= 0.5

    layer_h = 0.5
    for label in ["Layer 1", "Layer 2", "…", "Layer L (last)"]:
        fill = PURPLE_LIGHT if "Layer L" in label else GRAY
        edge = PURPLE if "Layer L" in label else "#9e9e9e"
        _, bottom = draw_box(
            ax, cx, y, 5.2, layer_h, label,
            dict(facecolor=fill, edgecolor=edge, linewidth=1.2, boxstyle="round,pad=0.2,rounding_size=0.15"),
            fontsize=9,
        )
        if "Layer L" in label:
            # Single callout box for the hook (no overlapping text/lines)
            callout_x, callout_w, callout_h = 8.0, 1.65, 0.58
            callout_left = callout_x - callout_w / 2
            callout_bottom = y - callout_h / 2
            callout = FancyBboxPatch(
                (callout_left, callout_bottom), callout_w, callout_h,
                facecolor="#ffebee", edgecolor=HOOK_COLOR, linewidth=1.0,
                boxstyle="round,pad=0.12,rounding_size=0.08",
            )
            ax.add_patch(callout)
            ax.text(callout_x, y + 0.08, "HOOK:", fontsize=7, color=HOOK_COLOR, ha="center", fontweight="bold")
            ax.text(callout_x, y - 0.12, "capture attn", fontsize=6.5, color=HOOK_COLOR, ha="center")
            ax.text(callout_x, y - 0.28, "weights", fontsize=6.5, color=HOOK_COLOR, ha="center")
            # Connector: short line from Layer L box edge to callout
            ax.plot([5 + 2.6, callout_left - 0.02], [y, y], color="#9e9e9e", lw=0.9, solid_capstyle="round")
        y = bottom - 0.22
    y_next = y - GAP
    draw_arrow_down(ax, cx, bottom, y_next)
    y = y_next - GAP

    # Patch features
    _, bottom = draw_box(ax, cx, y, w, 0.65, "Patch features", box_kw, 10)
    y_next = bottom - GAP
    draw_arrow_down(ax, cx, bottom, y_next)
    y = y_next - GAP

    # Action head
    draw_box(
        ax, cx, y, 5.5, 0.6, "Action head → Robot actions",
        dict(facecolor=PURPLE_LIGHT, edgecolor=PURPLE, linewidth=1.2, boxstyle="round,pad=0.25,rounding_size=0.15"),
        10,
    )
    ax.set_title("Model architecture & where we hook", fontsize=13, fontweight="bold", color=BLUE_DARK, pad=14)

    # ---- Right: Attention → heatmap pipeline ----
    ax = ax_right
    y = 16.0
    step_h = 0.82
    steps = [
        ("Attention weights\n(patch × patch matrix)", BLUE_LIGHT, BLUE_MID),
        ("Per-patch importance\n(e.g. mean over rows)", BLUE_LIGHT, BLUE_MID),
        ("2D grid (e.g. 24×24)", PURPLE_LIGHT, PURPLE),
        ("Upsample to image size\n(bilinear interpolation)", BLUE_LIGHT, BLUE_MID),
        ("Normalize [0, 1]", BLUE_LIGHT, BLUE_MID),
        ("Heatmap on image\n(overlay)", PURPLE_LIGHT, PURPLE),
    ]
    for i, (label, face, edge) in enumerate(steps):
        _, bottom = draw_box(
            ax, cx, y, w, step_h, label,
            dict(facecolor=face, edgecolor=edge, linewidth=1.2, boxstyle="round,pad=0.25,rounding_size=0.15"),
            10,
        )
        if i < len(steps) - 1:
            y_next = bottom - GAP
            draw_arrow_down(ax, cx, bottom, y_next)
            y = y_next - GAP
    ax.set_title("Attention → spatial heatmap", fontsize=13, fontweight="bold", color=BLUE_DARK, pad=14)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "assets", "how_it_works_architecture.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the how_it_works_architecture.png diagram for the README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, axes = plt.subplots(1, 3, figsize=(22, 14))
for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

# ── Colors ──
BLUE_BG = "#EDECFB"
BLUE_BORDER = "#3B3BD3"
GREEN_BG = "#E8F5E9"
GREEN_BORDER = "#388E3C"
CYAN_BG = "#E0F7FA"
CYAN_BORDER = "#00838F"
RED_BG = "#FFEBEE"
RED_BORDER = "#C62828"
GRAY_BG = "#F5F5F5"
GRAY_BORDER = "#9E9E9E"
TEXT_COLOR = "#3F4547"
HOOK_COLOR = "#C62828"

def box(ax, x, y, w, h, label, bg=BLUE_BG, border=BLUE_BORDER, fontsize=9,
        fontstyle="normal", fontweight="normal", ha="center"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=bg, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha=ha if ha == "center" else "center",
            va="center", fontsize=fontsize, color=TEXT_COLOR,
            fontweight=fontweight, fontstyle=fontstyle, wrap=True)

def arrow(ax, x1, y1, x2, y2, color=BLUE_BORDER):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=color, lw=1.5))

def hook_label(ax, x, y, label, color=HOOK_COLOR):
    rect = FancyBboxPatch((x, y), 2.2, 0.6, boxstyle="round,pad=0.1",
                          facecolor=RED_BG, edgecolor=color, linewidth=1.2,
                          linestyle="--")
    ax.add_patch(rect)
    ax.text(x + 1.1, y + 0.3, label, ha="center", va="center",
            fontsize=7.5, color=color, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════
# COLUMN 1: Model Architecture & Where We Intercept
# ═══════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_title("Model architecture & where we intercept", fontsize=13,
             fontweight="bold", color=BLUE_BORDER, pad=15)

# Camera image
box(ax, 2, 14.5, 6, 0.8, "Camera image (H x W x 3)", bg=GRAY_BG, border=GRAY_BORDER)
arrow(ax, 5, 14.5, 5, 14.0)

# SigLIP vision encoder block
encoder_rect = FancyBboxPatch((1.5, 11.0), 7, 2.8, boxstyle="round,pad=0.2",
                               facecolor="#F3F2FF", edgecolor=BLUE_BORDER,
                               linewidth=2, linestyle="-")
ax.add_patch(encoder_rect)
ax.text(5, 13.6, "SigLIP Vision Encoder", ha="center", va="center",
        fontsize=10, color=BLUE_BORDER, fontweight="bold")
ax.text(5, 13.1, "(12 layers, 12 heads, no CLS token)", ha="center", va="center",
        fontsize=8, color="#7F8385")

# Layers inside encoder
box(ax, 2.5, 12.2, 5, 0.5, "Patch embedding  (512px / 16px = 32x32 = 1024 patches)",
    fontsize=7.5)
arrow(ax, 5, 12.2, 5, 11.9)
box(ax, 2.5, 11.2, 5, 0.6, "Self-attention layers 1 ... 12",
    fontsize=8, fontweight="bold")

# Hook on encoder
hook_label(ax, 6.8, 11.3, "HOOK: fwd hook\non attn layers")

arrow(ax, 5, 11.0, 5, 10.5)

# Connector
box(ax, 2, 9.8, 6, 0.7, "Connector (pixel shuffle)\n1024 patches  ->  64 vision tokens",
    fontsize=8)
arrow(ax, 5, 9.8, 5, 9.3)

# VLM / SmolLM2
vlm_rect = FancyBboxPatch((1.5, 7.0), 7, 2.1, boxstyle="round,pad=0.2",
                            facecolor="#F3F2FF", edgecolor=BLUE_BORDER,
                            linewidth=2, linestyle="-")
ax.add_patch(vlm_rect)
ax.text(5, 8.8, "SmolLM2 (VLM)", ha="center", va="center",
        fontsize=10, color=BLUE_BORDER, fontweight="bold")

# Prefix tokens inside VLM
box(ax, 2.2, 7.3, 2, 0.7, "64 vision\ntokens", fontsize=7.5, bg="#E8EAF6", border="#5C6BC0")
box(ax, 4.3, 7.3, 2, 0.7, "language\ntokens", fontsize=7.5, bg="#E8EAF6", border="#5C6BC0")
box(ax, 6.5, 7.3, 1.7, 0.7, "state\ntokens", fontsize=7.5, bg="#E8EAF6", border="#5C6BC0")

ax.text(5, 8.2, "Self-attention over prefix  ->  builds KV cache",
        ha="center", va="center", fontsize=8, color="#7F8385")

# Arrow from VLM down: KV cache
arrow(ax, 5, 7.0, 5, 6.5)
ax.text(5.1, 6.7, "KV cache", ha="left", va="center", fontsize=8,
        color=GREEN_BORDER, fontweight="bold")

# Action expert
expert_rect = FancyBboxPatch((1.5, 3.8), 7, 2.5, boxstyle="round,pad=0.2",
                              facecolor="#E8F5E9", edgecolor=GREEN_BORDER,
                              linewidth=2, linestyle="-")
ax.add_patch(expert_rect)
ax.text(5, 6.0, "Action Expert", ha="center", va="center",
        fontsize=10, color=GREEN_BORDER, fontweight="bold")
ax.text(5, 5.5, "Q = expert action tokens", ha="center", va="center",
        fontsize=8, color=TEXT_COLOR)
ax.text(5, 5.0, "K, V = VLM prefix KV cache", ha="center", va="center",
        fontsize=8, color=TEXT_COLOR)
ax.text(5, 4.4, "Cross-attn: Q_expert attends to K_prefix\n(detected when Q_len != K_len)",
        ha="center", va="center", fontsize=7.5, color="#7F8385")

# Monkey-patch label
hook_label(ax, 6.8, 4.0, "MONKEY-PATCH:\neager_attn_fwd")

arrow(ax, 5, 3.8, 5, 3.3)

# Action output
box(ax, 2, 2.5, 6, 0.7, "Action chunk  ->  Robot actions",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=9, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════
# COLUMN 2: Self-Attention -> Heatmap Pipeline
# ═══════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_title("Self-attention  ->  heatmap", fontsize=13,
             fontweight="bold", color=BLUE_BORDER, pad=15)

box(ax, 2, 14.2, 6, 0.8, "Attention weights\n(heads, patches, patches)",
    fontsize=9)
arrow(ax, 5, 14.2, 5, 13.6)

box(ax, 2, 12.8, 6, 0.8, "Aggregation method:", fontsize=9, fontweight="bold")
ax.text(5, 12.5, "last-layer: use layer 12 only\n"
        "rollout: multiply across all 12 layers\n"
        "all-layers: keep each layer separate",
        ha="center", va="top", fontsize=7.5, color="#7F8385")
arrow(ax, 5, 11.6, 5, 11.2)

box(ax, 2, 10.4, 6, 0.8, "Average across 12 heads\n-> per-patch importance (1024,)",
    fontsize=8.5)
arrow(ax, 5, 10.4, 5, 9.8)

box(ax, 2, 9.0, 6, 0.8, "Reshape to 2D grid (32 x 32)", fontsize=9)
arrow(ax, 5, 9.0, 5, 8.4)

box(ax, 2, 7.6, 6, 0.8, "Bilinear upsample to image size\n(32x32  ->  512x512)",
    fontsize=8.5)
arrow(ax, 5, 7.6, 5, 7.0)

box(ax, 2, 6.2, 6, 0.8, "Normalize to [0, 1]", fontsize=9)
arrow(ax, 5, 6.2, 5, 5.6)

box(ax, 2, 4.8, 6, 0.8, "Optional: subtract positional baseline\n(--raw-attention skips this)",
    fontsize=8, fontstyle="italic")
arrow(ax, 5, 4.8, 5, 4.2)

box(ax, 2, 3.4, 6, 0.8, "Self-attention heatmap\n(jet colormap: blue to red)",
    bg="#E8EAF6", border="#5C6BC0", fontsize=9, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════
# COLUMN 3: Cross-Attention -> Heatmap + Co-attention
# ═══════════════════════════════════════════════════════════════════════
ax = axes[2]
ax.set_title("Cross-attention  ->  heatmap", fontsize=13,
             fontweight="bold", color=GREEN_BORDER, pad=15)

box(ax, 2, 14.2, 6, 0.8, "Intercepted softmax probs\n(expert_Q_len, prefix_K_len)",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=8.5)
arrow(ax, 5, 14.2, 5, 13.6, color=GREEN_BORDER)

box(ax, 2, 12.8, 6, 0.8, "Slice vision-token columns only\n(expert_Q_len, 64)",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=8.5)
arrow(ax, 5, 12.8, 5, 12.2, color=GREEN_BORDER)

box(ax, 2, 11.4, 6, 0.8, "Average across expert layers\nand query positions  ->  (64,)",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=8.5)
arrow(ax, 5, 11.4, 5, 10.8, color=GREEN_BORDER)

box(ax, 2, 10.0, 6, 0.8, "Undo pixel shuffle: reshape\n64 tokens  ->  8x8 grid",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=8.5)
arrow(ax, 5, 10.0, 5, 9.4, color=GREEN_BORDER)

box(ax, 2, 8.6, 6, 0.8, "Bilinear upsample to image size\n(8x8  ->  512x512)",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=8.5)
arrow(ax, 5, 8.6, 5, 8.0, color=GREEN_BORDER)

box(ax, 2, 7.2, 6, 0.8, "Normalize to [0, 1]",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=9)
arrow(ax, 5, 7.2, 5, 6.6, color=GREEN_BORDER)

box(ax, 2, 5.8, 6, 0.8, "Cross-attention heatmap\n(Greens colormap)",
    bg=GREEN_BG, border=GREEN_BORDER, fontsize=9, fontweight="bold")

# Co-attention
arrow(ax, 5, 5.8, 5, 5.2, color=CYAN_BORDER)

# Show the multiplication
ax.text(1.8, 4.8, "self-attn", ha="center", va="center", fontsize=8,
        color=BLUE_BORDER, fontweight="bold")
ax.text(2.8, 4.8, " x ", ha="center", va="center", fontsize=10,
        color=TEXT_COLOR, fontweight="bold")
ax.text(3.8, 4.8, "cross-attn", ha="center", va="center", fontsize=8,
        color=GREEN_BORDER, fontweight="bold")

box(ax, 2, 3.8, 6, 0.8, "Co-attention heatmap\n(cyan colormap: black -> cyan -> white)",
    bg=CYAN_BG, border=CYAN_BORDER, fontsize=9, fontweight="bold")
arrow(ax, 3.5, 5.2, 5, 4.6, color=CYAN_BORDER)

ax.text(5, 3.2, "Bright cyan = both visually salient\nAND action-relevant",
        ha="center", va="center", fontsize=8, color=CYAN_BORDER, fontstyle="italic")

plt.tight_layout(w_pad=2)
plt.savefig("/Users/subirmansukhani/Desktop/smolvla-inspect/assets/how_it_works_architecture.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved how_it_works_architecture.png")

"""
Model internals report — spectral alpha, attention entropy, head redundancy.

Imports from ``capture`` and ``data``.
"""

import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .capture import SigLIPAttentionCapture, DecoderAttentionCapture
from .data import find_vision_encoder, find_image_keys, get_episode_frames, build_policy_batch_from_sample


def compute_weightwatcher_alpha(policy):
    """
    Run WeightWatcher spectral analysis on each trainable (and frozen
    reference) component of SmolVLA.

    Returns dict keyed by component name, values are lists of
    ``{layer, alpha}`` dicts.  Components that are too small for
    reliable SVD are reported with ``alpha=None``.
    """
    try:
        import weightwatcher as ww
    except ImportError:
        print("  ERROR: weightwatcher not installed. Run:  pip install weightwatcher")
        print("  Skipping spectral alpha analysis.")
        return None

    components = {}

    # --- helper: run ww on a submodel, return list of {layer, alpha} ---
    def _analyze(name, submodel):
        try:
            watcher = ww.WeightWatcher(model=submodel)
            details = watcher.analyze(min_evals=50)
            results = []
            for idx, row in details.iterrows():
                results.append({"layer": idx, "alpha": row.get("alpha", None)})
            components[name] = results
        except Exception as e:
            print(f"    WARNING: WeightWatcher failed on {name}: {e}")
            components[name] = []

    vlm_with_expert = policy.model.vlm_with_expert

    # Trainable components
    print("  Analyzing expert layers...")
    _analyze("Expert (trainable)", vlm_with_expert.lm_expert)

    print("  Analyzing connector...")
    _analyze("Connector (trainable)", vlm_with_expert.get_vlm_model().connector)

    # Projection heads
    proj_names = [n for n, _ in policy.model.named_children()
                  if "proj" in n.lower()]
    if proj_names:
        print(f"  Analyzing projection heads ({', '.join(proj_names)})...")
        for pn in proj_names:
            _analyze(f"Projection/{pn} (trainable)", getattr(policy.model, pn))

    # Frozen reference components
    print("  Analyzing vision encoder (frozen reference)...")
    _analyze("Vision Encoder (frozen)", vlm_with_expert.get_vlm_model().vision_model)

    print("  Analyzing VLM text model (frozen reference)...")
    _analyze("VLM Text Model (frozen)", vlm_with_expert.get_vlm_model().text_model)

    return components


def compute_attention_entropy(attn_maps, num_patches):
    """
    Per-head, per-layer attention entropy as a fraction of maximum
    entropy (``log(num_patches)``).

    Args:
        attn_maps: list of ``(layer_idx, attn_weights)`` tuples.
            Each ``attn_weights`` has shape ``(batch, heads, patches, patches)``
            or ``(heads, patches, patches)``.
        num_patches: total number of patches (for max-entropy normalisation).

    Returns:
        list of ``{layer, head, entropy, entropy_ratio}`` dicts.
    """
    max_entropy = math.log(num_patches) if num_patches > 1 else 1.0
    eps = 1e-8
    results = []

    for layer_idx, attn in sorted(attn_maps, key=lambda x: x[0]):
        # Reduce to (heads, patches, patches)
        while attn.dim() > 3:
            attn = attn[0]
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)

        n_heads = attn.shape[0]
        for h in range(n_heads):
            head_attn = attn[h].float()  # (patches, patches)
            # Entropy per row, then average across rows
            ent = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1).mean().item()
            results.append({
                "layer": layer_idx,
                "head": h,
                "entropy": ent,
                "entropy_ratio": ent / max_entropy,
            })

    return results


def compute_head_redundancy(attn_maps):
    """
    Pairwise cosine similarity between flattened head attention patterns
    within each layer.

    Args:
        attn_maps: list of ``(layer_idx, attn_weights)`` tuples.

    Returns:
        list of ``{layer, mean_redundancy, max_redundancy}`` dicts.
    """
    results = []

    for layer_idx, attn in sorted(attn_maps, key=lambda x: x[0]):
        while attn.dim() > 3:
            attn = attn[0]
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)

        n_heads = attn.shape[0]
        if n_heads < 2:
            results.append({
                "layer": layer_idx,
                "mean_redundancy": 0.0,
                "max_redundancy": 0.0,
            })
            continue

        # Flatten each head's attention to a vector
        flat = attn.float().reshape(n_heads, -1)  # (heads, patches*patches)
        # Normalise
        flat_norm = F.normalize(flat, dim=1)
        # Pairwise cosine similarity
        sim = torch.mm(flat_norm, flat_norm.t())  # (heads, heads)

        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(n_heads, n_heads, dtype=torch.bool), diagonal=1)
        pairwise = sim[mask]
        results.append({
            "layer": layer_idx,
            "mean_redundancy": pairwise.mean().item(),
            "max_redundancy": pairwise.max().item(),
        })

    return results


def classify_status(alpha=None, entropy=None, redundancy=None, thresholds=None):
    """
    Classify a single metric value into a status string and ANSI color.

    Alpha thresholds (hardcoded, RMT-derived):
        <2 overcorrelated, 2-4 healthy, 4-6 undertrained, >6 severely undertrained

    Entropy/redundancy thresholds come from *thresholds* dict.

    Returns:
        ``(ansi_status_str, ansi_color_code, plain_label)``
    """
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if thresholds is None:
        thresholds = {}

    if alpha is not None:
        if alpha < 2:
            return f"{RED}overcorrelated{RESET}", RED, "overcorrelated"
        elif alpha <= 4:
            return f"{GREEN}healthy{RESET}", GREEN, "healthy"
        elif alpha <= 6:
            return f"{YELLOW}undertrained{RESET}", YELLOW, "undertrained"
        else:
            return f"{RED}severely undertrained{RESET}", RED, "severely undertrained"

    if entropy is not None:
        crit = thresholds.get("entropy_critical", 0.95)
        warn = thresholds.get("entropy_warn", 0.8)
        low = thresholds.get("entropy_low", 0.1)
        if entropy >= crit:
            return f"{RED}uniform/dead{RESET}", RED, "uniform/dead"
        elif entropy >= warn:
            return f"{YELLOW}unfocused{RESET}", YELLOW, "unfocused"
        elif entropy <= low:
            return f"{RED}collapsed{RESET}", RED, "collapsed"
        else:
            return f"{GREEN}healthy{RESET}", GREEN, "healthy"

    if redundancy is not None:
        r_crit = thresholds.get("redundancy_critical", 0.9)
        r_warn = thresholds.get("redundancy_warn", 0.7)
        if redundancy >= r_crit:
            return f"{RED}collapsed{RESET}", RED, "collapsed"
        elif redundancy >= r_warn:
            return f"{YELLOW}high redundancy{RESET}", YELLOW, "high redundancy"
        else:
            return f"{GREEN}diverse{RESET}", GREEN, "diverse"

    return "unknown", "", "unknown"


def _status_emoji(label):
    """Map a plain status label to a markdown-friendly indicator."""
    if label in ("healthy", "diverse"):
        return "OK"
    elif label in ("undertrained", "unfocused", "high redundancy"):
        return "WARN"
    else:
        return "CRITICAL"


def print_model_internals_report(ww_results, entropy_results, redundancy_results, thresholds):
    """
    Print a formatted terminal table with ANSI color-coded status per
    layer/component.
    """
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    print(f"\n{'=' * 78}")
    print(f"{BOLD}MODEL INTERNALS REPORT{RESET}")
    print(f"{'=' * 78}")

    # --- Section 1: Weight Spectral Analysis ---
    if ww_results is not None:
        print(f"\n{BOLD}1. Weight Spectral Analysis (alpha){RESET}")
        print(f"   Fits a power-law to each weight matrix's singular values.")
        print(f"   Alpha (\u03b1) measures how well-trained a layer is \u2014 values of 2-4 indicate")
        print(f"   strong correlation structure learned during training. High alpha means the")
        print(f"   layer hasn't learned enough structure; low alpha means overcorrelation.")
        print(f"   {DIM}Healthy: 2-4  |  Undertrained: 4-6  |  Overcorrelated: <2  |  Severe: >6{RESET}")
        _hdr_mean = "Mean \u03b1"
        _hdr_min = "Min \u03b1"
        _hdr_max = "Max \u03b1"
        print(f"   {'Component':<35} {'Wt Matrices':>11}  {_hdr_mean:>8}  {_hdr_min:>8}  {_hdr_max:>8}  Status")
        print(f"   {'-' * 77}")
        _dash = "\u2014"
        for comp_name, layers in ww_results.items():
            alphas = [l["alpha"] for l in layers if l.get("alpha") is not None]
            if not alphas:
                print(f"   {comp_name:<35} {_dash:>11}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {DIM}too small{RESET}")
                continue
            mean_a = sum(alphas) / len(alphas)
            min_a = min(alphas)
            max_a = max(alphas)
            status, _, _ = classify_status(alpha=mean_a)
            print(f"   {comp_name:<35} {len(layers):>11}  {mean_a:>8.2f}  {min_a:>8.2f}  {max_a:>8.2f}  {status}")
    else:
        print(f"\n{BOLD}1. Weight Spectral Analysis{RESET}")
        print(f"   {DIM}Skipped (weightwatcher not available){RESET}")

    # --- Section 2: Attention Entropy ---
    if entropy_results:
        print(f"\n{BOLD}2. Attention Entropy (fraction of max){RESET}")
        print(f"   Measures how spread out each attention head's focus is.")
        print(f"   Low entropy means the head attends to very few tokens (collapsed/dead).")
        print(f"   High entropy means the head spreads attention nearly uniformly (unfocused).")
        print(f"   Healthy heads are selective but not degenerate \u2014 attending to a meaningful subset.")
        print(f"   {DIM}Collapsed: <{thresholds.get('entropy_low', 0.1):.2f}  |  "
              f"Healthy: {thresholds.get('entropy_low', 0.1):.2f}-{thresholds.get('entropy_warn', 0.8):.2f}  |  "
              f"Unfocused: >{thresholds.get('entropy_warn', 0.8):.2f}  |  "
              f"Dead: >{thresholds.get('entropy_critical', 0.95):.2f}{RESET}")

        for comp_name, comp_entries in entropy_results.items():
            print(f"\n   {BOLD}{comp_name}{RESET}")
            by_layer = defaultdict(list)
            for e in comp_entries:
                by_layer[e["layer"]].append(e["entropy_ratio"])

            print(f"   {'Layer':>6}  {'Mean Ent':>10}  {'Min Ent':>10}  {'Max Ent':>10}  Status")
            print(f"   {'-' * 56}")
            for layer in sorted(by_layer.keys()):
                vals = by_layer[layer]
                mean_e = sum(vals) / len(vals)
                min_e = min(vals)
                max_e = max(vals)
                status, _, _ = classify_status(entropy=mean_e, thresholds=thresholds)
                print(f"   {layer:>6}  {mean_e:>10.4f}  {min_e:>10.4f}  {max_e:>10.4f}  {status}")
    else:
        print(f"\n{BOLD}2. Attention Entropy{RESET}")
        print(f"   {DIM}No data{RESET}")

    # --- Section 3: Head Redundancy ---
    if redundancy_results:
        print(f"\n{BOLD}3. Head Redundancy (cosine similarity){RESET}")
        print(f"   Measures how similar the attention heads are to each other within each layer.")
        print(f"   Each layer has multiple heads that should learn different patterns (e.g., one")
        print(f"   head for spatial relations, another for color). High similarity means heads are")
        print(f"   redundant \u2014 wasted capacity. Collapsed means nearly identical heads.")
        print(f"   {DIM}Diverse: <{thresholds.get('redundancy_warn', 0.7):.2f}  |  "
              f"High: >{thresholds.get('redundancy_warn', 0.7):.2f}  |  "
              f"Collapsed: >{thresholds.get('redundancy_critical', 0.9):.2f}{RESET}")

        for comp_name, comp_entries in redundancy_results.items():
            print(f"\n   {BOLD}{comp_name}{RESET}")
            print(f"   {'Layer':>6}  {'Mean Sim':>10}  {'Max Sim':>10}  Status")
            print(f"   {'-' * 46}")
            for r in comp_entries:
                status, _, _ = classify_status(redundancy=r["mean_redundancy"], thresholds=thresholds)
                print(f"   {r['layer']:>6}  {r['mean_redundancy']:>10.4f}  {r['max_redundancy']:>10.4f}  {status}")
    else:
        print(f"\n{BOLD}3. Head Redundancy{RESET}")
        print(f"   {DIM}No data{RESET}")

    print(f"\n{'=' * 78}\n")


def generate_model_internals_markdown(ww_results, entropy_results, redundancy_results, thresholds):
    """
    Build a Markdown report string from model internals results.
    """
    lines = []
    w = lines.append

    w("# Model Internals Report\n")

    # --- Section 1: Weight Spectral Analysis ---
    w("## 1. Weight Spectral Analysis (alpha)\n")
    w("Fits a power-law to each weight matrix's singular values. "
       "Alpha measures how well-trained a layer is \u2014 values of 2-4 indicate "
       "strong correlation structure learned during training. High alpha means the "
       "layer hasn't learned enough structure; low alpha means overcorrelation.\n")
    if ww_results is not None:
        w("> Healthy: 2-4 | Undertrained: 4-6 | Overcorrelated: <2 | Severe: >6\n")
        w("| Component | Wt Matrices | Mean alpha | Min alpha | Max alpha | Status |")
        w("|-----------|------------:|-----------:|----------:|----------:|--------|")
        for comp_name, layers in ww_results.items():
            alphas = [l["alpha"] for l in layers if l.get("alpha") is not None]
            if not alphas:
                w(f"| {comp_name} | -- | N/A | N/A | N/A | too small |")
                continue
            mean_a = sum(alphas) / len(alphas)
            min_a = min(alphas)
            max_a = max(alphas)
            _, _, label = classify_status(alpha=mean_a)
            badge = _status_emoji(label)
            w(f"| {comp_name} | {len(layers)} | {mean_a:.2f} | {min_a:.2f} | {max_a:.2f} | {badge} {label} |")
    else:
        w("*Skipped (weightwatcher not available)*\n")

    # --- Section 2: Attention Entropy ---
    w("\n## 2. Attention Entropy (fraction of max)\n")
    w("Measures how spread out each attention head's focus is. "
       "Low entropy means the head attends to very few tokens (collapsed/dead). "
       "High entropy means the head spreads attention nearly uniformly (unfocused). "
       "Healthy heads are selective but not degenerate \u2014 attending to a meaningful subset.\n")
    if entropy_results:
        low = thresholds.get("entropy_low", 0.1)
        warn = thresholds.get("entropy_warn", 0.8)
        crit = thresholds.get("entropy_critical", 0.95)
        w(f"> Collapsed: <{low:.2f} | Healthy: {low:.2f}-{warn:.2f} | Unfocused: >{warn:.2f} | Dead: >{crit:.2f}\n")

        for comp_name, comp_entries in entropy_results.items():
            w(f"\n### {comp_name}\n")
            by_layer = defaultdict(list)
            for e in comp_entries:
                by_layer[e["layer"]].append(e["entropy_ratio"])

            w("| Layer | Mean Ent | Min Ent | Max Ent | Status |")
            w("|------:|---------:|--------:|--------:|--------|")
            for layer in sorted(by_layer.keys()):
                vals = by_layer[layer]
                mean_e = sum(vals) / len(vals)
                min_e = min(vals)
                max_e = max(vals)
                _, _, label = classify_status(entropy=mean_e, thresholds=thresholds)
                badge = _status_emoji(label)
                w(f"| {layer} | {mean_e:.4f} | {min_e:.4f} | {max_e:.4f} | {badge} {label} |")
    else:
        w("*No data*\n")

    # --- Section 3: Head Redundancy ---
    w("\n## 3. Head Redundancy (cosine similarity)\n")
    w("Measures how similar the attention heads are to each other within each layer. "
       "Each layer has multiple heads that should learn different patterns (e.g., one "
       "head for spatial relations, another for color). High similarity means heads are "
       "redundant \u2014 wasted capacity. Collapsed means nearly identical heads.\n")
    if redundancy_results:
        r_warn = thresholds.get("redundancy_warn", 0.7)
        r_crit = thresholds.get("redundancy_critical", 0.9)
        w(f"> Diverse: <{r_warn:.2f} | High: >{r_warn:.2f} | Collapsed: >{r_crit:.2f}\n")

        for comp_name, comp_entries in redundancy_results.items():
            w(f"\n### {comp_name}\n")
            w("| Layer | Mean Sim | Max Sim | Status |")
            w("|------:|---------:|--------:|--------|")
            for r in comp_entries:
                _, _, label = classify_status(redundancy=r["mean_redundancy"], thresholds=thresholds)
                badge = _status_emoji(label)
                w(f"| {r['layer']} | {r['mean_redundancy']:.4f} | {r['max_redundancy']:.4f} | {badge} {label} |")
    else:
        w("*No data*\n")

    return "\n".join(lines)


def plot_model_internals_report(ww_results, entropy_results, redundancy_results, output_path):
    """
    3-panel vertical matplotlib figure saved to *output_path*.

    Panel 1: alpha per layer (bars) with reference lines at 2 and 6
    Panel 2: mean entropy ratio per layer
    Panel 3: mean head redundancy per layer
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    # --- Panel 1: Spectral Alpha ---
    ax1 = axes[0]
    if ww_results is not None:
        bar_labels = []
        bar_vals = []
        bar_colors = []
        separator_positions = []
        label_positions = []
        offset = 0

        for comp_name, layers in ww_results.items():
            alphas = [l["alpha"] for l in layers if l.get("alpha") is not None]
            if not alphas:
                continue
            comp_start = offset
            for i, a in enumerate(alphas):
                bar_labels.append(f"L{layers[i]['layer']}")
                bar_vals.append(a)
                if a < 2:
                    bar_colors.append("#e74c3c")       # red
                elif a <= 4:
                    bar_colors.append("#2ecc71")       # green
                elif a <= 6:
                    bar_colors.append("#f39c12")       # yellow
                else:
                    bar_colors.append("#e74c3c")       # red
                offset += 1
            comp_end = offset
            label_positions.append(((comp_start + comp_end - 1) / 2, comp_name))
            separator_positions.append(offset - 0.5)

        # Remove last separator
        if separator_positions:
            separator_positions.pop()

        if bar_vals:
            x = range(len(bar_vals))
            ax1.bar(x, bar_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax1.axhline(y=2, color="green", linestyle="--", linewidth=1, label="\u03b1=2 (lower healthy)")
            ax1.axhline(y=6, color="red", linestyle="--", linewidth=1, label="\u03b1=6 (upper healthy)")
            ax1.set_ylabel("Alpha (\u03b1)")
            ax1.legend(loc="upper right", fontsize=8)

            for sep_x in separator_positions:
                ax1.axvline(x=sep_x, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

            if len(bar_labels) > 30:
                step = max(1, len(bar_labels) // 20)
                ax1.set_xticks(range(0, len(bar_labels), step))
                ax1.set_xticklabels([bar_labels[i] for i in range(0, len(bar_labels), step)],
                                    rotation=45, ha="right", fontsize=6)
            else:
                ax1.set_xticks(x)
                ax1.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=6)

            # Component labels above bars
            for x_center, comp_label in label_positions:
                ax1.text(x_center, 1.02, comp_label, ha="center", va="bottom",
                         fontsize=7, fontweight="bold", transform=ax1.get_xaxis_transform())
        else:
            ax1.text(0.5, 0.5, "No alpha data (layers too small)",
                     ha="center", va="center", transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, "WeightWatcher not available",
                 ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Weight Spectral Analysis \u2014 Alpha per Layer", fontweight="bold", pad=20)

    # --- Panel 2: Attention Entropy (grouped by component) ---
    ax2 = axes[1]
    if entropy_results:
        bar_labels = []
        bar_means = []
        bar_colors = []
        separator_positions = []  # x positions for vertical lines between components
        label_positions = []      # (x_center, comp_name) for component labels
        offset = 0

        for comp_name, comp_entries in entropy_results.items():
            by_layer = defaultdict(list)
            for e in comp_entries:
                by_layer[e["layer"]].append(e["entropy_ratio"])
            layers_sorted = sorted(by_layer.keys())
            comp_start = offset

            for layer in layers_sorted:
                vals = by_layer[layer]
                mean_e = sum(vals) / len(vals)
                bar_labels.append(f"L{layer}")
                bar_means.append(mean_e)
                if mean_e >= 0.95:
                    bar_colors.append("#e74c3c")
                elif mean_e >= 0.8:
                    bar_colors.append("#f39c12")
                elif mean_e <= 0.1:
                    bar_colors.append("#e74c3c")
                else:
                    bar_colors.append("#2ecc71")
                offset += 1

            comp_end = offset
            label_positions.append(((comp_start + comp_end - 1) / 2, comp_name))
            if comp_end < sum(len(v) for v in [defaultdict(list)] * 0) or offset > 0:
                separator_positions.append(offset - 0.5)

        # Remove last separator (no line after the last component)
        if separator_positions:
            separator_positions.pop()

        ax2.bar(range(len(bar_means)), bar_means, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax2.axhline(y=0.8, color="#f39c12", linestyle="--", linewidth=1, label="warn (0.8)")
        ax2.axhline(y=0.95, color="#e74c3c", linestyle="--", linewidth=1, label="critical (0.95)")
        ax2.axhline(y=0.1, color="#e74c3c", linestyle=":", linewidth=1, label="collapsed (0.1)")

        # Vertical separators between components
        for sep_x in separator_positions:
            ax2.axvline(x=sep_x, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

        # Component labels at top
        for x_center, comp_label in label_positions:
            ax2.text(x_center, 1.02, comp_label, ha="center", va="bottom",
                     fontsize=7, fontweight="bold", transform=ax2.get_xaxis_transform())

        if len(bar_labels) > 40:
            step = max(1, len(bar_labels) // 30)
            ax2.set_xticks(range(0, len(bar_labels), step))
            ax2.set_xticklabels([bar_labels[i] for i in range(0, len(bar_labels), step)],
                                fontsize=6, rotation=45, ha="right")
        else:
            ax2.set_xticks(range(len(bar_labels)))
            ax2.set_xticklabels(bar_labels, fontsize=6, rotation=45, ha="right")
        ax2.set_ylabel("Entropy / max entropy")
        ax2.set_ylim(0, 1.12)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("Attention Entropy per Layer (mean across heads)", fontweight="bold", pad=20)

    # --- Panel 3: Head Redundancy (grouped by component) ---
    ax3 = axes[2]
    if redundancy_results:
        bar_labels = []
        bar_means = []
        bar_colors = []
        separator_positions = []
        label_positions = []
        offset = 0

        for comp_name, comp_entries in redundancy_results.items():
            comp_start = offset
            for r in comp_entries:
                bar_labels.append(f"L{r['layer']}")
                bar_means.append(r["mean_redundancy"])
                m = r["mean_redundancy"]
                if m >= 0.9:
                    bar_colors.append("#e74c3c")
                elif m >= 0.7:
                    bar_colors.append("#f39c12")
                else:
                    bar_colors.append("#2ecc71")
                offset += 1

            comp_end = offset
            label_positions.append(((comp_start + comp_end - 1) / 2, comp_name))
            separator_positions.append(offset - 0.5)

        # Remove last separator
        if separator_positions:
            separator_positions.pop()

        ax3.bar(range(len(bar_means)), bar_means, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax3.axhline(y=0.7, color="#f39c12", linestyle="--", linewidth=1, label="warn (0.7)")
        ax3.axhline(y=0.9, color="#e74c3c", linestyle="--", linewidth=1, label="critical (0.9)")

        for sep_x in separator_positions:
            ax3.axvline(x=sep_x, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

        for x_center, comp_label in label_positions:
            ax3.text(x_center, 1.02, comp_label, ha="center", va="bottom",
                     fontsize=7, fontweight="bold", transform=ax3.get_xaxis_transform())

        if len(bar_labels) > 40:
            step = max(1, len(bar_labels) // 30)
            ax3.set_xticks(range(0, len(bar_labels), step))
            ax3.set_xticklabels([bar_labels[i] for i in range(0, len(bar_labels), step)],
                                fontsize=6, rotation=45, ha="right")
        else:
            ax3.set_xticks(range(len(bar_labels)))
            ax3.set_xticklabels(bar_labels, fontsize=6, rotation=45, ha="right")
        ax3.set_ylabel("Mean cosine similarity")
        ax3.set_ylim(0, 1.12)
        ax3.legend(loc="upper right", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No redundancy data", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Head Redundancy per Layer", fontweight="bold", pad=20)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved model internals plot: {output_path}")


def run_model_internals_report(policy, dataset, args):
    """
    Orchestrator for model internals reporting.

    Step 1: Spectral alpha via WeightWatcher (no data needed).
    Step 2: Sample frames, run forward passes, compute entropy + redundancy.
    Step 3: Print terminal report and save plot.
    """
    device = torch.device(args.device)

    thresholds = {
        "entropy_warn": args.entropy_warn,
        "entropy_critical": args.entropy_critical,
        "entropy_low": args.entropy_low,
        "redundancy_warn": args.redundancy_warn,
        "redundancy_critical": args.redundancy_critical,
    }

    # ------------------------------------------------------------------
    # Step 1: WeightWatcher spectral analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("MODEL INTERNALS REPORT")
    print(f"{'=' * 70}")
    print("\n[1/3] Running WeightWatcher spectral analysis...")
    ww_results = compute_weightwatcher_alpha(policy)

    # ------------------------------------------------------------------
    # Step 2: Sample frames, capture per-head attention, compute metrics
    # ------------------------------------------------------------------
    print(f"\n[2/3] Capturing attention maps over {args.internals_frames} frames...")

    # Find vision encoder and set up hooks
    vision_encoder = find_vision_encoder(policy)
    if vision_encoder is None:
        print("  ERROR: Could not find vision encoder. Skipping attention metrics.")
        entropy_results = {}
        redundancy_results = {}
    else:
        # Force eager attention on SigLIP
        for mod in vision_encoder.modules():
            if getattr(mod, "config", None) is not None and hasattr(mod.config, "_attn_implementation"):
                mod.config._attn_implementation = "eager"

        # Force per-head weights (disable averaging) on SigLIP MHA
        mha_modules = []
        for mod in vision_encoder.modules():
            if isinstance(mod, torch.nn.MultiheadAttention):
                mha_modules.append((mod, getattr(mod, "average_attn_weights", True)))
                mod.average_attn_weights = False

        # Set up capture objects
        sigclip_capture = SigLIPAttentionCapture()
        sigclip_capture.register_hooks(vision_encoder)

        vlm_with_expert = policy.model.vlm_with_expert

        # SmolVLA bypasses self_attn.forward() — all attention flows through
        # eager_attention_forward(). DecoderAttentionCapture patches that
        # method to capture both self-attention (prefill) and cross-attention
        # (expert → VLM prefix).
        decoder_capture = DecoderAttentionCapture()
        print("  Setting up decoder attention capture (eager_attention_forward)...")
        decoder_capture.register(vlm_with_expert)

        # Resolve image key
        image_key = args.image_key
        if image_key is None:
            image_keys = find_image_keys(dataset)
            if image_keys:
                image_key = image_keys[0]
            else:
                print("  ERROR: No image keys found in dataset.")
                sigclip_capture.clear()
                decoder_capture.clear()
                for mod, orig in mha_modules:
                    mod.average_attn_weights = orig
                entropy_results = {}
                redundancy_results = {}
                image_key = None

        if image_key is not None:
            # Sample frames
            frame_pairs = get_episode_frames(
                dataset, args.episode, args.internals_frames, image_key,
            )

            # Determine SigLIP patch grid for entropy normalisation
            patch_size = getattr(vision_encoder, "patch_size", None) or getattr(
                getattr(vision_encoder, "config", None), "patch_size", 14,
            )
            img_size = getattr(
                getattr(vision_encoder, "config", None), "image_size", 384,
            )
            n_patches_side = img_size // patch_size
            sigclip_num_patches = n_patches_side * n_patches_side

            # Component names for the three attention sources
            COMP_SIGLIP = "SigLIP Vision (12L, 12H)"
            COMP_DECODER_SA = "VLM+Expert Joint Self-Attn (16L, 15H)"
            COMP_EXPERT_XA = "Expert-to-VLM Cross-Attn (16L, 8H)"

            all_entropy = {COMP_SIGLIP: [], COMP_DECODER_SA: [], COMP_EXPERT_XA: []}
            all_redundancy = {COMP_SIGLIP: [], COMP_DECODER_SA: [], COMP_EXPERT_XA: []}

            policy.eval()

            for i, (frame_idx, img_tensor) in enumerate(frame_pairs):
                sigclip_capture.reset_maps()
                decoder_capture.reset_maps()
                sample = dataset[frame_idx]

                with torch.no_grad():
                    # Use full policy forward so we capture all layers
                    policy.reset()
                    batch, _ = build_policy_batch_from_sample(
                        sample, policy, device, batch_size=1,
                        image_key_for_grad=None, dataset=dataset,
                        task_override=getattr(args, 'task', None),
                    )
                    try:
                        policy.select_action(batch)
                    except Exception as e:
                        print(f"    Frame {i} forward pass error: {e}")
                        continue

                # --- SigLIP ---
                counts = []
                siglip_attns = sigclip_capture.get_all_layer_attentions()
                if siglip_attns:
                    frame_ent = compute_attention_entropy(siglip_attns, sigclip_num_patches)
                    frame_red = compute_head_redundancy(siglip_attns)
                    all_entropy[COMP_SIGLIP].append(frame_ent)
                    all_redundancy[COMP_SIGLIP].append(frame_red)
                    counts.append(f"SigLIP={len(siglip_attns)}")
                else:
                    counts.append("SigLIP=0")

                # --- Decoder Self-Attention (prefill: VLM+Expert concatenated) ---
                sa_attns = decoder_capture.get_self_attn_layers()
                if sa_attns:
                    sa_num_tokens = sa_attns[0][1].shape[-1]
                    frame_ent = compute_attention_entropy(sa_attns, sa_num_tokens)
                    frame_red = compute_head_redundancy(sa_attns)
                    all_entropy[COMP_DECODER_SA].append(frame_ent)
                    all_redundancy[COMP_DECODER_SA].append(frame_red)
                    counts.append(f"Decoder SA={len(sa_attns)}")
                else:
                    counts.append("Decoder SA=0")

                # --- Expert Cross-Attention ---
                xa_attns_raw = decoder_capture.get_cross_attn_layers()
                if xa_attns_raw:
                    # Remap sequential indices → actual layer indices.
                    # During generation, each autoregressive step runs through
                    # all decoder layers, so call_idx % num_layers = layer.
                    num_dec_layers = vlm_with_expert.num_vlm_layers
                    xa_attns = [(idx % num_dec_layers, probs) for idx, probs in xa_attns_raw]
                    xa_num_tokens = xa_attns[0][1].shape[-1]
                    frame_ent = compute_attention_entropy(xa_attns, xa_num_tokens)
                    frame_red = compute_head_redundancy(xa_attns)
                    all_entropy[COMP_EXPERT_XA].append(frame_ent)
                    all_redundancy[COMP_EXPERT_XA].append(frame_red)
                    counts.append(f"Expert XA={len(xa_attns_raw)}")
                else:
                    counts.append("Expert XA=0")

                print(f"    Frame {i}: {', '.join(counts)}")

            # Cleanup
            sigclip_capture.clear()
            decoder_capture.clear()

            # Restore MHA averaging setting
            for mod, orig in mha_modules:
                mod.average_attn_weights = orig

            # Average metrics across frames, per component
            entropy_results = {}
            for comp_name, frame_list in all_entropy.items():
                if not frame_list:
                    continue
                ent_accum = defaultdict(lambda: defaultdict(list))
                for frame_ent in frame_list:
                    for e in frame_ent:
                        ent_accum[(e["layer"], e["head"])]["entropy_ratio"].append(e["entropy_ratio"])
                        ent_accum[(e["layer"], e["head"])]["entropy"].append(e["entropy"])

                comp_results = []
                for (layer, head), vals in sorted(ent_accum.items()):
                    comp_results.append({
                        "layer": layer,
                        "head": head,
                        "entropy": sum(vals["entropy"]) / len(vals["entropy"]),
                        "entropy_ratio": sum(vals["entropy_ratio"]) / len(vals["entropy_ratio"]),
                    })
                if comp_results:
                    entropy_results[comp_name] = comp_results

            redundancy_results = {}
            for comp_name, frame_list in all_redundancy.items():
                if not frame_list:
                    continue
                red_accum = defaultdict(lambda: {"mean": [], "max": []})
                for frame_red in frame_list:
                    for r in frame_red:
                        red_accum[r["layer"]]["mean"].append(r["mean_redundancy"])
                        red_accum[r["layer"]]["max"].append(r["max_redundancy"])

                comp_results = []
                for layer in sorted(red_accum.keys()):
                    vals = red_accum[layer]
                    comp_results.append({
                        "layer": layer,
                        "mean_redundancy": sum(vals["mean"]) / len(vals["mean"]),
                        "max_redundancy": sum(vals["max"]) / len(vals["max"]),
                    })
                if comp_results:
                    redundancy_results[comp_name] = comp_results

    # ------------------------------------------------------------------
    # Step 3: Print report and save plot
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating report...")
    print_model_internals_report(ww_results, entropy_results, redundancy_results, thresholds)

    os.makedirs(args.output_dir, exist_ok=True)

    md_path = os.path.join(args.output_dir, "model_internals_report.md")
    md_text = generate_model_internals_markdown(ww_results, entropy_results, redundancy_results, thresholds)
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"  Saved markdown report: {md_path}")

    plot_path = os.path.join(args.output_dir, "model_internals_report.png")
    plot_model_internals_report(ww_results, entropy_results, redundancy_results, plot_path)

    print(f"Done! Reports saved to {args.output_dir}/")
    return {
        "weightwatcher": ww_results,
        "entropy": entropy_results,
        "redundancy": redundancy_results,
        "markdown_path": md_path,
        "plot_path": plot_path,
    }

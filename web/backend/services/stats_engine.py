"""Compute numerical statistics from visualization data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data_loader


def compute_heatmap_stats(heatmaps: list[list[list[float]]]) -> dict:
    """Compute per-frame and aggregate stats for a set of heatmaps.

    Works for self_attention, cross_attention, saliency, gradcam_siglip,
    gradcam_connector.
    """
    per_frame: list[dict] = []
    arrays: list[np.ndarray] = []

    for i, hm in enumerate(heatmaps):
        arr = np.array(hm, dtype=np.float32)
        arrays.append(arr)
        flat = arr.flatten()
        max_val = float(flat.max()) if flat.size > 0 else 0.0
        mean_val = float(flat.mean()) if flat.size > 0 else 0.0

        # Shannon entropy ratio (0 = peaked, 1 = uniform)
        entropy_ratio = _entropy_ratio(flat)

        # Peak location as percentage of image dimensions
        if arr.ndim == 2 and arr.size > 0:
            peak_idx = int(np.argmax(arr))
            peak_row, peak_col = divmod(peak_idx, arr.shape[1])
            peak_y_pct = round(100.0 * peak_row / max(arr.shape[0] - 1, 1), 1)
            peak_x_pct = round(100.0 * peak_col / max(arr.shape[1] - 1, 1), 1)
        else:
            peak_y_pct = peak_x_pct = 0.0

        # Coverage: fraction of pixels above 30% of max
        threshold = 0.3 * max_val if max_val > 0 else 0
        coverage_pct = round(100.0 * float((flat > threshold).mean()), 1) if flat.size > 0 else 0.0

        # Spatial centroid (weighted average position)
        centroid_y, centroid_x = _spatial_centroid(arr)

        # Gini coefficient
        gini = _gini(flat)

        per_frame.append({
            "frame": i,
            "entropy_ratio": round(entropy_ratio, 4),
            "peak_location": {"x_pct": peak_x_pct, "y_pct": peak_y_pct},
            "coverage_pct": coverage_pct,
            "centroid": {"x_pct": round(centroid_x, 1), "y_pct": round(centroid_y, 1)},
            "gini": round(gini, 4),
            "max_intensity": round(max_val, 4),
            "mean_intensity": round(mean_val, 4),
        })

    # Aggregate stats
    aggregate: dict = {}
    if len(arrays) >= 2:
        # Temporal stability: mean cosine similarity between consecutive frames
        cos_sims = []
        for j in range(len(arrays) - 1):
            a = arrays[j].flatten()
            b = arrays[j + 1].flatten()
            if a.shape == b.shape:
                sim = _cosine_sim(a, b)
                cos_sims.append(sim)
        aggregate["temporal_stability"] = round(float(np.mean(cos_sims)), 4) if cos_sims else None

        # Centroid drift: total Euclidean distance of centroid across frames
        centroids = [(f["centroid"]["x_pct"], f["centroid"]["y_pct"]) for f in per_frame]
        total_drift = sum(
            np.sqrt((centroids[j + 1][0] - centroids[j][0]) ** 2 +
                     (centroids[j + 1][1] - centroids[j][1]) ** 2)
            for j in range(len(centroids) - 1)
        )
        aggregate["centroid_drift_total"] = round(float(total_drift), 2)

    if per_frame:
        aggregate["mean_entropy_ratio"] = round(float(np.mean([f["entropy_ratio"] for f in per_frame])), 4)
        aggregate["mean_coverage_pct"] = round(float(np.mean([f["coverage_pct"] for f in per_frame])), 1)
        aggregate["mean_gini"] = round(float(np.mean([f["gini"] for f in per_frame])), 4)

    return {"per_frame": per_frame, "aggregate": aggregate}


def compute_per_head_stats(heads: list, entropies: list | None) -> dict:
    """Stats for per-head attention data."""
    if not heads:
        return {}

    head_arrays = [np.array(h, dtype=np.float32).flatten() for h in heads]
    num_heads = len(head_arrays)

    # Inter-head cosine similarity matrix
    sim_matrix = np.zeros((num_heads, num_heads))
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            sim = _cosine_sim(head_arrays[i], head_arrays[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim
    mean_inter_head_sim = float(sim_matrix[np.triu_indices(num_heads, k=1)].mean()) if num_heads > 1 else 0.0

    result: dict = {
        "num_heads": num_heads,
        "mean_inter_head_similarity": round(mean_inter_head_sim, 4),
    }

    if entropies:
        ent_arr = np.array(entropies, dtype=np.float32)
        max_ent = float(ent_arr.max())
        min_ent = float(ent_arr.min())
        # Dead heads: entropy > 95% of max (near-uniform)
        dead_threshold = 0.95 * max_ent if max_ent > 0 else float("inf")
        dead_heads = [int(i) for i, e in enumerate(entropies) if e > dead_threshold]
        # Specialized heads: entropy < 20% of max (highly focused)
        spec_threshold = 0.2 * max_ent if max_ent > 0 else 0
        specialized_heads = [int(i) for i, e in enumerate(entropies) if e < spec_threshold]

        result["dead_heads"] = dead_heads
        result["specialized_heads"] = specialized_heads
        result["entropy_range"] = {"min": round(min_ent, 4), "max": round(max_ent, 4)}

    return result


def compute_vision_vs_state_stats(chart_data: list[dict]) -> dict:
    """Stats for vision vs state balance chart data."""
    if not chart_data:
        return {}

    vision_shares = []
    for point in chart_data:
        v = point.get("vision_norm", point.get("vision", 0))
        s = point.get("state_norm", point.get("state", 0))
        total = v + s
        vision_shares.append(v / total if total > 0 else 0.5)

    mean_vision_share = float(np.mean(vision_shares))

    # Linear regression slope for trend
    if len(vision_shares) >= 2:
        x = np.arange(len(vision_shares), dtype=np.float32)
        coeffs = np.polyfit(x, vision_shares, 1)
        slope = float(coeffs[0])
    else:
        slope = 0.0

    # Max imbalance ratio
    max_imbalance = max(
        max(s, 1e-8) / max(1 - s, 1e-8) if s > 0.5 else max(1 - s, 1e-8) / max(s, 1e-8)
        for s in vision_shares
    ) if vision_shares else 1.0

    return {
        "mean_vision_share": round(mean_vision_share, 4),
        "trend_slope": round(slope, 6),
        "trend_direction": "increasing_vision" if slope > 0.005 else "decreasing_vision" if slope < -0.005 else "stable",
        "max_imbalance_ratio": round(max_imbalance, 2),
        "num_frames": len(chart_data),
    }


def compute_per_action_dim_stats(dim_data: dict) -> dict:
    """Stats for per-action-dimension GradCAM data."""
    if not dim_data:
        return {}

    # dim_data is {frame_idx: {dim_idx: [[float]]}}
    # Aggregate across frames: compute mean attribution strength per dim
    dim_strengths: dict[int, list[float]] = {}
    all_dim_arrays: dict[int, list[np.ndarray]] = {}

    for frame_idx, dims in dim_data.items():
        if not isinstance(dims, dict):
            continue
        for dim_idx, hm in dims.items():
            dim_idx = int(dim_idx)
            arr = np.array(hm, dtype=np.float32)
            dim_strengths.setdefault(dim_idx, []).append(float(arr.mean()))
            all_dim_arrays.setdefault(dim_idx, []).append(arr)

    # Ranking by mean attribution strength
    rankings = []
    for dim_idx, strengths in sorted(dim_strengths.items()):
        rankings.append({
            "dim": dim_idx,
            "mean_strength": round(float(np.mean(strengths)), 6),
        })
    rankings.sort(key=lambda x: x["mean_strength"], reverse=True)

    # Spatial overlap (IoU of top-20% regions) between dim pairs for first frame
    first_frame = min(dim_data.keys()) if dim_data else None
    overlap_pairs: list[dict] = []
    if first_frame is not None:
        dims_first = dim_data[first_frame]
        if isinstance(dims_first, dict):
            dim_keys = sorted(int(k) for k in dims_first.keys())
            for i_idx, d1 in enumerate(dim_keys):
                for d2 in dim_keys[i_idx + 1:]:
                    a = np.array(dims_first[d1], dtype=np.float32)
                    b = np.array(dims_first[d2], dtype=np.float32)
                    if a.shape == b.shape:
                        iou = _top_k_iou(a, b, top_frac=0.2)
                        overlap_pairs.append({
                            "dim_a": d1, "dim_b": d2,
                            "iou": round(iou, 4),
                        })

    return {
        "strength_ranking": rankings,
        "spatial_overlap": overlap_pairs[:10],  # Limit output size
    }


def compute_run_summary_stats(run_dir: Path, manifest: dict) -> dict:
    """Aggregate stats across all available viz types for a single run."""
    avail = manifest.get("available_visualizations", {})
    stats: dict = {}

    if avail.get("self_attention"):
        hm = data_loader.load_self_attention_heatmaps(run_dir)
        if hm:
            stats["self_attention"] = compute_heatmap_stats(hm)

    if avail.get("cross_attention"):
        hm = data_loader.load_cross_attention_heatmaps(run_dir)
        if hm:
            stats["cross_attention"] = compute_heatmap_stats(hm)

    if avail.get("saliency"):
        hm = data_loader.load_gradient_data(run_dir, "saliency")
        if hm:
            stats["saliency"] = compute_heatmap_stats(hm)

    if avail.get("gradcam_siglip"):
        hm = data_loader.load_gradient_data(run_dir, "gradcam_siglip")
        if hm:
            stats["gradcam_siglip"] = compute_heatmap_stats(hm)

    if avail.get("gradcam_connector"):
        hm = data_loader.load_gradient_data(run_dir, "gradcam_connector")
        if hm:
            stats["gradcam_connector"] = compute_heatmap_stats(hm)

    if avail.get("per_head"):
        ph = data_loader.load_per_head_data(run_dir)
        if ph:
            stats["per_head"] = compute_per_head_stats(
                ph.get("heads", []), ph.get("entropies"))

    if avail.get("vision_vs_state"):
        vs = data_loader.load_vision_vs_state(run_dir)
        if vs:
            stats["vision_vs_state"] = compute_vision_vs_state_stats(vs)

    if avail.get("per_action_dim"):
        dims = data_loader.load_per_action_dim(run_dir)
        if dims:
            stats["per_action_dim"] = compute_per_action_dim_stats(dims)

    return stats


def compute_cross_run_stats(run_dirs: list[Path],
                            manifests: list[dict]) -> dict:
    """Compute deltas between runs for shared viz types."""
    if len(run_dirs) < 2:
        return {}

    all_summaries = []
    for rd, mf in zip(run_dirs, manifests):
        all_summaries.append(compute_run_summary_stats(rd, mf))

    # Compare first two runs (primary comparison)
    a, b = all_summaries[0], all_summaries[1]
    deltas: dict = {}

    for vt in set(a.keys()) & set(b.keys()):
        agg_a = a[vt].get("aggregate", {})
        agg_b = b[vt].get("aggregate", {})
        delta: dict = {}
        for key in set(agg_a.keys()) | set(agg_b.keys()):
            va = agg_a.get(key)
            vb = agg_b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta[key] = {"run_0": round(va, 4), "run_1": round(vb, 4),
                              "delta": round(vb - va, 4)}
        if delta:
            deltas[vt] = delta

    return {"per_run_summaries": [s for s in all_summaries], "deltas": deltas}


def format_stats_for_prompt(stats: dict) -> str:
    """Convert stats dict to human-readable text for prompt injection."""
    if not stats:
        return "(No numerical statistics available for this visualization.)"

    lines: list[str] = []

    # Handle both per-viz stats and run summary stats
    per_frame = stats.get("per_frame")
    aggregate = stats.get("aggregate")

    if aggregate:
        lines.append("**Aggregate Statistics:**")
        for key, val in sorted(aggregate.items()):
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                lines.append(f"- {label}: {val}")
            elif val is not None:
                lines.append(f"- {label}: {val}")

    if per_frame:
        lines.append(f"\n**Per-Frame Statistics** ({len(per_frame)} frames):")
        for f in per_frame:
            fi = f.get("frame", "?")
            ent = f.get("entropy_ratio", "?")
            cov = f.get("coverage_pct", "?")
            gini = f.get("gini", "?")
            peak = f.get("peak_location", {})
            centroid = f.get("centroid", {})
            lines.append(
                f"  Frame {fi}: entropy={ent}, coverage={cov}%, gini={gini}, "
                f"peak=({peak.get('x_pct', '?')}%, {peak.get('y_pct', '?')}%), "
                f"centroid=({centroid.get('x_pct', '?')}%, {centroid.get('y_pct', '?')}%)"
            )

    # Handle nested viz-type stats (run summary format)
    for key, val in stats.items():
        if key in ("per_frame", "aggregate"):
            continue
        if isinstance(val, dict):
            lines.append(f"\n**{key.replace('_', ' ').title()} Stats:**")
            inner_agg = val.get("aggregate", val)
            for k, v in sorted(inner_agg.items()):
                if k == "per_frame":
                    continue
                label = k.replace("_", " ").title()
                if isinstance(v, (int, float)):
                    lines.append(f"- {label}: {v}")
                elif isinstance(v, dict):
                    lines.append(f"- {label}: {v}")
                elif isinstance(v, list) and len(v) <= 5:
                    lines.append(f"- {label}: {v}")

    return "\n".join(lines) if lines else "(No statistics computed.)"


# -- Internal helpers --

def _entropy_ratio(flat: np.ndarray) -> float:
    """Shannon entropy ratio (0=peaked, 1=uniform)."""
    if flat.size == 0:
        return 0.0
    p = flat - flat.min()
    total = p.sum()
    if total == 0:
        return 1.0
    p = p / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log2(p)))
    max_entropy = np.log2(flat.size)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _spatial_centroid(arr: np.ndarray) -> tuple[float, float]:
    """Weighted centroid as percentage of image dimensions."""
    if arr.ndim != 2 or arr.size == 0:
        return (50.0, 50.0)
    total = arr.sum()
    if total == 0:
        return (50.0, 50.0)
    rows, cols = np.indices(arr.shape)
    cy = float((rows * arr).sum() / total)
    cx = float((cols * arr).sum() / total)
    cy_pct = 100.0 * cy / max(arr.shape[0] - 1, 1)
    cx_pct = 100.0 * cx / max(arr.shape[1] - 1, 1)
    return (cy_pct, cx_pct)


def _gini(flat: np.ndarray) -> float:
    """Gini coefficient (0=uniform, 1=all mass in one element)."""
    if flat.size == 0:
        return 0.0
    sorted_vals = np.sort(flat.flatten())
    n = sorted_vals.size
    index = np.arange(1, n + 1)
    total = sorted_vals.sum()
    if total == 0:
        return 0.0
    return float((2 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _top_k_iou(a: np.ndarray, b: np.ndarray, top_frac: float = 0.2) -> float:
    """IoU of top-k% activated regions between two heatmaps."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    k = max(1, int(a_flat.size * top_frac))

    a_thresh = np.partition(a_flat, -k)[-k]
    b_thresh = np.partition(b_flat, -k)[-k]

    a_mask = a.flatten() >= a_thresh
    b_mask = b.flatten() >= b_thresh

    intersection = float(np.logical_and(a_mask, b_mask).sum())
    union = float(np.logical_or(a_mask, b_mask).sum())
    return intersection / union if union > 0 else 0.0

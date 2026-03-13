"""Diagnostic matrix builder and anomaly detection.

Aggregates per-signal, per-region attribution across frames and runs a
battery of anomaly detectors to surface potential model failure modes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .models import (
    Anomaly,
    ConnectorAnalysis,
    DatasetDiversityReport,
    DiagnosticMatrix,
    OcclusionMap,
    TemporalTrajectory,
)
from .regions import attribute_to_regions, foreground_ratio, spatial_prior_ratio


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def build_diagnostic_matrix(
    frames: list[np.ndarray],
    segmentation,
    attention_heatmaps: list[np.ndarray] | None = None,
    gradcam_siglip_heatmaps: list[np.ndarray] | None = None,
    cross_attention_heatmaps: list[np.ndarray] | None = None,
    saliency_heatmaps: list[np.ndarray] | None = None,
    gradcam_connector_heatmaps: list[np.ndarray] | None = None,
    per_action_dim_maps: dict[str, list[np.ndarray]] | None = None,
    vision_vs_state: list[dict] | None = None,
    positional_baseline: np.ndarray | None = None,
    language_diff: dict | None = None,
    # Extended diagnostic inputs
    temporal_trajectories: list[TemporalTrajectory] | None = None,
    occlusion_map: OcclusionMap | None = None,
    connector_analysis: ConnectorAnalysis | None = None,
    dataset_diversity: DatasetDiversityReport | None = None,
) -> DiagnosticMatrix:
    """Build a :class:`DiagnosticMatrix` from available signals.

    Parameters
    ----------
    frames:
        List of (H, W, 3) RGB frames.
    segmentation:
        :class:`SceneSegmentation` providing region masks.
    attention_heatmaps, gradcam_siglip_heatmaps, ...:
        Per-frame (H, W) heatmaps for each signal type, or ``None`` to skip.
    per_action_dim_maps:
        Optional mapping ``action_dim_name -> list_of_heatmaps``.
    vision_vs_state:
        Per-frame dicts with at least a ``"vision_share"`` key.
    positional_baseline:
        A single (H, W) heatmap representing positional bias.
    language_diff:
        Dict with language-sensitivity diagnostics; may contain
        ``"max_shift"`` used for the ``language_diff_max_shift`` scalar.

    Returns
    -------
    DiagnosticMatrix
    """
    # Map signal names to their heatmap lists
    signal_map: dict[str, list[np.ndarray]] = {}
    if attention_heatmaps is not None:
        signal_map["attention"] = attention_heatmaps
    if gradcam_siglip_heatmaps is not None:
        signal_map["gradcam_siglip"] = gradcam_siglip_heatmaps
    if cross_attention_heatmaps is not None:
        signal_map["cross_attention"] = cross_attention_heatmaps
    if saliency_heatmaps is not None:
        signal_map["saliency"] = saliency_heatmaps
    if gradcam_connector_heatmaps is not None:
        signal_map["gradcam_connector"] = gradcam_connector_heatmaps

    signal_types = list(signal_map.keys())
    regions = segmentation.region_names()
    n_frames = len(frames)

    # Per-frame attribution breakdown: one dict per frame
    per_frame: list[dict[str, dict[str, float]]] = []
    for frame_idx in range(n_frames):
        frame_data: dict[str, dict[str, float]] = {}
        for sig_name, heatmap_list in signal_map.items():
            if frame_idx < len(heatmap_list):
                frame_data[sig_name] = attribute_to_regions(
                    heatmap_list[frame_idx], segmentation, signal_name=sig_name,
                )
        per_frame.append(frame_data)

    # Average attribution across frames
    attribution_mass: dict[str, dict[str, float]] = {}
    for sig_name in signal_types:
        region_sums: dict[str, float] = {r: 0.0 for r in regions}
        count = 0
        for frame_data in per_frame:
            if sig_name in frame_data:
                for r in regions:
                    region_sums[r] += frame_data[sig_name].get(r, 0.0)
                count += 1
        if count > 0:
            attribution_mass[sig_name] = {r: region_sums[r] / count for r in regions}
        else:
            attribution_mass[sig_name] = {r: 0.0 for r in regions}

    # Per-action-dim breakdown
    per_action_dim: dict[str, dict[str, dict[str, float]]] | None = None
    if per_action_dim_maps is not None:
        per_action_dim = {}
        for dim_name, heatmap_list in per_action_dim_maps.items():
            # Average across frames for this action dim
            region_sums: dict[str, float] = {r: 0.0 for r in regions}
            count = 0
            for hm in heatmap_list:
                shares = attribute_to_regions(hm, segmentation, signal_name=dim_name)
                for r in regions:
                    region_sums[r] += shares.get(r, 0.0)
                count += 1
            if count > 0:
                per_action_dim[dim_name] = {
                    "attribution": {r: region_sums[r] / count for r in regions},
                }
            else:
                per_action_dim[dim_name] = {"attribution": {r: 0.0 for r in regions}}

    # Scalars
    scalars: dict = {}

    if vision_vs_state is not None and len(vision_vs_state) > 0:
        vision_shares = [
            vs["vision_share"] for vs in vision_vs_state
            if vs is not None and "vision_share" in vs
        ]
        if vision_shares:
            scalars["vision_share"] = float(np.mean(vision_shares))

    if positional_baseline is not None and attention_heatmaps is not None and len(attention_heatmaps) > 0:
        scalars["positional_baseline_ratio"] = spatial_prior_ratio(
            attention_heatmaps[0], positional_baseline,
        )

    if attention_heatmaps is not None and len(attention_heatmaps) > 0:
        scalars["foreground_ratio_attn"] = foreground_ratio(
            attention_heatmaps[0], segmentation,
        )

    if gradcam_siglip_heatmaps is not None and len(gradcam_siglip_heatmaps) > 0:
        scalars["foreground_ratio_gradcam"] = foreground_ratio(
            gradcam_siglip_heatmaps[0], segmentation,
        )

    if language_diff is not None and "max_shift" in language_diff:
        scalars["language_diff_max_shift"] = float(language_diff["max_shift"])

    # Cross-attention entropy
    if cross_attention_heatmaps is not None and len(cross_attention_heatmaps) > 0:
        entropies: list[float] = []
        for hm in cross_attention_heatmaps:
            flat = hm.flatten().astype(np.float64)
            total = flat.sum()
            if total > 0:
                p = flat / total
                p = p[p > 0]
                entropies.append(float(-np.sum(p * np.log2(p))))
        if entropies:
            scalars["cross_attention_entropy"] = float(np.mean(entropies))

    # Occlusion sensitivity scalars
    if occlusion_map is not None:
        scalars["occlusion_max_delta"] = occlusion_map.max_delta
        scalars["occlusion_mean_delta"] = occlusion_map.mean_delta

    return DiagnosticMatrix(
        signal_types=signal_types,
        regions=regions,
        attribution_mass=attribution_mass,
        per_frame=per_frame,
        per_action_dim=per_action_dim,
        scalars=scalars,
        temporal_trajectories=temporal_trajectories,
        occlusion=occlusion_map,
        connector_analysis=connector_analysis,
    )


# ---------------------------------------------------------------------------
# Anomaly detectors
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def detect_high_background_attribution(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Background attribution > 60% for any causal signal (gradcam_siglip, saliency)."""
    causal_signals = {"gradcam_siglip", "saliency"}
    worst_signal = None
    worst_bg = 0.0

    for sig in causal_signals:
        if sig not in matrix.attribution_mass:
            continue
        bg = matrix.attribution_mass[sig].get("background", 0.0)
        if bg > worst_bg:
            worst_bg = bg
            worst_signal = sig

    if worst_signal is None or worst_bg <= 0.60:
        return None

    severity = "critical" if worst_bg > 0.75 else "warning"
    return Anomaly(
        type="high_background_attribution",
        severity=severity,
        description=(
            f"Background receives {worst_bg:.1%} of {worst_signal} attribution, "
            f"suggesting the model relies heavily on non-object regions."
        ),
        evidence={"signal": worst_signal, "background_share": worst_bg},
    )


def detect_attention_gradcam_divergence(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Any region differs by > 0.3 between attention and gradcam_siglip."""
    if "attention" not in matrix.attribution_mass or "gradcam_siglip" not in matrix.attribution_mass:
        return None

    attn = matrix.attribution_mass["attention"]
    gc = matrix.attribution_mass["gradcam_siglip"]
    divergent_regions: dict[str, float] = {}

    for region in matrix.regions:
        diff = attn.get(region, 0.0) - gc.get(region, 0.0)
        if abs(diff) > 0.3:
            divergent_regions[region] = diff

    if not divergent_regions:
        return None

    return Anomaly(
        type="attention_gradcam_divergence",
        severity="warning",
        description=(
            f"Attention and GradCAM diverge by >0.3 for regions: "
            f"{', '.join(divergent_regions.keys())}. "
            f"Positive = looking but not using; negative = using without looking."
        ),
        evidence={"divergent_regions": divergent_regions},
    )


def detect_dead_state_pathway(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Vision share > 99.5%, indicating the state pathway may be dead."""
    vision_share = matrix.scalars.get("vision_share")
    if vision_share is None:
        return None
    # vision_share is a 0–1 fraction from gradient.py
    if vision_share <= 0.995:
        return None

    pct = vision_share * 100
    return Anomaly(
        type="dead_state_pathway",
        severity="warning",
        description=(
            f"Vision pathway accounts for {pct:.1f}% of action "
            f"prediction, suggesting the proprioceptive/state input is ignored."
        ),
        evidence={"vision_share": vision_share},
    )


def detect_low_object_attribution(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Non-background, non-gripper task objects with < 10% gradcam attribution."""
    if "gradcam_siglip" not in matrix.attribution_mass:
        return None

    gc = matrix.attribution_mass["gradcam_siglip"]
    skip_labels = {"background", "gripper"}
    low_objects: dict[str, float] = {}
    worst_share = 1.0

    for region in matrix.regions:
        if region.lower() in skip_labels:
            continue
        share = gc.get(region, 0.0)
        if share < 0.10:
            low_objects[region] = share
            worst_share = min(worst_share, share)

    if not low_objects:
        return None

    severity = "critical" if worst_share < 0.05 else "warning"
    return Anomaly(
        type="low_object_attribution",
        severity=severity,
        description=(
            f"Task objects have low GradCAM attribution: "
            f"{', '.join(f'{k} ({v:.1%})' for k, v in low_objects.items())}. "
            f"The model may not be attending to manipulation targets."
        ),
        evidence={"low_objects": low_objects},
    )


def detect_spatial_shortcut(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Positional baseline ratio > 0.6 AND low object attribution exists."""
    pos_ratio = matrix.scalars.get("positional_baseline_ratio")
    if pos_ratio is None or pos_ratio <= 0.6:
        return None

    # Check whether low object attribution is also present
    low_obj = detect_low_object_attribution(matrix, internals, dataset_diversity)
    if low_obj is None:
        return None

    return Anomaly(
        type="spatial_shortcut",
        severity="critical",
        description=(
            f"High positional baseline similarity ({pos_ratio:.2f}) combined with "
            f"low object attribution suggests the model may be using spatial "
            f"shortcuts rather than understanding scene content."
        ),
        evidence={
            "positional_baseline_ratio": pos_ratio,
            "low_object_evidence": low_obj.evidence,
        },
    )


def detect_language_insensitivity(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Language diff max shift < 0.05, indicating instruction-insensitivity."""
    max_shift = matrix.scalars.get("language_diff_max_shift")
    if max_shift is None:
        return None
    if max_shift >= 0.05:
        return None

    return Anomaly(
        type="language_insensitivity",
        severity="warning",
        description=(
            f"Changing the task instruction produces a maximum attribution "
            f"shift of only {max_shift:.4f}, suggesting the model may be "
            f"insensitive to language conditioning."
        ),
        evidence={"language_diff_max_shift": max_shift},
    )


def detect_unstable_gradcam(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """GradCAM region attribution varies widely across frames (std/mean > 1.0)."""
    if "gradcam_siglip" not in matrix.attribution_mass:
        return None
    if len(matrix.per_frame) < 2:
        return None

    # Collect per-frame values for each region
    unstable_regions: dict[str, float] = {}
    for region in matrix.regions:
        values = []
        for frame_data in matrix.per_frame:
            if "gradcam_siglip" in frame_data:
                values.append(frame_data["gradcam_siglip"].get(region, 0.0))
        if len(values) < 2:
            continue
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if mean > 1e-6 and std / mean > 1.0:
            unstable_regions[region] = std / mean

    if not unstable_regions:
        return None

    return Anomaly(
        type="unstable_gradcam",
        severity="info",
        description=(
            f"GradCAM attribution is unstable across frames for regions: "
            f"{', '.join(f'{k} (CV={v:.2f})' for k, v in unstable_regions.items())}."
        ),
        evidence={"unstable_regions": unstable_regions},
    )


# ---------------------------------------------------------------------------
# Extended detectors
# ---------------------------------------------------------------------------


def detect_low_dataset_diversity(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Flag memorisation risks from low dataset diversity."""
    if dataset_diversity is None:
        return None

    issues: list[str] = []
    evidence: dict = {}
    severity = "info"

    # Check object position variance
    low_position_objects: list[str] = []
    for obj_label, stats in dataset_diversity.object_position_stats.items():
        count = stats.get("count", 0)
        std_x = stats.get("std_x", 999)
        std_y = stats.get("std_y", 999)
        if count > 0 and std_x < 15 and std_y < 15:
            low_position_objects.append(obj_label)

    if low_position_objects:
        issues.append(
            f"Objects with low position variance (std < 15px): "
            f"{', '.join(low_position_objects)}. Memorisation risk."
        )
        evidence["low_position_objects"] = low_position_objects
        severity = "warning"

    # Check background diversity
    if dataset_diversity.background_diversity_score < 0.02:
        issues.append(
            f"Background diversity score is {dataset_diversity.background_diversity_score:.4f}, "
            f"indicating single-environment overfitting."
        )
        evidence["background_diversity_score"] = dataset_diversity.background_diversity_score
        severity = "warning"

    # Check task string diversity
    unique_count = dataset_diversity.task_string_diversity.get("unique_count", 0)
    if unique_count == 1:
        issues.append("Only 1 unique task string — no language grounding.")
        evidence["unique_task_strings"] = unique_count
        # Escalate to critical if combined with low position variance
        if low_position_objects:
            severity = "critical"
        else:
            severity = "warning"

    if not issues:
        return None

    return Anomaly(
        type="low_dataset_diversity",
        severity=severity,
        description=" ".join(issues),
        evidence=evidence,
    )


def detect_gripper_fixation(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Flag when causal signals attribute > 40% to gripper region."""
    causal_signals = {"gradcam_siglip", "saliency"}
    worst_signal = None
    worst_share = 0.0

    for sig in causal_signals:
        if sig not in matrix.attribution_mass:
            continue
        for region_key in ("robot gripper", "gripper"):
            share = matrix.attribution_mass[sig].get(region_key, 0.0)
            if share > worst_share:
                worst_share = share
                worst_signal = sig

    if worst_signal is None or worst_share <= 0.40:
        return None

    return Anomaly(
        type="gripper_fixation",
        severity="warning",
        description=(
            f"Gripper region receives {worst_share:.1%} of {worst_signal} "
            f"attribution, suggesting the model fixates on the robot gripper "
            f"rather than the manipulation target."
        ),
        evidence={"signal": worst_signal, "gripper_share": worst_share},
    )


def detect_cross_attention_diffuse(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Flag near-uniform cross-attention entropy (> 5.0 bits)."""
    entropy = matrix.scalars.get("cross_attention_entropy")
    if entropy is None:
        return None
    # log2(64) = 6.0 — entropy > 5.0 is near-uniform for ~64 vision tokens
    if entropy <= 5.0:
        return None

    return Anomaly(
        type="cross_attention_diffuse",
        severity="warning",
        description=(
            f"Cross-attention entropy is {entropy:.2f} bits (near-uniform "
            f"threshold: 5.0). The action expert may not be selectively "
            f"attending to relevant vision tokens."
        ),
        evidence={"cross_attention_entropy": entropy},
    )


def detect_action_attention_misalignment(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Flag when translation action dims attribute primarily to background."""
    if matrix.per_action_dim is None:
        return None

    # Look for translation-related dims (first two, or keys with x/y/dx/dy)
    translation_dims: list[str] = []
    all_dims = list(matrix.per_action_dim.keys())
    for dim_name in all_dims:
        lower = dim_name.lower()
        if any(k in lower for k in ("x", "y", "dx", "dy")):
            translation_dims.append(dim_name)

    # Fall back to first two dims if no named translation dims found
    if not translation_dims and len(all_dims) >= 2:
        translation_dims = all_dims[:2]

    if not translation_dims:
        return None

    bg_dims: list[str] = []
    for dim_name in translation_dims:
        dim_data = matrix.per_action_dim[dim_name]
        attrib = dim_data.get("attribution", {})
        if not attrib:
            continue
        top_region = max(attrib, key=lambda r: attrib[r])
        if top_region == "background":
            bg_dims.append(dim_name)

    if len(bg_dims) < len(translation_dims) or not bg_dims:
        return None

    return Anomaly(
        type="action_attention_misalignment",
        severity="warning",
        description=(
            f"Translation action dimensions ({', '.join(bg_dims)}) all attribute "
            f"primarily to 'background'. The model may not be grounding spatial "
            f"actions on task-relevant objects."
        ),
        evidence={"background_attributed_dims": bg_dims},
    )


def detect_temporal_attention_instability(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Flag unstable or poorly-tracking attention trajectories."""
    if matrix.temporal_trajectories is None:
        return None

    worst_smoothness = 0.0
    worst_correlation = 1.0
    evidence: dict = {}

    for traj in matrix.temporal_trajectories:
        if traj.smoothness > worst_smoothness:
            worst_smoothness = traj.smoothness
        if traj.object_tracking_correlation < worst_correlation:
            worst_correlation = traj.object_tracking_correlation

    bad_smoothness = worst_smoothness > 0.15
    bad_correlation = worst_correlation < 0.3

    if not bad_smoothness and not bad_correlation:
        return None

    evidence["worst_smoothness"] = worst_smoothness
    evidence["worst_object_tracking_correlation"] = worst_correlation

    if bad_smoothness and bad_correlation:
        severity = "critical"
        description = (
            f"Attention centroid is unstable (smoothness={worst_smoothness:.3f}, "
            f"threshold 0.15) and does not track objects "
            f"(correlation={worst_correlation:.3f}, threshold 0.3). "
            f"The model may be attending randomly rather than following scene dynamics."
        )
    elif bad_smoothness:
        severity = "warning"
        description = (
            f"Attention centroid jumps significantly between frames "
            f"(smoothness={worst_smoothness:.3f}, threshold 0.15), "
            f"suggesting temporally unstable visual attention."
        )
    else:
        severity = "warning"
        description = (
            f"Attention trajectory has low correlation with object motion "
            f"(correlation={worst_correlation:.3f}, threshold 0.3), "
            f"suggesting the model does not track manipulation targets."
        )

    return Anomaly(
        type="temporal_attention_instability",
        severity=severity,
        description=description,
        evidence=evidence,
    )


def detect_single_region_dependency(
    matrix: DiagnosticMatrix,
    internals: dict | None,
    dataset_diversity=None,
) -> Anomaly | None:
    """Flag when all action dims attribute to the same non-background region."""
    if matrix.per_action_dim is None:
        return None

    top_regions: list[str] = []
    for dim_name, dim_data in matrix.per_action_dim.items():
        attrib = dim_data.get("attribution", {})
        if not attrib:
            continue
        top_region = max(attrib, key=lambda r: attrib[r])
        top_regions.append(top_region)

    if len(top_regions) < 2:
        return None

    # Check if all dims have the same top region
    if len(set(top_regions)) != 1:
        return None

    common_region = top_regions[0]

    # If background, other detectors handle it
    if common_region == "background":
        return None

    return Anomaly(
        type="single_region_dependency",
        severity="info",
        description=(
            f"All {len(top_regions)} action dimensions attribute primarily to "
            f"'{common_region}'. The model may be reading a single scene region "
            f"for all action outputs rather than using spatially diverse cues."
        ),
        evidence={
            "common_region": common_region,
            "num_action_dims": len(top_regions),
        },
    )


# ---------------------------------------------------------------------------
# Aggregate detector
# ---------------------------------------------------------------------------

_ALL_DETECTORS: list[Callable] = [
    detect_high_background_attribution,
    detect_attention_gradcam_divergence,
    detect_dead_state_pathway,
    detect_low_object_attribution,
    detect_spatial_shortcut,
    detect_language_insensitivity,
    detect_unstable_gradcam,
    # Extended detectors
    detect_low_dataset_diversity,
    detect_gripper_fixation,
    detect_cross_attention_diffuse,
    detect_action_attention_misalignment,
    detect_temporal_attention_instability,
    detect_single_region_dependency,
]


def detect_anomalies(
    matrix: DiagnosticMatrix,
    internals: dict | None = None,
    dataset_diversity=None,
) -> list[Anomaly]:
    """Run all anomaly detectors and return results sorted by severity.

    Order: critical first, then warning, then info.
    """
    anomalies: list[Anomaly] = []
    for detector in _ALL_DETECTORS:
        result = detector(matrix, internals, dataset_diversity)
        if result is not None:
            anomalies.append(result)

    anomalies.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 99))
    return anomalies

"""Disambiguate spatial-prior learning from object-feature grounding."""

from __future__ import annotations

from .models import (
    CounterfactualResult,
    DatasetDiversityReport,
    DiagnosticEvidence,
    DiagnosticMatrix,
    QKProbeReport,
    SceneSegmentation,
    SemanticProbeReport,
    SpatialObjectDiagnosis,
)


def choose_target_object(
    scene: SceneSegmentation,
    preferred_labels: list[str] | None = None,
) -> str | None:
    """Pick the most task-relevant object label present in the segmented scene."""
    available = {
        obj.label
        for obj in scene.objects
        if "gripper" not in obj.label.lower()
    }
    if preferred_labels:
        for label in preferred_labels:
            if label in available and "gripper" not in label.lower():
                return label

    for obj in scene.objects:
        if "gripper" not in obj.label.lower():
            return obj.label
    return None


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _pick_counterfactual(
    cf_results: list[CounterfactualResult],
    test_type: str,
    *,
    target_object: str | None = None,
) -> CounterfactualResult | None:
    candidates = [r for r in cf_results if r.test_type == test_type]
    if not candidates:
        return None

    if target_object is not None:
        targeted = [
            r for r in candidates
            if (r.metrics or {}).get("target_object") == target_object
        ]
        if targeted:
            candidates = targeted

    return max(
        candidates,
        key=lambda r: (float(r.action_delta_l2), len(r.metrics or {})),
    )


def _sorted_evidence(items: list[DiagnosticEvidence]) -> list[DiagnosticEvidence]:
    return sorted(items, key=lambda item: item.score, reverse=True)


def summarize_spatial_object_diagnosis(diag: SpatialObjectDiagnosis | None) -> str:
    """Render the structured diagnosis as a short text summary."""
    if diag is None:
        return "Spatial-vs-object diagnosis unavailable."

    verdict_labels = {
        "spatial_prior": "primarily spatial-prior driven",
        "object_grounded": "primarily object-feature grounded",
        "mixed": "a mixed strategy that uses both object cues and spatial priors",
        "inconclusive": "inconclusive",
    }
    spatial_top = diag.spatial_evidence[0].summary if diag.spatial_evidence else "limited direct spatial-prior evidence"
    object_top = diag.object_evidence[0].summary if diag.object_evidence else "limited direct object-grounding evidence"
    return (
        f"Target object: {diag.target_object or 'N/A'}. "
        f"Verdict: {verdict_labels.get(diag.verdict, diag.verdict)} "
        f"(confidence {diag.confidence:.0%}, spatial score {diag.spatial_score:.2f}, "
        f"object score {diag.object_score:.2f}). "
        f"Strongest spatial-prior evidence: {spatial_top}. "
        f"Strongest object-grounding evidence: {object_top}."
    )


def build_spatial_object_diagnosis(
    matrix: DiagnosticMatrix,
    *,
    target_object: str | None,
    cf_results: list[CounterfactualResult],
    dataset_diversity: DatasetDiversityReport | None = None,
    semantic_probe: SemanticProbeReport | None = None,
    qk_probe: QKProbeReport | None = None,
) -> SpatialObjectDiagnosis:
    """Summarize whether the policy is reading a location or an object."""
    spatial_evidence: list[DiagnosticEvidence] = []
    object_evidence: list[DiagnosticEvidence] = []
    spatial_sum = 0.0
    spatial_weight = 0.0
    object_sum = 0.0
    object_weight = 0.0
    key_metrics: dict = {}

    def add_spatial(source: str, strength: float, weight: float, summary: str, details: dict | None = None):
        nonlocal spatial_sum, spatial_weight
        if strength <= 0.0:
            return
        spatial_sum += strength * weight
        spatial_weight += weight
        spatial_evidence.append(
            DiagnosticEvidence(
                source=source,
                score=strength,
                summary=summary,
                details=details or {},
            )
        )

    def add_object(source: str, strength: float, weight: float, summary: str, details: dict | None = None):
        nonlocal object_sum, object_weight
        if strength <= 0.0:
            return
        object_sum += strength * weight
        object_weight += weight
        object_evidence.append(
            DiagnosticEvidence(
                source=source,
                score=strength,
                summary=summary,
                details=details or {},
            )
        )

    pos_ratio = matrix.scalars.get("positional_baseline_ratio")
    if pos_ratio is not None:
        key_metrics["positional_baseline_ratio"] = pos_ratio
        strength = _scale(float(pos_ratio), 0.45, 0.80)
        add_spatial(
            "positional_baseline",
            strength,
            0.24,
            f"Attention resembles a content-free positional baseline (ratio={float(pos_ratio):.2f}).",
            {"positional_baseline_ratio": pos_ratio},
        )

    if target_object is not None:
        gradcam_share = matrix.attribution_mass.get("gradcam_siglip", {}).get(target_object)
        if gradcam_share is not None:
            key_metrics["target_gradcam_share"] = gradcam_share
            add_object(
                "gradcam_target_share",
                _scale(float(gradcam_share), 0.08, 0.28),
                0.18,
                f"GradCAM places causal mass on '{target_object}' (share={float(gradcam_share):.1%}).",
                {"target_object": target_object, "target_gradcam_share": gradcam_share},
            )
            add_spatial(
                "low_target_gradcam_share",
                _scale(0.12 - float(gradcam_share), 0.01, 0.10),
                0.16,
                f"GradCAM gives little causal mass to '{target_object}' (share={float(gradcam_share):.1%}).",
                {"target_object": target_object, "target_gradcam_share": gradcam_share},
            )

        attention_share = matrix.attribution_mass.get("attention", {}).get(target_object)
        if attention_share is not None:
            key_metrics["target_attention_share"] = attention_share
            add_object(
                "attention_target_share",
                _scale(float(attention_share), 0.10, 0.30),
                0.08,
                f"Self-attention also allocates mass to '{target_object}' (share={float(attention_share):.1%}).",
                {"target_object": target_object, "target_attention_share": attention_share},
            )

    if semantic_probe is not None and semantic_probe.primary_frame is not None:
        frame = semantic_probe.primary_frame
        if frame.target_semantic_peak is not None:
            key_metrics["target_semantic_peak"] = float(frame.target_semantic_peak)
        if frame.target_semantic_mean_on_causal_patches is not None:
            key_metrics["target_semantic_mean_on_causal_patches"] = float(
                frame.target_semantic_mean_on_causal_patches
            )
        if frame.target_margin_over_best_non_target is not None:
            margin = float(frame.target_margin_over_best_non_target)
            key_metrics["target_margin_over_best_non_target"] = margin
            add_object(
                "semantic_target_margin",
                _scale(margin, 0.02, 0.18),
                0.18,
                f"Causal patches look more like '{target_object}' than other detected objects "
                f"(margin={margin:.3f}).",
                {
                    "target_object": target_object,
                    "best_non_target_label": frame.best_non_target_label,
                    "target_margin_over_best_non_target": margin,
                },
            )
            add_spatial(
                "weak_semantic_target_margin",
                _scale(0.05 - margin, 0.01, 0.10),
                0.14,
                f"Causal patches are not semantically distinctive for '{target_object}' "
                f"(margin={margin:.3f}).",
                {
                    "target_object": target_object,
                    "best_non_target_label": frame.best_non_target_label,
                    "target_margin_over_best_non_target": margin,
                },
            )
        if frame.causal_semantic_alignment is not None:
            alignment = float(frame.causal_semantic_alignment)
            key_metrics["causal_semantic_alignment"] = alignment
            add_object(
                "causal_semantic_alignment",
                _scale(alignment, 0.15, 0.45),
                0.10,
                f"High-causal patches contain clear target semantics for '{target_object}' "
                f"(alignment={alignment:.3f}).",
                {"target_object": target_object, "causal_semantic_alignment": alignment},
            )
        if frame.background_semantic_gap is not None:
            bg_gap = float(frame.background_semantic_gap)
            key_metrics["background_semantic_gap"] = bg_gap
            add_object(
                "background_semantic_gap",
                _scale(bg_gap, 0.03, 0.18),
                0.10,
                f"Target semantics are stronger on high-causal patches than on the low-causal background "
                f"(gap={bg_gap:.3f}).",
                {"target_object": target_object, "background_semantic_gap": bg_gap},
            )
            add_spatial(
                "low_background_semantic_gap",
                _scale(0.04 - bg_gap, 0.01, 0.08),
                0.10,
                f"Target semantics are no stronger on causal patches than on the surrounding background "
                f"(gap={bg_gap:.3f}).",
                {"target_object": target_object, "background_semantic_gap": bg_gap},
            )

    for traj in matrix.temporal_trajectories or []:
        corr = float(traj.object_tracking_correlation)
        key_metrics["object_tracking_correlation"] = max(
            corr,
            float(key_metrics.get("object_tracking_correlation", -1.0)),
        )
        add_object(
            "temporal_tracking",
            _scale(corr, 0.25, 0.75),
            0.12,
            f"{traj.signal_type} follows detected object motion over time (corr={corr:.2f}).",
            {"signal_type": traj.signal_type, "object_tracking_correlation": corr},
        )
        add_spatial(
            "poor_temporal_tracking",
            _scale(0.18 - corr, 0.01, 0.18),
            0.08,
            f"{traj.signal_type} does not track object motion well (corr={corr:.2f}).",
            {"signal_type": traj.signal_type, "object_tracking_correlation": corr},
        )

    if dataset_diversity is not None and target_object is not None:
        stats = dataset_diversity.object_position_stats.get(target_object)
        if stats:
            sx = stats.get("std_x")
            sy = stats.get("std_y")
            if sx is not None and sy is not None:
                mean_std = (float(sx) + float(sy)) / 2.0
                key_metrics["target_position_std_px"] = mean_std
                add_spatial(
                    "dataset_position_variance",
                    _scale(25.0 - mean_std, 2.0, 20.0),
                    0.14,
                    f"'{target_object}' appears in a narrow positional band in the dataset (std≈{mean_std:.1f}px).",
                    {"target_object": target_object, "std_x": sx, "std_y": sy},
                )

    relocation = _pick_counterfactual(
        cf_results,
        "object_relocation",
        target_object=target_object,
    )
    if relocation is not None:
        metrics = relocation.metrics or {}
        follow_ratio = metrics.get("focus_follow_ratio")
        anchor_ratio = metrics.get("anchor_retention_ratio")
        semantic_follow_ratio = metrics.get("semantic_follow_ratio")
        semantic_anchor_ratio = metrics.get("semantic_anchor_ratio")
        if follow_ratio is not None:
            key_metrics["relocation_follow_ratio"] = float(follow_ratio)
        if anchor_ratio is not None:
            key_metrics["relocation_anchor_ratio"] = float(anchor_ratio)
        if semantic_follow_ratio is not None:
            key_metrics["relocation_semantic_follow_ratio"] = float(semantic_follow_ratio)
        if semantic_anchor_ratio is not None:
            key_metrics["relocation_semantic_anchor_ratio"] = float(semantic_anchor_ratio)

        if follow_ratio is not None and anchor_ratio is not None:
            add_object(
                "relocation_follow_probe",
                _scale(float(follow_ratio) - float(anchor_ratio), 0.05, 0.35),
                0.26,
                f"After relocation, GradCAM follows the moved object more than the old anchor (follow={float(follow_ratio):.2f}, anchor={float(anchor_ratio):.2f}).",
                metrics,
            )
            add_spatial(
                "relocation_anchor_probe",
                _scale(float(anchor_ratio) - float(follow_ratio), 0.05, 0.35),
                0.26,
                f"After relocation, GradCAM stays anchored to the old location more than the moved object (anchor={float(anchor_ratio):.2f}, follow={float(follow_ratio):.2f}).",
                metrics,
            )
        if semantic_follow_ratio is not None and semantic_anchor_ratio is not None:
            add_object(
                "semantic_relocation_follow_probe",
                _scale(float(semantic_follow_ratio) - float(semantic_anchor_ratio), 0.05, 0.35),
                0.20,
                f"After relocation, target semantics follow the moved object more than the old anchor "
                f"(follow={float(semantic_follow_ratio):.2f}, anchor={float(semantic_anchor_ratio):.2f}).",
                metrics,
            )
            add_spatial(
                "semantic_relocation_anchor_probe",
                _scale(float(semantic_anchor_ratio) - float(semantic_follow_ratio), 0.05, 0.35),
                0.20,
                f"After relocation, target semantics stay more anchored to the old location than to the moved object "
                f"(anchor={float(semantic_anchor_ratio):.2f}, follow={float(semantic_follow_ratio):.2f}).",
                metrics,
            )

    occlusion = _pick_counterfactual(
        cf_results,
        "occlusion_targeted",
        target_object=target_object,
    )
    if occlusion is not None:
        delta = float(occlusion.action_delta_l2)
        key_metrics["occlusion_action_delta"] = delta
        semantic_drop = occlusion.metrics.get("occlusion_target_semantic_drop") if occlusion.metrics else None
        add_object(
            "target_occlusion_response",
            _scale(delta, 0.02, 0.10),
            0.18,
            f"Occluding '{target_object}' changes the predicted action (delta={delta:.4f}).",
            {"target_object": target_object, "action_delta_l2": delta},
        )
        add_spatial(
            "target_occlusion_insensitivity",
            _scale(0.015 - delta, 0.002, 0.013),
            0.08,
            f"Occluding '{target_object}' barely changes the predicted action (delta={delta:.4f}).",
            {"target_object": target_object, "action_delta_l2": delta},
        )
        if semantic_drop is not None:
            semantic_drop = float(semantic_drop)
            key_metrics["occlusion_target_semantic_drop"] = semantic_drop
            add_object(
                "target_occlusion_semantic_drop",
                _scale(semantic_drop, 0.03, 0.20),
                0.12,
                f"Occluding '{target_object}' removes target-semantic evidence from the target region "
                f"(drop={semantic_drop:.3f}).",
                {"target_object": target_object, "occlusion_target_semantic_drop": semantic_drop},
            )
            add_spatial(
                "target_occlusion_semantic_insensitivity",
                _scale(0.02 - semantic_drop, 0.005, 0.02),
                0.08,
                f"Occluding '{target_object}' does not materially reduce target-semantic evidence "
                f"(drop={semantic_drop:.3f}).",
                {"target_object": target_object, "occlusion_target_semantic_drop": semantic_drop},
            )

    background = _pick_counterfactual(cf_results, "background_substitution")
    if background is not None:
        delta = float(background.action_delta_l2)
        key_metrics["background_action_delta"] = delta
        add_spatial(
            "background_substitution",
            _scale(delta, 0.02, 0.10),
            0.12,
            f"Replacing the background changes the action noticeably (delta={delta:.4f}).",
            {"action_delta_l2": delta},
        )

    recolor = _pick_counterfactual(
        cf_results,
        "object_recolor",
        target_object=target_object,
    )
    if recolor is not None:
        delta = float(recolor.action_delta_l2)
        key_metrics["recolor_action_delta"] = delta
        add_object(
            "object_recolor",
            _scale(delta, 0.015, 0.08),
            0.08,
            f"Changing '{target_object}' appearance changes the action (delta={delta:.4f}).",
            {"target_object": target_object, "action_delta_l2": delta},
        )

    if qk_probe is not None:
        key_metrics["qk_semantic_head_fraction"] = qk_probe.semantic_head_fraction
        key_metrics["qk_positional_head_fraction"] = qk_probe.positional_head_fraction
        if qk_probe.dominant_head_type == "semantic":
            add_object(
                "qk_semantic_heads",
                _scale(
                    qk_probe.semantic_head_fraction - qk_probe.positional_head_fraction,
                    0.05,
                    0.40,
                ),
                0.14,
                qk_probe.summary,
                {"dominant_head_type": qk_probe.dominant_head_type},
            )
        elif qk_probe.dominant_head_type == "positional":
            add_spatial(
                "qk_positional_heads",
                _scale(
                    qk_probe.positional_head_fraction - qk_probe.semantic_head_fraction,
                    0.05,
                    0.40,
                ),
                0.14,
                qk_probe.summary,
                {"dominant_head_type": qk_probe.dominant_head_type},
            )

    spatial_score = spatial_sum / spatial_weight if spatial_weight > 1e-8 else 0.0
    object_score = object_sum / object_weight if object_weight > 1e-8 else 0.0

    if target_object is None:
        verdict = "inconclusive"
        confidence = 0.0
        summary = (
            "No non-gripper task object was detected, so the diagnostic cannot "
            "reliably separate spatial shortcut learning from object grounding."
        )
    elif spatial_score >= 0.55 and object_score < 0.40:
        verdict = "spatial_prior"
        confidence = min(0.95, 0.45 + 0.35 * spatial_score + 0.25 * (spatial_score - object_score))
        summary = (
            f"The model appears more anchored to a memorized location than to '{target_object}' itself. "
            f"Spatial-prior evidence outweighs object-grounding evidence "
            f"({spatial_score:.2f} vs {object_score:.2f})."
        )
    elif object_score >= 0.55 and spatial_score < 0.40:
        verdict = "object_grounded"
        confidence = min(0.95, 0.45 + 0.35 * object_score + 0.25 * (object_score - spatial_score))
        summary = (
            f"The model appears to be using features of '{target_object}' rather than a memorized anchor point. "
            f"Object-grounding evidence outweighs spatial-prior evidence "
            f"({object_score:.2f} vs {spatial_score:.2f})."
        )
    elif spatial_score >= 0.40 and object_score >= 0.40:
        verdict = "mixed"
        confidence = min(0.90, 0.40 + 0.30 * ((spatial_score + object_score) / 2.0))
        summary = (
            f"The model appears to use a mixed strategy: it has some grounding on '{target_object}', "
            f"but it also retains a meaningful spatial prior for the usual location "
            f"({spatial_score:.2f} spatial vs {object_score:.2f} object)."
        )
    else:
        verdict = "inconclusive"
        confidence = min(0.70, 0.20 + 0.40 * max(spatial_score, object_score))
        summary = (
            f"The available evidence is not strong enough to cleanly separate spatial-prior use from "
            f"object grounding for '{target_object}'. The model may be weakly grounded, weakly shortcut-driven, "
                "or the available probes may be too noisy."
        )

    return SpatialObjectDiagnosis(
        target_object=target_object,
        verdict=verdict,
        confidence=confidence,
        spatial_score=spatial_score,
        object_score=object_score,
        summary=summary,
        key_metrics=key_metrics,
        spatial_evidence=_sorted_evidence(spatial_evidence),
        object_evidence=_sorted_evidence(object_evidence),
    )

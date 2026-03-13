from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# Scene-level models
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    """A single detected object in a scene."""

    label: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    mask: np.ndarray | None = None  # H, W binary mask


@dataclass
class SceneSegmentation:
    """Full scene segmentation: detected objects + background."""

    objects: list[DetectedObject]
    background_mask: np.ndarray  # H, W binary
    image_shape: tuple[int, int]

    def get_mask(self, label: str) -> np.ndarray | None:
        """Return the union of all masks for objects matching *label*, or None."""
        merged: np.ndarray | None = None
        for obj in self.objects:
            if obj.label == label and obj.mask is not None:
                if merged is None:
                    merged = obj.mask.copy()
                else:
                    merged |= obj.mask
        return merged

    def region_names(self) -> list[str]:
        """Unique object labels plus ``'background'``."""
        seen: set[str] = set()
        names: list[str] = []
        for obj in self.objects:
            if obj.label not in seen:
                seen.add(obj.label)
                names.append(obj.label)
        names.append("background")
        return names


# ---------------------------------------------------------------------------
# Dataset diversity
# ---------------------------------------------------------------------------

@dataclass
class DatasetDiversityReport:
    """Summary statistics about dataset diversity."""

    object_position_stats: dict[str, dict]  # per-object centroid mean/std/range
    background_diversity_score: float  # 0-1
    lighting_stats: dict  # mean_brightness, contrast_variance
    task_string_diversity: dict  # unique_count, embedding_spread
    num_episodes_sampled: int


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """A single detected anomaly."""

    type: str
    severity: str  # "critical" | "warning" | "info"
    description: str
    evidence: dict


# ---------------------------------------------------------------------------
# Diagnostic matrix
# ---------------------------------------------------------------------------

@dataclass
class TemporalTrajectory:
    """Attention centroid trajectory across episode frames."""

    frame_indices: list[int]
    centroids: list[tuple[float, float]]  # (cx, cy) per frame
    smoothness: float  # mean inter-frame centroid displacement (lower = smoother)
    object_tracking_correlation: float  # correlation with GT object centroid (-1 to 1)
    signal_type: str  # which signal was used (attention, gradcam_siglip, etc.)


@dataclass
class OcclusionMap:
    """Systematic occlusion sensitivity map."""

    sensitivity_map: np.ndarray  # (grid_h, grid_w) action L2 delta per patch
    patch_size: int
    stride: int
    max_delta: float
    mean_delta: float


@dataclass
class ConnectorAnalysis:
    """Pre/post connector attribution comparison."""

    pre_connector_shares: dict[str, float]  # region -> share (from 1024-patch space)
    post_connector_shares: dict[str, float]  # region -> share (from 64-token space)
    information_loss_per_region: dict[str, float]  # region -> |pre - post|
    total_information_loss: float  # sum of per-region losses


@dataclass
class DiagnosticMatrix:
    """Attribution matrix: signal types x regions, with optional per-frame and
    per-action-dim breakdowns."""

    signal_types: list[str]
    regions: list[str]
    attribution_mass: dict[str, dict[str, float]]  # signal -> region -> float
    per_frame: list[dict[str, dict[str, float]]]
    per_action_dim: dict[str, dict[str, dict[str, float]]] | None = None
    scalars: dict = field(default_factory=dict)
    temporal_trajectories: list[TemporalTrajectory] | None = None
    occlusion: OcclusionMap | None = None
    connector_analysis: ConnectorAnalysis | None = None

    # -- analysis ----------------------------------------------------------

    def anomalies(self) -> list[Anomaly]:
        """Placeholder anomaly detection – returns an empty list for now."""
        return []

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the matrix to a plain dict."""
        d = {
            "signal_types": self.signal_types,
            "regions": self.regions,
            "attribution_mass": self.attribution_mass,
            "per_frame": self.per_frame,
            "per_action_dim": self.per_action_dim,
            "scalars": self.scalars,
        }
        if self.temporal_trajectories:
            d["temporal_trajectories"] = [
                {"frame_indices": t.frame_indices, "centroids": t.centroids,
                 "smoothness": t.smoothness,
                 "object_tracking_correlation": t.object_tracking_correlation,
                 "signal_type": t.signal_type}
                for t in self.temporal_trajectories
            ]
        if self.occlusion is not None:
            d["occlusion"] = {
                "sensitivity_map": self.occlusion.sensitivity_map.tolist(),
                "patch_size": self.occlusion.patch_size,
                "stride": self.occlusion.stride,
                "max_delta": self.occlusion.max_delta,
                "mean_delta": self.occlusion.mean_delta,
            }
        if self.connector_analysis is not None:
            d["connector_analysis"] = asdict(self.connector_analysis)
        return d

    def to_markdown(self) -> str:
        """Render the attribution matrix as a Markdown table.

        Signal types are rows; regions are columns.  Each cell shows the
        fraction of total attribution mass that falls in that region for that
        signal (values sum to ~1.0 across each row).
        """
        _SIG_LABELS = {
            "attention": "Self-attention",
            "gradcam_siglip": "GradCAM (SigLIP)",
            "gradcam_connector": "GradCAM (Connector)",
            "saliency": "Saliency",
            "cross_attention": "Cross-attention",
        }
        lines: list[str] = []

        lines.append("Each cell is the fraction of total attribution mass in that region (rows sum to 1.0).")
        lines.append("")
        header = "| Signal \\ Region | " + " | ".join(self.regions) + " |"
        sep = "|---|" + "---|" * len(self.regions)
        lines.append(header)
        lines.append(sep)
        for sig in self.signal_types:
            row_data = self.attribution_mass.get(sig, {})
            cells = [f"{row_data.get(r, 0.0):.4f}" for r in self.regions]
            label = _SIG_LABELS.get(sig, sig)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        if self.scalars:
            _SCALAR_HELP = {
                "vision_share": ("Vision share",
                                 "Fraction of gradient norm from vision vs proprioceptive state "
                                 "(1.0 = vision-only, 0.5 = balanced). Values near 1.0 suggest "
                                 "proprioceptive state is ignored."),
                "positional_baseline_ratio": ("Positional baseline ratio",
                                              "Cosine similarity between the model's attention and "
                                              "a blank-image positional baseline (0.0 = no spatial "
                                              "shortcut, 1.0 = pure memorized coordinates). "
                                              "Values > 0.7 indicate spatial shortcut learning."),
                "foreground_ratio_attn": ("Foreground ratio (attention)",
                                          "Fraction of self-attention on foreground objects vs "
                                          "background (1.0 = all foreground, 0.0 = all background). "
                                          "Values < 0.3 indicate weak object grounding."),
                "foreground_ratio_gradcam": ("Foreground ratio (GradCAM)",
                                             "Fraction of GradCAM attribution on foreground objects "
                                             "vs background. Values < 0.3 indicate the model's "
                                             "causal signal is dominated by background."),
            }
            lines.append("")
            lines.append("### Scalar Metrics")
            lines.append("")
            lines.append("| Metric | Value | Interpretation |")
            lines.append("|---|---|---|")
            for key, val in self.scalars.items():
                label, desc = _SCALAR_HELP.get(key, (key, ""))
                lines.append(f"| {label} | {float(val):.4f} | {desc} |")

        if self.temporal_trajectories:
            lines.append("")
            lines.append("### Temporal Trajectories")
            lines.append("")
            for t in self.temporal_trajectories:
                lines.append(
                    f"- {t.signal_type}: smoothness={t.smoothness:.4f}, "
                    f"object tracking correlation={t.object_tracking_correlation:.4f}, "
                    f"{len(t.frame_indices)} frames"
                )

        if self.occlusion is not None:
            lines.append("")
            lines.append("### Occlusion Sensitivity")
            lines.append("")
            lines.append(f"- Patch size: {self.occlusion.patch_size}px, stride: {self.occlusion.stride}px")
            lines.append(f"- Max delta: {self.occlusion.max_delta:.4f}, mean delta: {self.occlusion.mean_delta:.4f}")

        if self.connector_analysis is not None:
            ca = self.connector_analysis
            lines.append("")
            lines.append("### Connector Bottleneck Analysis")
            lines.append("")
            lines.append("The pixel-shuffle connector compresses 1024 SigLIP patches into 64 VLM tokens (16x compression).")
            lines.append("This table shows how much attribution each region retains through the connector.")
            lines.append("")
            lines.append("| Region | Pre-connector | Post-connector | Loss | Loss % |")
            lines.append("|---|---|---|---|---|")
            for r, loss in ca.information_loss_per_region.items():
                pre = ca.pre_connector_shares.get(r, 0)
                post = ca.post_connector_shares.get(r, 0)
                loss_pct = (loss / pre * 100) if pre > 1e-6 else 0.0
                lines.append(f"| {r} | {pre:.4f} | {post:.4f} | {loss:.4f} | {loss_pct:.1f}% |")
            lines.append(f"| **Total** | | | **{ca.total_information_loss:.4f}** | |")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A testable hypothesis generated from anomalies."""

    id: str
    description: str
    confidence: float  # 0-1
    supporting_anomalies: list[str]
    test_type: str
    test_params: dict
    expected_if_true: str
    expected_if_false: str
    confirms_on_change: bool = True  # False = hypothesis confirmed when action does NOT change


@dataclass
class CounterfactualResult:
    """Result of a counterfactual intervention test."""

    hypothesis_id: str
    test_type: str
    action_delta_l2: float
    action_delta_per_dim: list[float]
    gradcam_shift: float
    attribution_shift_per_region: dict[str, float]
    confirmed: bool
    metrics: dict = field(default_factory=dict)
    visual_comparison: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Evidence & findings
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticEvidence:
    """A weighted piece of evidence for a diagnosis."""

    source: str
    score: float  # 0-1 normalized contribution
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class SpatialObjectDiagnosis:
    """Structured diagnosis separating spatial priors from object grounding."""

    target_object: str | None
    verdict: str  # "spatial_prior" | "object_grounded" | "mixed" | "inconclusive"
    confidence: float  # 0-1
    spatial_score: float  # 0-1
    object_score: float  # 0-1
    summary: str
    key_metrics: dict = field(default_factory=dict)
    spatial_evidence: list[DiagnosticEvidence] = field(default_factory=list)
    object_evidence: list[DiagnosticEvidence] = field(default_factory=list)


@dataclass
class SemanticFrameSummary:
    """Semantic-probe summary for a single image/frame."""

    frame_id: str
    target_object: str | None
    candidate_labels: list[str] = field(default_factory=list)
    best_non_target_label: str | None = None
    target_semantic_peak: float | None = None
    target_semantic_mean_on_causal_patches: float | None = None
    target_margin_over_best_non_target: float | None = None
    causal_semantic_alignment: float | None = None
    background_semantic_gap: float | None = None
    summary: str = ""


@dataclass
class SemanticCounterfactualSummary:
    """Semantic summary derived from a counterfactual probe."""

    test_type: str
    summary: str
    metrics: dict = field(default_factory=dict)


@dataclass
class SemanticProbeReport:
    """Top-level semantic object-vs-location probe report."""

    target_object: str | None
    summary: str
    primary_frame: SemanticFrameSummary | None = None
    frames: list[SemanticFrameSummary] = field(default_factory=list)
    counterfactuals: dict[str, SemanticCounterfactualSummary] = field(default_factory=dict)


@dataclass
class QKHeadSummary:
    """Per-head QK decomposition summary for the final SigLIP layer."""

    head_index: int
    head_type: str
    score: float
    semantic_map_correlation: float
    positional_baseline_correlation: float
    target_region_logit_mass: float
    background_logit_mass: float
    old_anchor_logit_mass: float | None = None
    moved_object_logit_mass: float | None = None


@dataclass
class QKProbeReport:
    """Compact summary of last-layer SigLIP QK behavior."""

    layer: str
    summary: str
    dominant_head_type: str
    semantic_head_fraction: float
    positional_head_fraction: float
    mixed_head_fraction: float
    top_heads: list[QKHeadSummary] = field(default_factory=list)


@dataclass
class EvidenceEntry:
    """A single piece of evidence collected during a diagnostic phase."""

    phase: str
    primitive_name: str
    data: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Finding:
    """A human-readable finding with observation, test, interpretation, and fix."""

    id: str
    severity: str  # "critical" | "warning" | "info"
    title: str
    observation: str
    test_description: str
    test_result: str
    interpretation: str
    fix: str
    expected_impact: str
    evidence_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def _numpy_serialiser(obj: object) -> object:
    """JSON-compatible conversion for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class DiagnosticReport:
    """Full diagnostic report produced at the end of a run."""

    metadata: dict
    scene: SceneSegmentation
    dataset_diversity: DatasetDiversityReport | None
    matrix: DiagnosticMatrix
    semantic_probe: SemanticProbeReport | None
    qk_probe: QKProbeReport | None
    spatial_object_diagnosis: SpatialObjectDiagnosis | None
    anomalies: list[Anomaly]
    hypotheses: list[Hypothesis]
    counterfactual_results: list[CounterfactualResult]
    findings: list[Finding]
    llm_synthesis: str = ""
    cf_skip_reason: str = ""  # "skipped", "no_model", "budget_exhausted", or ""

    # -- helpers -----------------------------------------------------------

    def _serialisable_scene(self) -> dict:
        """Convert *SceneSegmentation* to a JSON-safe dict."""
        objs = []
        for o in self.scene.objects:
            d: dict = {
                "label": o.label,
                "box": list(o.box),
                "score": o.score,
            }
            if o.mask is not None:
                d["mask_shape"] = list(o.mask.shape)
            objs.append(d)
        return {
            "objects": objs,
            "background_mask_shape": list(self.scene.background_mask.shape),
            "image_shape": list(self.scene.image_shape),
        }

    def _serialisable_counterfactuals(self) -> list[dict]:
        results = []
        for cr in self.counterfactual_results:
            d = {
                "hypothesis_id": cr.hypothesis_id,
                "test_type": cr.test_type,
                "action_delta_l2": cr.action_delta_l2,
                "action_delta_per_dim": cr.action_delta_per_dim,
                "gradcam_shift": cr.gradcam_shift,
                "attribution_shift_per_region": cr.attribution_shift_per_region,
                "confirmed": cr.confirmed,
                "metrics": cr.metrics,
            }
            if cr.visual_comparison is not None:
                d["visual_comparison_shape"] = list(cr.visual_comparison.shape)
            results.append(d)
        return results

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> str:
        """Serialise the full report to a JSON string.

        Numpy arrays are converted to lists; masks are represented by shape
        only to keep the output compact.
        """
        payload = {
            "metadata": self.metadata,
            "scene": self._serialisable_scene(),
            "dataset_diversity": asdict(self.dataset_diversity) if self.dataset_diversity else None,
            "matrix": self.matrix.to_dict(),
            "semantic_probe": asdict(self.semantic_probe) if self.semantic_probe else None,
            "qk_probe": asdict(self.qk_probe) if self.qk_probe else None,
            "spatial_object_diagnosis": (
                asdict(self.spatial_object_diagnosis)
                if self.spatial_object_diagnosis
                else None
            ),
            "anomalies": [asdict(a) for a in self.anomalies],
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "counterfactual_results": self._serialisable_counterfactuals(),
            "findings": [asdict(f) for f in self.findings],
            "llm_synthesis": self.llm_synthesis,
        }
        return json.dumps(payload, indent=2, default=_numpy_serialiser)

    # -- markdown ----------------------------------------------------------

    def to_markdown(self) -> str:
        """Generate a full Markdown report."""
        sections: list[str] = []

        # ── Title & Executive Summary ─────────────────────────────
        sections.append("# Diagnostic Report")
        sections.append("")

        # Quick severity tally at top
        n_crit = sum(1 for f in self.findings if f.severity == "critical")
        n_warn = sum(1 for f in self.findings if f.severity == "warning")
        n_info = sum(1 for f in self.findings if f.severity == "info")
        if self.findings:
            parts = []
            if n_crit:
                parts.append(f"**{n_crit} critical**")
            if n_warn:
                parts.append(f"**{n_warn} warning{'s' if n_warn != 1 else ''}**")
            if n_info:
                parts.append(f"{n_info} info")
            sections.append(f"> {', '.join(parts)} finding{'s' if len(self.findings) != 1 else ''} detected.")
            sections.append("")

        # ── Metadata ──────────────────────────────────────────────
        sections.append("## Run Configuration")
        sections.append("")
        _META_LABELS = {
            "task_string": "Task instruction",
            "episode_idx": "Episode index",
            "image_key": "Image key",
            "device": "Compute device",
            "post_hoc": "Post-hoc mode",
            "model_id": "Model",
            "dataset_id": "Dataset",
        }
        for k, v in self.metadata.items():
            label = _META_LABELS.get(k, k)
            sections.append(f"| {label} | {v} |")
        sections.append("")

        # ── Scene Segmentation ────────────────────────────────────
        sections.append("## Scene Segmentation")
        sections.append("")
        h, w = self.scene.image_shape
        sections.append(f"Image: {w} x {h} pixels")
        sections.append("")
        if self.scene.objects:
            sections.append("| Object | Bounding Box (x1, y1, x2, y2) | Detection Confidence |")
            sections.append("|---|---|---|")
            for obj in self.scene.objects:
                x1, y1, x2, y2 = obj.box
                sections.append(
                    f"| {obj.label} | ({x1}, {y1}) to ({x2}, {y2}) | {obj.score:.4f} |"
                )
            sections.append("")
            sections.append(
                "*Bounding box coordinates are in pixels, origin at top-left. "
                "(x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner. "
                "Detection confidence is a 0\u20131 score from OWL-ViT v2; higher means "
                "the detector is more certain the object is present.*"
            )
            sections.append("")

        # ── Dataset Diversity ─────────────────────────────────────
        if self.dataset_diversity:
            dd = self.dataset_diversity
            sections.append("## Dataset Diversity")
            sections.append("")
            sections.append(f"Sampled **{dd.num_episodes_sampled}** episodes from the dataset.")
            sections.append("")

            # Diversity metrics table
            bg_score = dd.background_diversity_score
            brightness = dd.lighting_stats.get("mean_brightness", 0)
            contrast = dd.lighting_stats.get("contrast_variance", 0)
            unique_tasks = dd.task_string_diversity.get("unique_count", 0)
            embed_spread = dd.task_string_diversity.get("embedding_spread", 0)

            # Assess quality
            def _assess_bg(s: float) -> str:
                if s < 0.05:
                    return "Very low — near-identical backgrounds across episodes"
                if s < 0.2:
                    return "Low — limited background variation"
                if s < 0.5:
                    return "Moderate"
                return "Good — diverse backgrounds"

            def _assess_tasks(n: int) -> str:
                if n <= 1:
                    return "No language diversity — memorization risk"
                if n < 5:
                    return "Low — limited paraphrasing"
                return "Good"

            def _assess_spread(s: float) -> str:
                if s < 0.01:
                    return "No semantic variation"
                if s < 0.1:
                    return "Low semantic diversity"
                return "Diverse instructions"

            sections.append("| Metric | Value | Assessment |")
            sections.append("|---|---|---|")
            sections.append(
                f"| Background diversity | {bg_score:.4f} | {_assess_bg(bg_score)} |"
            )
            sections.append(
                f"| Mean brightness | {float(brightness):.4f} | "
                f"{'Dark' if float(brightness) < 60 else 'Normal' if float(brightness) < 150 else 'Bright'} "
                f"(0\u2013255 scale) |"
            )
            sections.append(
                f"| Contrast variance | {float(contrast):.4f} | "
                f"{'Low' if float(contrast) < 100 else 'Normal' if float(contrast) < 500 else 'High'} "
                f"(higher = more contrast variation) |"
            )
            sections.append(
                f"| Unique task strings | {unique_tasks} | {_assess_tasks(unique_tasks)} |"
            )
            sections.append(
                f"| Task embedding spread | {float(embed_spread):.4f} | {_assess_spread(float(embed_spread))} |"
            )
            sections.append("")

            sections.append(
                "*Background diversity (0\u20131): cosine distance between average "
                "background patches across episodes. 0 = identical, 1 = maximally "
                "different. Task embedding spread: standard deviation of SigLIP text "
                "embeddings for unique task strings; 0 = all identical.*"
            )
            sections.append("")

            # Object position stats
            sections.append("### Object Position Variability")
            sections.append("")
            sections.append(
                "Position statistics across sampled frames. Low standard deviation "
                "(< 15px) indicates the object appears in nearly the same position "
                "every episode — a memorization risk."
            )
            sections.append("")
            sections.append("| Object | Count | Mean (x, y) | Std Dev (x, y) | Risk |")
            sections.append("|---|---|---|---|---|")
            for obj_label, stats in dd.object_position_stats.items():
                count = stats.get("count", 0)
                mx = stats.get("mean_x", 0)
                my = stats.get("mean_y", 0)
                sx = stats.get("std_x", 0)
                sy = stats.get("std_y", 0)
                risk = ""
                if sx < 15 and sy < 15:
                    risk = "HIGH — near-fixed position"
                elif sx < 30 and sy < 30:
                    risk = "Medium"
                else:
                    risk = "Low"
                sections.append(
                    f"| {obj_label} | {count} | ({mx:.1f}, {my:.1f}) | "
                    f"({sx:.1f}, {sy:.1f}) | {risk} |"
                )
            sections.append("")

        # ── Diagnostic Matrix ─────────────────────────────────────
        sections.append("## Diagnostic Matrix")
        sections.append("")
        sections.append(self.matrix.to_markdown())
        sections.append("")

        # ── Spatial vs Object Diagnosis ──────────────────────────
        if self.spatial_object_diagnosis is not None:
            diag = self.spatial_object_diagnosis
            verdict_labels = {
                "spatial_prior": "Primarily spatial-prior driven",
                "object_grounded": "Primarily object-feature grounded",
                "mixed": "Mixed strategy",
                "inconclusive": "Inconclusive",
            }
            sections.append("## Spatial vs Object Learning")
            sections.append("")
            sections.append(diag.summary)
            sections.append("")
            sections.append("| Metric | Value |")
            sections.append("|---|---|")
            sections.append(f"| Target object | {diag.target_object or 'N/A'} |")
            sections.append(f"| Verdict | {verdict_labels.get(diag.verdict, diag.verdict)} |")
            sections.append(f"| Confidence | {diag.confidence:.0%} |")
            sections.append(f"| Spatial-prior score | {diag.spatial_score:.4f} |")
            sections.append(f"| Object-grounding score | {diag.object_score:.4f} |")
            for key, val in diag.key_metrics.items():
                if isinstance(val, float):
                    rendered = f"{val:.4f}"
                else:
                    rendered = str(val)
                sections.append(f"| {key.replace('_', ' ').title()} | {rendered} |")
            sections.append("")

            if diag.spatial_evidence:
                sections.append("### Evidence For Spatial Priors")
                sections.append("")
                for item in diag.spatial_evidence:
                    sections.append(f"- {item.summary} (score={item.score:.2f}, source={item.source})")
                sections.append("")

            if diag.object_evidence:
                sections.append("### Evidence For Object Grounding")
                sections.append("")
                for item in diag.object_evidence:
                    sections.append(f"- {item.summary} (score={item.score:.2f}, source={item.source})")
                sections.append("")

        # ── Semantic Probe ───────────────────────────────────────
        if self.semantic_probe is not None:
            probe = self.semantic_probe
            sections.append("## Semantic Feature Probe")
            sections.append("")
            sections.append(probe.summary)
            sections.append("")

            if probe.primary_frame is not None:
                frame = probe.primary_frame
                sections.append("### Primary Frame")
                sections.append("")
                sections.append("| Metric | Value |")
                sections.append("|---|---|")
                sections.append(f"| Target object | {frame.target_object or 'N/A'} |")
                sections.append(f"| Candidate labels | {', '.join(frame.candidate_labels)} |")
                sections.append(f"| Best non-target label | {frame.best_non_target_label or 'N/A'} |")
                for key in (
                    "target_semantic_peak",
                    "target_semantic_mean_on_causal_patches",
                    "target_margin_over_best_non_target",
                    "causal_semantic_alignment",
                    "background_semantic_gap",
                ):
                    value = getattr(frame, key)
                    rendered = f"{value:.4f}" if isinstance(value, float) else "N/A"
                    sections.append(f"| {key.replace('_', ' ').title()} | {rendered} |")
                if frame.summary:
                    sections.append("")
                    sections.append(frame.summary)
                sections.append("")

            if probe.counterfactuals:
                sections.append("### Counterfactual Semantic Follow-Through")
                sections.append("")
                for name, summary in probe.counterfactuals.items():
                    sections.append(f"- **{name.replace('_', ' ')}**: {summary.summary}")
                sections.append("")

        # ── QK Probe ─────────────────────────────────────────────
        if self.qk_probe is not None:
            probe = self.qk_probe
            sections.append("## QK Decomposition")
            sections.append("")
            sections.append(probe.summary)
            sections.append("")
            sections.append("| Metric | Value |")
            sections.append("|---|---|")
            sections.append(f"| Layer | {probe.layer} |")
            sections.append(f"| Dominant head type | {probe.dominant_head_type} |")
            sections.append(f"| Semantic head fraction | {probe.semantic_head_fraction:.4f} |")
            sections.append(f"| Positional head fraction | {probe.positional_head_fraction:.4f} |")
            sections.append(f"| Mixed head fraction | {probe.mixed_head_fraction:.4f} |")
            sections.append("")
            if probe.top_heads:
                sections.append("### Top Heads")
                sections.append("")
                sections.append("| Head | Type | Score | Semantic Corr | Positional Corr | Target Mass | Background Mass |")
                sections.append("|---|---|---|---|---|---|---|")
                for head in probe.top_heads:
                    sections.append(
                        f"| {head.head_index} | {head.head_type} | {head.score:.4f} | "
                        f"{head.semantic_map_correlation:.4f} | {head.positional_baseline_correlation:.4f} | "
                        f"{head.target_region_logit_mass:.4f} | {head.background_logit_mass:.4f} |"
                    )
                sections.append("")

        # ── Anomalies ─────────────────────────────────────────────
        sections.append("## Detected Anomalies")
        sections.append("")
        if self.anomalies:
            sections.append("| Severity | Anomaly | Description |")
            sections.append("|---|---|---|")
            for a in self.anomalies:
                icon = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}.get(
                    a.severity, a.severity.upper()
                )
                sections.append(f"| {icon} | {a.type} | {a.description} |")
        else:
            sections.append("No anomalies detected.")
        sections.append("")

        # ── Hypotheses + Counterfactual Results (merged) ──────────
        sections.append("## Hypotheses & Counterfactual Tests")
        sections.append("")
        sections.append(
            "Each hypothesis is a testable claim about model behavior derived from "
            "the anomalies above. Confidence (0\u2013100%) reflects how strongly the "
            "diagnostic evidence supports the hypothesis *before* running the "
            "counterfactual test. The counterfactual test then attempts to confirm "
            "or reject the hypothesis by applying a controlled perturbation (e.g., "
            "moving an object, swapping the background) and measuring the change in "
            "the model's predicted actions."
        )
        sections.append("")

        if self.hypotheses:
            # Build a lookup from hypothesis_id to counterfactual result
            cf_map = {cr.hypothesis_id: cr for cr in self.counterfactual_results}

            for h in self.hypotheses:
                cr = cf_map.get(h.id)
                if cr:
                    verdict = "CONFIRMED" if cr.confirmed else "NOT CONFIRMED"
                    verdict_icon = "CONFIRMED" if cr.confirmed else "NOT CONFIRMED"
                else:
                    verdict = "Not tested"
                    verdict_icon = "UNTESTED"

                sections.append(f"### {h.id}: {h.description}")
                sections.append("")

                # Summary table
                sections.append(f"| | |")
                sections.append(f"|---|---|")
                sections.append(f"| **Confidence** | {h.confidence:.0%} |")
                sections.append(f"| **Supporting anomalies** | {', '.join(h.supporting_anomalies)} |")
                sections.append(f"| **Test type** | {h.test_type} |")

                if cr:
                    sections.append(f"| **Verdict** | **{verdict_icon}** |")
                    sections.append(f"| **Action delta (L2)** | {cr.action_delta_l2:.4f} |")
                    if cr.gradcam_shift > 0:
                        sections.append(f"| **GradCAM shift** | {cr.gradcam_shift:.4f} |")
                    if cr.metrics:
                        metric_parts = []
                        for key, value in cr.metrics.items():
                            if isinstance(value, float):
                                metric_parts.append(f"{key}={value:.4f}")
                            else:
                                metric_parts.append(f"{key}={value}")
                        sections.append(f"| **Probe metrics** | {'; '.join(metric_parts)} |")
                    if cr.attribution_shift_per_region:
                        shifts = ", ".join(
                            f"{k}: {v:+.4f}" for k, v in cr.attribution_shift_per_region.items()
                        )
                        sections.append(f"| **Attribution shifts** | {shifts} |")
                elif h.test_type == "none":
                    sections.append(f"| **Verdict** | Matrix evidence sufficient (no test needed) |")
                else:
                    reason_msg = {
                        "skipped": "Counterfactuals skipped by user",
                        "no_model": "No model loaded (post-hoc mode)",
                        "budget_exhausted": "Counterfactual budget exhausted",
                    }.get(self.cf_skip_reason, "Not run")
                    sections.append(f"| **Verdict** | {reason_msg} |")

                sections.append("")

                # Expected outcomes
                sections.append(f"- **If true**: {h.expected_if_true}")
                sections.append(f"- **If false**: {h.expected_if_false}")
                sections.append("")
        else:
            sections.append("No hypotheses generated.")
            sections.append("")

        # ── Findings ──────────────────────────────────────────────
        sections.append("## Findings & Recommendations")
        sections.append("")
        if self.findings:
            for i, f in enumerate(self.findings):
                sev = f.severity.upper()
                sections.append(f"### {i+1}. [{sev}] {f.title}")
                sections.append("")
                sections.append(f"**Observation**: {f.observation}")
                sections.append("")
                if f.test_description and f.test_description != "See counterfactual results below.":
                    sections.append(f"**Test**: {f.test_description}")
                    sections.append("")
                if f.test_result:
                    sections.append(f"**Result**: {f.test_result}")
                    sections.append("")
                sections.append(f"**Interpretation**: {f.interpretation}")
                sections.append("")
                sections.append(f"**Recommended fix**: {f.fix}")
                sections.append("")
                sections.append(f"**Expected impact**: {f.expected_impact}")
                sections.append("")
        else:
            sections.append("No findings.")
            sections.append("")

        # ── Narrative Synthesis ───────────────────────────────────
        if self.llm_synthesis:
            sections.append("## Overall Assessment")
            sections.append("")
            sections.append(self.llm_synthesis)
            sections.append("")

        return "\n".join(sections)

    # -- persistence -------------------------------------------------------

    def save(self, run_dir: str) -> tuple[str, str]:
        """Save report to ``run_dir/diagnostic/`` as ``report.json`` and ``report.md``.

        Returns the paths to the JSON and Markdown files.
        """
        out_dir = os.path.join(run_dir, "diagnostic")
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, "report.json")
        md_path = os.path.join(out_dir, "report.md")

        with open(json_path, "w") as f:
            f.write(self.to_json())

        with open(md_path, "w") as f:
            f.write(self.to_markdown())

        return json_path, md_path


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

@dataclass
class RunSnapshot:
    """Loaded data for one run being compared."""

    label: str
    run_dir: str
    diagnostic: dict  # parsed report.json
    internals: dict | None = None  # parsed model_internals section


@dataclass
class AttributionDelta:
    """Change in attribution for one signal/region pair across two runs."""

    signal: str
    region: str
    values: list[float]  # one per run, in run order
    delta: float  # last - first
    pct_change: float | None  # percentage change (None if base is 0)


@dataclass
class CounterfactualDelta:
    """Change in a counterfactual test result across runs."""

    test_type: str
    values: list[float | None]  # action_delta_l2 per run (None = not run)
    delta: float | None  # None if either value missing
    pct_change: float | None
    confirmed: list[bool | None]  # per run (None = not run)


@dataclass
class WeightAlphaDelta:
    """Change in WeightWatcher alpha for a component across runs."""

    component: str
    values: list[float]  # mean alpha per run
    delta: float
    pct_change: float | None
    status: list[str]  # per run


@dataclass
class ComparisonReport:
    """Comparison of two or more diagnostic runs."""

    labels: list[str]
    run_dirs: list[str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    attribution_deltas: list[AttributionDelta] = field(default_factory=list)
    scalar_deltas: dict[str, list[float | None]] = field(default_factory=dict)
    counterfactual_deltas: list[CounterfactualDelta] = field(default_factory=list)
    anomaly_summary: dict[str, list[str]] = field(default_factory=dict)
    weight_alpha_deltas: list[WeightAlphaDelta] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    verdict: str = ""

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, default=_numpy_serialiser)

    # -- markdown ----------------------------------------------------------

    def to_markdown(self) -> str:
        s: list[str] = []
        s.append("# Comparison Report")
        s.append("")
        s.append(f"Generated: {self.created_at}")
        s.append("")

        # Runs table
        s.append("## Runs")
        s.append("")
        s.append("| # | Label | Run Directory |")
        s.append("|---|---|---|")
        for i, (label, rd) in enumerate(zip(self.labels, self.run_dirs)):
            s.append(f"| {i + 1} | {label} | `{rd}` |")
        s.append("")

        # Attribution matrix comparison
        if self.attribution_deltas:
            s.append("## Attribution Matrix Comparison")
            s.append("")
            hdr = "| Signal | Region | " + " | ".join(self.labels) + " | Delta | Change |"
            sep = "|" + "---|" * (len(self.labels) + 4)
            s.append(hdr)
            s.append(sep)
            for ad in self.attribution_deltas:
                vals = " | ".join(f"{v:.4f}" for v in ad.values)
                sign = "+" if ad.delta > 0 else ""
                pct = f"{ad.pct_change:+.1f}%" if ad.pct_change is not None else "N/A"
                s.append(f"| {ad.signal} | {ad.region} | {vals} | {sign}{ad.delta:.4f} | {pct} |")
            s.append("")

        # Scalars
        if self.scalar_deltas:
            s.append("## Scalar Comparison")
            s.append("")
            hdr = "| Scalar | " + " | ".join(self.labels) + " | Delta |"
            sep = "|" + "---|" * (len(self.labels) + 2)
            s.append(hdr)
            s.append(sep)
            for key, vals in self.scalar_deltas.items():
                vcells = " | ".join(f"{v:.6f}" if v is not None else "N/A" for v in vals)
                if any(v is None for v in vals):
                    s.append(f"| {key} | {vcells} | N/A |")
                else:
                    delta = vals[-1] - vals[0]
                    sign = "+" if delta > 0 else ""
                    s.append(f"| {key} | {vcells} | {sign}{delta:.6f} |")
            s.append("")

        # Counterfactual comparison
        if self.counterfactual_deltas:
            s.append("## Counterfactual Comparison")
            s.append("")
            hdr = "| Test | " + " | ".join(f"{l} (L2)" for l in self.labels) + " | Delta | Change | Confirmed |"
            sep = "|" + "---|" * (len(self.labels) + 4)
            s.append(hdr)
            s.append(sep)
            for cd in self.counterfactual_deltas:
                vals = " | ".join(f"{v:.4f}" if v is not None else "N/A" for v in cd.values)
                if cd.delta is not None:
                    sign = "+" if cd.delta > 0 else ""
                    delta_str = f"{sign}{cd.delta:.4f}"
                else:
                    delta_str = "N/A"
                pct = f"{cd.pct_change:+.1f}%" if cd.pct_change is not None else "N/A"
                conf = " → ".join("Y" if c else ("N" if c is not None else "—") for c in cd.confirmed)
                s.append(f"| {cd.test_type} | {vals} | {delta_str} | {pct} | {conf} |")
            s.append("")

        # Anomaly summary
        if self.anomaly_summary:
            s.append("## Anomaly Comparison")
            s.append("")
            all_types: list[str] = []
            for types in self.anomaly_summary.values():
                for t in types:
                    if t not in all_types:
                        all_types.append(t)
            if all_types:
                hdr = "| Anomaly | " + " | ".join(self.labels) + " |"
                sep = "|" + "---|" * (len(self.labels) + 1)
                s.append(hdr)
                s.append(sep)
                for atype in all_types:
                    cells = []
                    for label in self.labels:
                        present = atype in self.anomaly_summary.get(label, [])
                        cells.append("PRESENT" if present else "resolved")
                    s.append(f"| {atype} | " + " | ".join(cells) + " |")
                s.append("")

        # Weight spectral comparison
        if self.weight_alpha_deltas:
            s.append("## Weight Spectral Analysis (alpha)")
            s.append("")
            hdr = "| Component | " + " | ".join(f"{l} (alpha)" for l in self.labels) + " | Delta | Change | Status |"
            sep = "|" + "---|" * (len(self.labels) + 4)
            s.append(hdr)
            s.append(sep)
            for wa in self.weight_alpha_deltas:
                vals = " | ".join(f"{v:.2f}" for v in wa.values)
                sign = "+" if wa.delta > 0 else ""
                pct = f"{wa.pct_change:+.1f}%" if wa.pct_change is not None else "N/A"
                status = " → ".join(wa.status)
                s.append(f"| {wa.component} | {vals} | {sign}{wa.delta:.2f} | {pct} | {status} |")
            s.append("")

        # Verdict
        if self.verdict:
            s.append("## Verdict")
            s.append("")
            s.append(self.verdict)
            s.append("")

        # Recommendations
        if self.recommendations:
            s.append("## Recommendations")
            s.append("")
            for i, rec in enumerate(self.recommendations, 1):
                s.append(f"{i}. {rec}")
            s.append("")

        return "\n".join(s)

    # -- persistence -------------------------------------------------------

    def save(self, output_dir: str) -> tuple[str, str]:
        """Save to *output_dir* as ``comparison_report.json`` and ``.md``."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "comparison_report.json")
        md_path = os.path.join(output_dir, "comparison_report.md")
        with open(json_path, "w") as f:
            f.write(self.to_json())
        with open(md_path, "w") as f:
            f.write(self.to_markdown())
        return json_path, md_path

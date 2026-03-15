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

    # -- deserialisation ---------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "DiagnosticReport":
        """Reconstruct a *DiagnosticReport* from a JSON-loaded dict.

        This is the inverse of :meth:`to_json` — it recreates all nested
        dataclass objects so that :meth:`to_markdown` can be called again
        (e.g. after changing the report template).

        Masks are **not** restored (they are stored as shapes only in JSON).
        """
        # Scene
        scene_data = data.get("scene", {})
        objects = []
        for o in scene_data.get("objects", []):
            objects.append(DetectedObject(
                label=o["label"],
                box=tuple(o["box"]),
                score=o["score"],
                mask=None,
            ))
        image_shape = tuple(scene_data.get("image_shape", (0, 0)))
        bg_shape = tuple(scene_data.get("background_mask_shape", image_shape))
        scene = SceneSegmentation(
            objects=objects,
            background_mask=np.zeros(bg_shape, dtype=bool),
            image_shape=image_shape,
        )

        # Dataset diversity
        dd_data = data.get("dataset_diversity")
        dataset_diversity = None
        if dd_data:
            dataset_diversity = DatasetDiversityReport(**dd_data)

        # Matrix
        m = data.get("matrix", {})
        matrix = DiagnosticMatrix(
            signal_types=m.get("signal_types", []),
            regions=m.get("regions", []),
            attribution_mass=m.get("attribution_mass", {}),
            per_frame=m.get("per_frame", {}),
            scalars=m.get("scalars"),
            temporal_trajectories=[
                TemporalTrajectory(**t) for t in m.get("temporal_trajectories", [])
            ] if m.get("temporal_trajectories") else None,
            occlusion=OcclusionSensitivity(**m["occlusion"]) if m.get("occlusion") else None,
            connector_analysis=(
                ConnectorAnalysis(**m["connector_analysis"])
                if m.get("connector_analysis") else None
            ),
            per_action_dim=m.get("per_action_dim"),
        )

        # Semantic probe
        sp_data = data.get("semantic_probe")
        semantic_probe = None
        if sp_data:
            primary = sp_data.get("primary_frame")
            primary_frame = SemanticFrameSummary(**primary) if primary else None
            frames = [SemanticFrameSummary(**f) for f in sp_data.get("frames", [])]
            cfs = {}
            for name, cf_data in sp_data.get("counterfactuals", {}).items():
                cfs[name] = SemanticCounterfactualSummary(**cf_data)
            semantic_probe = SemanticProbeReport(
                target_object=sp_data.get("target_object"),
                summary=sp_data.get("summary", ""),
                primary_frame=primary_frame,
                frames=frames,
                counterfactuals=cfs,
            )

        # QK probe
        qk_data = data.get("qk_probe")
        qk_probe = None
        if qk_data:
            top_heads = [QKHeadSummary(**h) for h in qk_data.get("top_heads", [])]
            qk_probe = QKProbeReport(
                layer=qk_data.get("layer", ""),
                summary=qk_data.get("summary", ""),
                dominant_head_type=qk_data.get("dominant_head_type", ""),
                semantic_head_fraction=qk_data.get("semantic_head_fraction", 0),
                positional_head_fraction=qk_data.get("positional_head_fraction", 0),
                mixed_head_fraction=qk_data.get("mixed_head_fraction", 0),
                top_heads=top_heads,
            )

        # Spatial object diagnosis
        sod_data = data.get("spatial_object_diagnosis")
        spatial_object_diagnosis = None
        if sod_data:
            spatial_ev = [
                DiagnosticEvidence(**e) for e in sod_data.get("spatial_evidence", [])
            ]
            object_ev = [
                DiagnosticEvidence(**e) for e in sod_data.get("object_evidence", [])
            ]
            spatial_object_diagnosis = SpatialObjectDiagnosis(
                target_object=sod_data.get("target_object"),
                verdict=sod_data.get("verdict", "inconclusive"),
                confidence=sod_data.get("confidence", 0),
                spatial_score=sod_data.get("spatial_score", 0),
                object_score=sod_data.get("object_score", 0),
                summary=sod_data.get("summary", ""),
                key_metrics=sod_data.get("key_metrics", {}),
                spatial_evidence=spatial_ev,
                object_evidence=object_ev,
            )

        # Anomalies
        anomalies = [Anomaly(**a) for a in data.get("anomalies", [])]

        # Hypotheses
        hypotheses = [Hypothesis(**h) for h in data.get("hypotheses", [])]

        # Counterfactual results
        cf_results = []
        for cr in data.get("counterfactual_results", []):
            cf_results.append(CounterfactualResult(
                hypothesis_id=cr["hypothesis_id"],
                test_type=cr["test_type"],
                action_delta_l2=cr.get("action_delta_l2", 0),
                action_delta_per_dim=cr.get("action_delta_per_dim", []),
                gradcam_shift=cr.get("gradcam_shift", 0),
                attribution_shift_per_region=cr.get("attribution_shift_per_region", {}),
                confirmed=cr.get("confirmed", False),
                metrics=cr.get("metrics", {}),
                visual_comparison=None,
            ))

        # Findings
        findings = [Finding(**f) for f in data.get("findings", [])]

        return cls(
            metadata=data.get("metadata", {}),
            scene=scene,
            dataset_diversity=dataset_diversity,
            matrix=matrix,
            semantic_probe=semantic_probe,
            qk_probe=qk_probe,
            spatial_object_diagnosis=spatial_object_diagnosis,
            anomalies=anomalies,
            hypotheses=hypotheses,
            counterfactual_results=cf_results,
            findings=findings,
            llm_synthesis=data.get("llm_synthesis", ""),
            cf_skip_reason=data.get("cf_skip_reason", ""),
        )

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
            verdict_explanations = {
                "spatial_prior": (
                    "The model appears to rely on memorised positions rather "
                    "than recognising objects by their visual features. It may "
                    "fail when objects move to new positions."
                ),
                "object_grounded": (
                    "The model appears to recognise objects by their visual "
                    "appearance (shape, colour, texture) rather than relying "
                    "on fixed positions."
                ),
                "mixed": (
                    "The model uses a combination of spatial-position cues "
                    "and object-feature cues. Some behaviours may generalise "
                    "while others may be brittle to layout changes."
                ),
                "inconclusive": (
                    "There is not enough evidence to determine whether the "
                    "model relies on spatial position or object features. "
                    "Consider collecting more data or running additional "
                    "counterfactual tests."
                ),
            }
            sections.append("## Spatial vs Object Learning")
            sections.append("")
            sections.append(diag.summary)
            sections.append("")
            sections.append(
                "*This section answers a key question: does the model recognise "
                "objects by their appearance, or does it simply memorise where "
                "objects usually appear? Terms marked with superscript numbers "
                "(e.g. \u00b9) are defined in the Glossary at the end of this report.*"
            )
            sections.append("")

            # Verdict + top-level scores as bullet list (avoids wide table)
            sections.append(
                f"- **Target object:** {diag.target_object or 'N/A'}"
            )
            sections.append(
                f"- **Verdict:** {verdict_labels.get(diag.verdict, diag.verdict)} "
                f"— {verdict_explanations.get(diag.verdict, '')}"
            )
            sections.append(
                f"- **Confidence:** {diag.confidence:.0%}"
            )
            sections.append(
                f"- **Spatial-prior score\u00b9:** {diag.spatial_score:.4f} "
                f"(0–1, higher = model memorises positions)"
            )
            sections.append(
                f"- **Object-grounding score\u00b2:** {diag.object_score:.4f} "
                f"(0–1, higher = model recognises objects)"
            )
            sections.append("")

            # Render key_metrics with explanations
            # Format: (scale, short_description)
            _KEY_METRIC_INFO: dict[str, tuple[str, str]] = {
                "positional_baseline_ratio": (
                    "0–1",
                    "Similarity to blank-image baseline\u00b3. >0.6 = spatial shortcut",
                ),
                "foreground_ratio": (
                    "0–1",
                    "Causal attention on foreground vs background\u2074. <0.3 = weak",
                ),
                "vision_share": (
                    "0–1",
                    "Vision vs state gradient fraction\u2075. >0.99 = state ignored",
                ),
                "focus_follow_ratio": (
                    "0–1",
                    "Causal attention that followed moved object\u2076. Higher = object-grounded",
                ),
                "anchor_retention_ratio": (
                    "0–1",
                    "Causal attention that stayed at old position\u2076. Higher = spatial-prior",
                ),
                "semantic_follow_ratio": (
                    "0–1",
                    "Semantic evidence at new object location\u2076",
                ),
                "semantic_anchor_ratio": (
                    "0–1",
                    "Semantic evidence remaining at old location\u2076",
                ),
                "relocation_follow_ratio": (
                    "0–1",
                    "Causal attention that followed moved object\u2076. Higher = object-grounded",
                ),
                "relocation_anchor_ratio": (
                    "0–1",
                    "Causal attention that stayed at old position\u2076. Higher = spatial-prior",
                ),
                "relocation_semantic_follow_ratio": (
                    "0–1",
                    "Semantic evidence at new object location\u2076",
                ),
                "relocation_semantic_anchor_ratio": (
                    "0–1",
                    "Semantic evidence remaining at old location\u2076",
                ),
                "target_gradcam_share": (
                    "0–1",
                    "Fraction of GradCAM attribution on the target object",
                ),
                "target_attention_share": (
                    "0–1",
                    "Fraction of self-attention on the target object",
                ),
                "target_semantic_peak": (
                    "0–1",
                    "Peak cosine similarity between any patch and target label\u2077",
                ),
                "target_semantic_mean_on_causal_patches": (
                    "0–1",
                    "Mean target similarity on causal patches\u2077\u00b7\u2078",
                ),
                "target_margin_over_best_non_target": (
                    "\u22121 to 1",
                    "Target similarity minus best non-target\u2079. <0.08 = weak",
                ),
                "causal_semantic_alignment": (
                    "0–1",
                    "Alignment between target similarity map and GradCAM\u2078",
                ),
                "background_semantic_gap": (
                    "\u22121 to 1",
                    "Target similarity on causal patches minus on background\u2078",
                ),
                "target_position_std_px": (
                    "pixels",
                    "Std dev of target object position across episodes. <15 = near-fixed",
                ),
                "occlusion_action_delta": (
                    "\u22650",
                    "Action L2 change when target is occluded\u00b9\u00b9. Higher = model needs the object",
                ),
                "occlusion_target_semantic_drop": (
                    "\u22121 to 1",
                    "Change in target similarity after occlusion. Negative = degraded",
                ),
                "background_action_delta": (
                    "\u22650",
                    "Action L2 change when background is replaced\u00b9\u00b9",
                ),
                "recolor_action_delta": (
                    "\u22650",
                    "Action L2 change when target is recoloured\u00b9\u00b9",
                ),
                "qk_semantic_head_fraction": (
                    "0–1",
                    "Fraction of attention heads classified as semantic\u00b9\u2070",
                ),
                "qk_positional_head_fraction": (
                    "0–1",
                    "Fraction of attention heads classified as positional\u00b9\u2070",
                ),
            }

            sections.append("| Metric | Value | Scale | Meaning |")
            sections.append("|---|---|---|---|")
            for key, val in diag.key_metrics.items():
                if isinstance(val, float):
                    rendered = f"{val:.4f}"
                else:
                    rendered = str(val)
                info = _KEY_METRIC_INFO.get(key)
                if info:
                    scale, interp = info
                else:
                    # Auto-infer: floats between 0–1 are likely ratios
                    if isinstance(val, float) and 0 <= val <= 1:
                        scale = "0–1"
                    elif isinstance(val, float):
                        scale = "\u22650"
                    else:
                        scale = "—"
                    interp = key.replace("_", " ")
                display_key = key.replace("_", " ").title()
                sections.append(
                    f"| {display_key} | {rendered} | {scale} | {interp} |"
                )
            sections.append("")

            if diag.spatial_evidence:
                sections.append("### Evidence For Spatial Priors")
                sections.append("")
                sections.append(
                    "*Each item below contributes to the spatial-prior score. "
                    "\"score\" is that item's contribution (0.0–1.0).*"
                )
                sections.append("")
                for item in diag.spatial_evidence:
                    sections.append(f"- {item.summary} (score={item.score:.2f}, source={item.source})")
                sections.append("")

            if diag.object_evidence:
                sections.append("### Evidence For Object Grounding")
                sections.append("")
                sections.append(
                    "*Each item below contributes to the object-grounding "
                    "score. \"score\" is that item's contribution (0.0–1.0).*"
                )
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
            sections.append(
                "*This probe checks whether the vision encoder's internal "
                "patch representations actually encode the target object's "
                "identity. It compares each patch's embedding to text "
                "descriptions of each object using cosine similarity. "
                "All similarity scores below are on a 0.0–1.0 scale.*"
            )
            sections.append("")

            if probe.primary_frame is not None:
                frame = probe.primary_frame
                sections.append(
                    "### Analysed Frame"
                )
                sections.append("")
                sections.append(
                    "*This is the main frame from the sampled episode that "
                    "the semantic probe was run on. If counterfactual "
                    "follow-through results appear below, those are computed "
                    "on modified versions of this same frame.*"
                )
                sections.append("")

                _SEMANTIC_METRIC_INFO: dict[str, tuple[str, str]] = {
                    "target_semantic_peak": (
                        "0–1",
                        "Peak patch-to-text similarity for target\u2077",
                    ),
                    "target_semantic_mean_on_causal_patches": (
                        "0–1",
                        "Mean target similarity on causal patches\u2077\u00b7\u2078",
                    ),
                    "target_margin_over_best_non_target": (
                        "\u22121 to 1",
                        "Target minus best non-target similarity\u2079. <0.08 = weak",
                    ),
                    "causal_semantic_alignment": (
                        "0–1",
                        "Alignment of target similarity with GradCAM map\u2078",
                    ),
                    "background_semantic_gap": (
                        "\u22121 to 1",
                        "Causal vs background target similarity\u2078. Positive = good",
                    ),
                }

                sections.append("| Metric | Value | Scale | Meaning |")
                sections.append("|---|---|---|---|")
                sections.append(f"| Target object | {frame.target_object or 'N/A'} | — | Object being probed |")
                sections.append(f"| Candidate labels | {', '.join(frame.candidate_labels)} | — | All labels tested |")
                sections.append(f"| Best non-target label | {frame.best_non_target_label or 'N/A'} | — | Strongest non-target match (used for margin) |")
                for key in (
                    "target_semantic_peak",
                    "target_semantic_mean_on_causal_patches",
                    "target_margin_over_best_non_target",
                    "causal_semantic_alignment",
                    "background_semantic_gap",
                ):
                    value = getattr(frame, key)
                    rendered = f"{value:.4f}" if isinstance(value, float) else "N/A"
                    scale, interp = _SEMANTIC_METRIC_INFO.get(
                        key, ("—", "See Glossary")
                    )
                    display_key = key.replace("_", " ").title()
                    sections.append(
                        f"| {display_key} | {rendered} | {scale} | {interp} |"
                    )
                if frame.summary:
                    sections.append("")
                    sections.append(frame.summary)
                sections.append("")

            if probe.counterfactuals:
                sections.append("### Counterfactual Semantic Follow-Through")
                sections.append("")
                sections.append(
                    "*After running each counterfactual (e.g., relocating or "
                    "occluding the object), the semantic probe is re-run to "
                    "see how the encoder's representation of the target "
                    "object changed.*"
                )
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
            sections.append(
                "*This analysis classifies each attention head in the "
                "vision encoder's last layer as \"semantic\" (attends to "
                "object features), \"positional\" (attends to fixed spatial "
                "locations), or \"mixed\". A model dominated by positional "
                "heads may be using spatial shortcuts\u00b9\u2070.*"
            )
            sections.append("")
            sections.append("| Metric | Value | Scale | Meaning |")
            sections.append("|---|---|---|---|")
            sections.append(f"| Layer | {probe.layer} | — | Encoder layer analysed |")
            sections.append(
                f"| Dominant head type | {probe.dominant_head_type} "
                f"| — | Most common head type |"
            )
            sections.append(
                f"| Semantic head fraction | {probe.semantic_head_fraction:.4f} "
                f"| 0–1 | Heads attending to visual content\u00b9\u2070 |"
            )
            sections.append(
                f"| Positional head fraction | {probe.positional_head_fraction:.4f} "
                f"| 0–1 | Heads attending to spatial position\u00b9\u2070 |"
            )
            sections.append(
                f"| Mixed head fraction | {probe.mixed_head_fraction:.4f} "
                f"| 0–1 | Heads using both strategies |"
            )
            sections.append("")
            if probe.top_heads:
                sections.append(
                    "### Most Relevant Heads (ranked by absolute score)"
                )
                sections.append("")
                sections.append(
                    "*These are the 3 heads with the highest absolute "
                    "classification score. \"Score\" (−1 to +1) measures "
                    "how strongly semantic vs positional the head is: "
                    "positive = semantic, negative = positional. Correlations "
                    "are Pearson r (−1 to +1). Mass values (0.0–1.0) show "
                    "what fraction of the head's attention lands on the "
                    "target vs background region.*"
                )
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
            "Each hypothesis is a testable claim about model behaviour derived "
            "from the anomalies above. The system tests each hypothesis by "
            "applying a controlled perturbation to the input (e.g., moving an "
            "object, swapping the background) and measuring how much the model's "
            "predicted actions change."
        )
        sections.append("")
        sections.append(
            "**How to read the verdicts:**"
        )
        sections.append("")
        sections.append(
            "- **CONFIRMED** — the counterfactual test produced evidence "
            "that supports the hypothesis. The hypothesis is likely true "
            "and should be addressed."
        )
        sections.append(
            "- **NOT CONFIRMED** — the counterfactual test did NOT find "
            "evidence for this hypothesis. This does not necessarily mean "
            "the hypothesis is wrong, but the specific test did not support "
            "it. The model may still have this issue via a mechanism the "
            "test does not capture."
        )
        sections.append("")
        sections.append(
            "**Confirmation threshold:** An action delta L2 > **0.02** is "
            "considered a significant change. For hypotheses where "
            "`confirms_on_change=true` (most cases), a significant change "
            "confirms the hypothesis. For hypotheses where "
            "`confirms_on_change=false` (e.g., language insensitivity), a "
            "*lack* of change confirms the hypothesis."
        )
        sections.append("")
        sections.append(
            "**Confidence** (0–100%) reflects how strongly the diagnostic "
            "evidence supports the hypothesis *before* running the test. "
            "If a hypothesis is not confirmed, its confidence is reduced "
            "by 60% in the final findings."
        )
        sections.append("")

        if self.hypotheses:
            # Build a lookup from hypothesis_id to counterfactual result
            cf_map = {cr.hypothesis_id: cr for cr in self.counterfactual_results}

            # Metric display names: (label, scale_hint)
            # scale_hint is appended in parentheses after the value
            _METRIC_DISPLAY: dict[str, tuple[str, str]] = {
                "target_object": ("Target object", ""),
                "fill": ("Occlusion fill type", ""),
                "replacement": ("Background replacement", ""),
                "shift_pixels": ("Shift distance", "pixels"),
                "focus_follow_ratio": ("Focus follow ratio", "0–1, higher = object-grounded"),
                "anchor_retention_ratio": ("Anchor retention ratio", "0–1, higher = spatial-prior"),
                "semantic_follow_ratio": ("Semantic follow ratio", "0–1"),
                "semantic_anchor_ratio": ("Semantic anchor ratio", "0–1"),
                "focus_shift_gap": ("Focus shift gap", "follow − anchor"),
                "semantic_shift_gap": ("Semantic shift gap", "follow − anchor"),
                "original_target_semantic_peak": ("Semantic peak (before)", "0–1"),
                "occluded_target_semantic_peak": ("Semantic peak (after occlusion)", "0–1"),
                "occlusion_target_semantic_drop": ("Semantic drop from occlusion", "negative = degraded"),
                "original_target_gradcam_share": ("GradCAM share (before)", "0–1"),
                "modified_old_anchor_share": ("GradCAM at old position (after)", "0–1"),
                "modified_new_object_share": ("GradCAM at new position (after)", "0–1"),
                "modified_old_anchor_target_semantic": ("Semantic at old position (after)", "0–1"),
                "modified_new_object_target_semantic": ("Semantic at new position (after)", "0–1"),
                "hue_shift": ("Hue shift applied", "0–1"),
                "brightness_delta": ("Brightness change", ""),
                "contrast_delta": ("Contrast change", ""),
                "position": ("Distractor position", "x, y pixels"),
                "size": ("Distractor size", "pixels"),
                "distractor_size": ("Distractor size", "pixels"),
                "distractor_source": ("Distractor source", ""),
                "replacement_task": ("Replacement task string", ""),
            }

            for h in self.hypotheses:
                cr = cf_map.get(h.id)

                sections.append(f"### {h.id}: {h.description}")
                sections.append("")

                # Summary table — use a proper header row
                sections.append("| Property | Details |")
                sections.append("|---|---|")
                sections.append(f"| **Confidence** | {h.confidence:.0%} |")
                sections.append(f"| **Supporting anomalies** | {', '.join(h.supporting_anomalies)} |")
                sections.append(f"| **Test type** | {h.test_type} |")

                if cr:
                    if cr.confirmed:
                        verdict_text = (
                            "**CONFIRMED** — the perturbation caused a "
                            "significant action change (L2 > 0.02), "
                            "supporting this hypothesis"
                        )
                        if getattr(h, "confirms_on_change", True) is False:
                            verdict_text = (
                                "**CONFIRMED** — the perturbation did NOT "
                                "cause a significant action change (L2 ≤ 0.02), "
                                "confirming the model is insensitive to this feature"
                            )
                    else:
                        verdict_text = (
                            "**NOT CONFIRMED** — the perturbation did not "
                            "produce the expected result. This hypothesis is "
                            "less likely but not definitively ruled out"
                        )
                        if getattr(h, "confirms_on_change", True) is False:
                            verdict_text = (
                                "**NOT CONFIRMED** — the perturbation caused "
                                "a significant action change (L2 > 0.02), "
                                "meaning the model IS sensitive to this "
                                "feature, contrary to the hypothesis"
                            )
                    sections.append(f"| **Verdict** | {verdict_text} |")
                    sections.append(
                        f"| **Action delta (L2)** | {cr.action_delta_l2:.4f} "
                        f"(threshold: 0.02) |"
                    )
                    if cr.gradcam_shift > 0:
                        sections.append(
                            f"| **GradCAM shift** | {cr.gradcam_shift:.4f} "
                            f"(how much the attention pattern changed) |"
                        )
                    if cr.metrics:
                        sections.append(f"| **Probe metrics** | — |")
                        for key, value in cr.metrics.items():
                            info = _METRIC_DISPLAY.get(key)
                            if info:
                                label, scale_hint = info
                            else:
                                label = key.replace("_", " ").title()
                                # Auto-infer scale for floats 0–1
                                if isinstance(value, float) and 0 <= value <= 1:
                                    scale_hint = "0–1"
                                elif isinstance(value, float):
                                    scale_hint = ""
                                else:
                                    scale_hint = ""
                            if isinstance(value, float):
                                val_str = f"{value:.4f}"
                            else:
                                val_str = str(value)
                            if scale_hint:
                                sections.append(
                                    f"| {label} | {val_str} ({scale_hint}) |"
                                )
                            else:
                                sections.append(
                                    f"| {label} | {val_str} |"
                                )
                    if cr.attribution_shift_per_region:
                        sections.append(
                            f"| **Attribution shifts** | + = gained, \u2212 = lost |"
                        )
                        for k, v in cr.attribution_shift_per_region.items():
                            sections.append(f"| {k} | {v:+.4f} |")
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

        # ── Glossary ──────────────────────────────────────────────
        sections.append("---")
        sections.append("")
        sections.append("## Glossary")
        sections.append("")
        sections.append(
            "*Definitions for technical terms used throughout this report. "
            "Superscript numbers (\u00b9\u00b2\u00b3...) in the sections above "
            "correspond to entries here.*"
        )
        sections.append("")
        glossary_entries = [
            (
                "1",
                "Spatial-prior score",
                "A weighted sum (0.0–1.0) of all evidence suggesting the "
                "model relies on memorised spatial positions rather than "
                "visual object features. Evidence includes: high positional "
                "baseline ratio, low foreground attribution, attention "
                "staying at the old location after object relocation, and "
                "low dataset position diversity. Higher = stronger spatial-"
                "prior behaviour.",
            ),
            (
                "2",
                "Object-grounding score",
                "A weighted sum (0.0–1.0) of all evidence suggesting the "
                "model recognises objects by their visual appearance. "
                "Evidence includes: high GradCAM attribution on the target "
                "object, semantic probe margin, attention following the "
                "object after relocation, and large action deltas when the "
                "target is occluded. Higher = stronger object grounding.",
            ),
            (
                "3",
                "Positional baseline ratio",
                "Cosine similarity (0.0–1.0) between the model's attention "
                "pattern on the actual image and its attention on a blank "
                "(all-zeros) image. A blank image has no visual content, so "
                "any attention pattern it produces is a pure spatial prior. "
                "If the real-image attention is very similar to the blank-"
                "image attention (ratio > 0.6), the model is attending to "
                "fixed positions regardless of what is actually in the scene.",
            ),
            (
                "4",
                "Foreground ratio",
                "Fraction (0.0–1.0) of the model's GradCAM or attention "
                "signal that lands on foreground objects (everything except "
                "background). A value of 0.3 means only 30% of the model's "
                "causal attention is on task-relevant objects; the remaining "
                "70% is on the background.",
            ),
            (
                "5",
                "Vision share",
                "Fraction (0.0–1.0) of the total gradient magnitude that "
                "flows through the vision (image) pathway vs the "
                "proprioceptive state pathway. Computed as: "
                "||∇_image|| / (||∇_image|| + ||∇_state||). "
                "A value > 0.99 means the model effectively ignores the "
                "robot's joint positions / state input.",
            ),
            (
                "6",
                "Relocation metrics (follow ratio, anchor ratio)",
                "After the object relocation counterfactual moves the "
                "target object to a new position:\n"
                "  - **Focus follow ratio**: fraction of the model's causal "
                "attention (GradCAM) that moved to the new location. "
                "High = object-grounded.\n"
                "  - **Anchor retention ratio**: fraction of causal "
                "attention that stayed at the OLD (now-empty) location. "
                "High = spatial-prior (the model is \"anchored\" to the "
                "memorised position).\n"
                "  - **Semantic follow/anchor ratios**: same idea but "
                "measured via the semantic probe (patch-to-text similarity) "
                "instead of GradCAM. These measure whether the encoder's "
                "internal representation of the object followed it or "
                "stayed behind.\n"
                "All ratios are 0.0–1.0. In a perfectly object-grounded "
                "model, follow ratios are high and anchor ratios are low.",
            ),
            (
                "7",
                "Semantic peak / similarity",
                "Cosine similarity (0.0–1.0) between a vision encoder "
                "patch embedding and a text embedding of the object label. "
                "Computed using the same SigLIP encoder that the model "
                "uses for vision. \"Peak\" is the maximum similarity across "
                "all patches. Higher means the encoder has strong internal "
                "evidence for the presence of that object.",
            ),
            (
                "8",
                "Causal patches",
                "The intersection of the top-5% GradCAM patches and the "
                "top-5% attention patches. These are the image patches that "
                "are both (a) looked at by the encoder AND (b) causally "
                "influence the action output. Measuring metrics on causal "
                "patches tells us about the patches the model actually "
                "*uses*, not just all patches in the image.",
            ),
            (
                "9",
                "Margin (target margin over best non-target)",
                "The difference in cosine similarity between the target "
                "object's text label and the best non-target object's text "
                "label, measured on causal patches. A positive margin means "
                "the encoder distinguishes the target from distractors. "
                "A margin < 0.08 means the encoder cannot reliably tell "
                "the target apart from other objects in the scene.",
            ),
            (
                "10",
                "QK head classification (semantic / positional / mixed)",
                "Each attention head in the vision encoder's last layer is "
                "classified by comparing:\n"
                "  - **Semantic correlation**: Pearson r between the head's "
                "attention map and the target-object text similarity map.\n"
                "  - **Positional correlation**: Pearson r between the "
                "head's attention map and a content-free positional baseline "
                "(blank-image attention).\n"
                "  - **Relocation bonus**: whether the head's attention "
                "follows a relocated object.\n"
                "Classification: advantage = semantic_corr − positional_corr "
                "+ relocation_bonus. If advantage > 0.08 → semantic; "
                "< −0.08 → positional; otherwise → mixed.\n"
                "\"Fraction\" values (0.0–1.0) show what proportion of "
                "all heads fall into each category.",
            ),
            (
                "11",
                "Action delta (L2)",
                "The L2 (Euclidean) norm of the difference between the "
                "model's predicted action on the original image and the "
                "predicted action on the modified image. A larger delta "
                "means the perturbation had a bigger effect on the model's "
                "behaviour. The confirmation threshold is 0.02 — deltas "
                "above this are considered significant.",
            ),
            (
                "12",
                "GradCAM shift",
                "The L2 distance between the GradCAM heatmap on the "
                "original image and the GradCAM heatmap on the modified "
                "image, after normalisation. Measures how much the model's "
                "attention pattern changed due to the perturbation.",
            ),
            (
                "13",
                "Attribution shift",
                "The change in each region's share of total causal "
                "attention (GradCAM) after a perturbation. Positive means "
                "that region gained attention; negative means it lost "
                "attention. Values are in the 0.0–1.0 scale (fractions of "
                "total attention).",
            ),
        ]
        _SUPER = {
            "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
            "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078",
            "9": "\u2079", "10": "\u00b9\u2070", "11": "\u00b9\u00b9",
            "12": "\u00b9\u00b2", "13": "\u00b9\u00b3",
        }
        for ref_num, term, definition in glossary_entries:
            sup = _SUPER.get(ref_num, ref_num)
            sections.append(f"**{sup} {term}**")
            sections.append("")
            sections.append(definition)
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

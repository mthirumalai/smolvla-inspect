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

        Signal types are rows; regions are columns.
        """
        lines: list[str] = []
        header = "| Signal \\ Region | " + " | ".join(self.regions) + " |"
        sep = "|" + "---|" * (len(self.regions) + 1)
        lines.append(header)
        lines.append(sep)
        for sig in self.signal_types:
            row_data = self.attribution_mass.get(sig, {})
            cells = [f"{row_data.get(r, 0.0):.4f}" for r in self.regions]
            lines.append(f"| {sig} | " + " | ".join(cells) + " |")

        if self.scalars:
            lines.append("")
            lines.append("**Scalars**")
            lines.append("")
            for key, val in self.scalars.items():
                lines.append(f"- {key}: {val}")

        if self.temporal_trajectories:
            lines.append("")
            lines.append("**Temporal Trajectories**")
            lines.append("")
            for t in self.temporal_trajectories:
                lines.append(
                    f"- {t.signal_type}: smoothness={t.smoothness:.3f}, "
                    f"object_tracking_corr={t.object_tracking_correlation:.3f}, "
                    f"{len(t.frame_indices)} frames"
                )

        if self.occlusion is not None:
            lines.append("")
            lines.append("**Occlusion Sensitivity**")
            lines.append("")
            lines.append(f"- Patch size: {self.occlusion.patch_size}, stride: {self.occlusion.stride}")
            lines.append(f"- Max delta: {self.occlusion.max_delta:.4f}, mean delta: {self.occlusion.mean_delta:.4f}")

        if self.connector_analysis is not None:
            ca = self.connector_analysis
            lines.append("")
            lines.append("**Connector Bottleneck Analysis**")
            lines.append("")
            lines.append(f"- Total information loss: {ca.total_information_loss:.4f}")
            for r, loss in ca.information_loss_per_region.items():
                lines.append(f"  - {r}: pre={ca.pre_connector_shares.get(r, 0):.4f} → post={ca.post_connector_shares.get(r, 0):.4f} (loss={loss:.4f})")

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
    visual_comparison: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Evidence & findings
# ---------------------------------------------------------------------------

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

        # Title
        sections.append("# Diagnostic Report")
        sections.append("")

        # Metadata
        sections.append("## Metadata")
        sections.append("")
        for k, v in self.metadata.items():
            sections.append(f"- **{k}**: {v}")
        sections.append("")

        # Scene
        sections.append("## Scene Segmentation")
        sections.append("")
        sections.append(f"Image shape: {self.scene.image_shape}")
        sections.append(f"Detected objects: {len(self.scene.objects)}")
        sections.append("")
        if self.scene.objects:
            sections.append("| Label | Box | Score |")
            sections.append("|---|---|---|")
            for obj in self.scene.objects:
                sections.append(f"| {obj.label} | {obj.box} | {obj.score:.3f} |")
            sections.append("")

        # Dataset diversity
        if self.dataset_diversity:
            dd = self.dataset_diversity
            sections.append("## Dataset Diversity")
            sections.append("")
            sections.append(f"- Episodes sampled: {dd.num_episodes_sampled}")
            sections.append(f"- Background diversity score: {dd.background_diversity_score:.3f}")
            sections.append(f"- Mean brightness: {dd.lighting_stats.get('mean_brightness', 'N/A')}")
            sections.append(f"- Contrast variance: {dd.lighting_stats.get('contrast_variance', 'N/A')}")
            sections.append(f"- Unique task strings: {dd.task_string_diversity.get('unique_count', 'N/A')}")
            sections.append(f"- Task embedding spread: {dd.task_string_diversity.get('embedding_spread', 'N/A')}")
            sections.append("")
            sections.append("### Object Position Stats")
            sections.append("")
            for obj_label, stats in dd.object_position_stats.items():
                sections.append(f"**{obj_label}**: {stats}")
            sections.append("")

        # Diagnostic matrix
        sections.append("## Diagnostic Matrix")
        sections.append("")
        sections.append(self.matrix.to_markdown())
        sections.append("")

        # Anomalies
        sections.append("## Anomalies")
        sections.append("")
        if self.anomalies:
            for a in self.anomalies:
                icon = {"critical": "[CRITICAL]", "warning": "[WARNING]", "info": "[INFO]"}.get(
                    a.severity, f"[{a.severity.upper()}]"
                )
                sections.append(f"- {icon} **{a.type}**: {a.description}")
        else:
            sections.append("No anomalies detected.")
        sections.append("")

        # Hypotheses
        sections.append("## Hypotheses")
        sections.append("")
        if self.hypotheses:
            for h in self.hypotheses:
                sections.append(f"### {h.id}: {h.description}")
                sections.append("")
                sections.append(f"- Confidence: {h.confidence:.2f}")
                sections.append(f"- Test type: {h.test_type}")
                sections.append(f"- Expected if true: {h.expected_if_true}")
                sections.append(f"- Expected if false: {h.expected_if_false}")
                sections.append(f"- Supporting anomalies: {', '.join(h.supporting_anomalies)}")
                sections.append("")
        else:
            sections.append("No hypotheses generated.")
            sections.append("")

        # Counterfactual results
        sections.append("## Counterfactual Results")
        sections.append("")
        if self.counterfactual_results:
            for cr in self.counterfactual_results:
                status = "CONFIRMED" if cr.confirmed else "NOT CONFIRMED"
                sections.append(f"### {cr.hypothesis_id} [{status}]")
                sections.append("")
                sections.append(f"- Test type: {cr.test_type}")
                sections.append(f"- Action delta L2: {cr.action_delta_l2:.4f}")
                sections.append(f"- GradCAM shift: {cr.gradcam_shift:.4f}")
                sections.append(f"- Attribution shift per region: {cr.attribution_shift_per_region}")
                sections.append("")
        else:
            reason_msg = {
                "skipped": "Counterfactual tests skipped by user (--skip-counterfactuals).",
                "no_model": "No counterfactual tests run (no model loaded — post-hoc mode).",
                "budget_exhausted": "Counterfactual budget exhausted.",
            }.get(self.cf_skip_reason, "No counterfactual tests run.")
            sections.append(reason_msg)
            sections.append("")

        # Findings
        sections.append("## Findings")
        sections.append("")
        if self.findings:
            for f in self.findings:
                sev = f.severity.upper()
                sections.append(f"### [{sev}] {f.title}")
                sections.append("")
                sections.append(f"**Observation**: {f.observation}")
                sections.append("")
                sections.append(f"**Test**: {f.test_description}")
                sections.append("")
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

        # LLM synthesis
        if self.llm_synthesis:
            sections.append("## LLM Synthesis")
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
    values: list[float]  # action_delta_l2 per run
    delta: float
    pct_change: float | None
    confirmed: list[bool]  # per run


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
    scalar_deltas: dict[str, list[float]] = field(default_factory=dict)
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
                vcells = " | ".join(f"{v:.6f}" for v in vals)
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
                vals = " | ".join(f"{v:.4f}" for v in cd.values)
                sign = "+" if cd.delta > 0 else ""
                pct = f"{cd.pct_change:+.1f}%" if cd.pct_change is not None else "N/A"
                conf = " → ".join("Y" if c else "N" for c in cd.confirmed)
                s.append(f"| {cd.test_type} | {vals} | {sign}{cd.delta:.4f} | {pct} | {conf} |")
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

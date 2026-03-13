"""Compare two or more diagnostic runs and produce a ComparisonReport."""

from __future__ import annotations

import json
import os
import re
from statistics import mean

from .models import (
    AttributionDelta,
    ComparisonReport,
    CounterfactualDelta,
    RunSnapshot,
    WeightAlphaDelta,
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_diagnostic_json(run_dir: str) -> dict | None:
    """Load report.json from a run directory."""
    path = os.path.join(run_dir, "diagnostic", "report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # report.json may be double-encoded as a string
    if isinstance(data, str):
        data = json.loads(data)
    return data


def _load_weightwatcher_json(run_dir: str) -> dict | None:
    """Load weightwatcher.json from a run's model_internals data."""
    path = os.path.join(run_dir, "data", "model_internals", "weightwatcher.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_internals_from_markdown(run_dir: str) -> dict | None:
    """Fallback: parse weight spectral table from model_internals_report.md."""
    path = os.path.join(run_dir, "images", "model_internals_report.md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    # Parse the alpha table rows
    components: dict[str, dict] = {}
    for m in re.finditer(
        r"\|\s*(.+?)\s*\|\s*(\d+|--)\s*\|\s*([\d.]+|N/A)\s*\|"
        r"\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|\s*(.+?)\s*\|",
        text,
    ):
        name = m.group(1).strip()
        mean_alpha = m.group(3).strip()
        status = m.group(6).strip()
        if mean_alpha == "N/A":
            continue
        components[name] = {
            "mean_alpha": float(mean_alpha),
            "status": status,
        }
    return components if components else None


def load_run_snapshot(run_dir: str, label: str) -> RunSnapshot:
    """Load all comparison-relevant data from a run directory."""
    diag = _load_diagnostic_json(run_dir)
    if diag is None:
        raise FileNotFoundError(
            f"No diagnostic report found in {run_dir}/diagnostic/report.json"
        )

    # Try structured JSON first, fall back to markdown parsing
    ww = _load_weightwatcher_json(run_dir)
    internals: dict | None = None
    if ww:
        # Compute per-component summary from raw layer data
        internals = {}
        for component, layers in ww.items():
            alphas = [entry["alpha"] for entry in layers if "alpha" in entry]
            if alphas:
                internals[component] = {
                    "mean_alpha": mean(alphas),
                    "min_alpha": min(alphas),
                    "max_alpha": max(alphas),
                }
    else:
        md_data = _load_internals_from_markdown(run_dir)
        if md_data:
            internals = md_data

    return RunSnapshot(
        label=label,
        run_dir=run_dir,
        diagnostic=diag,
        internals=internals,
    )


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def _pct(base: float, new: float) -> float | None:
    if abs(base) < 1e-12:
        return None
    return ((new - base) / abs(base)) * 100.0


def _compute_attribution_deltas(
    snapshots: list[RunSnapshot],
) -> tuple[list[AttributionDelta], dict[str, list[float]]]:
    """Compare attribution matrices across runs."""
    deltas: list[AttributionDelta] = []
    scalar_deltas: dict[str, list[float]] = {}

    # Collect all signal/region combos
    matrices = [s.diagnostic.get("matrix", {}) for s in snapshots]
    all_signals: list[str] = []
    all_regions: list[str] = []
    for m in matrices:
        for sig in m.get("signal_types", []):
            if sig not in all_signals:
                all_signals.append(sig)
        for reg in m.get("regions", []):
            if reg not in all_regions:
                all_regions.append(reg)

    for sig in all_signals:
        for reg in all_regions:
            values = []
            for m in matrices:
                val = m.get("attribution_mass", {}).get(sig, {}).get(reg, 0.0)
                values.append(val)
            delta = values[-1] - values[0]
            deltas.append(AttributionDelta(
                signal=sig,
                region=reg,
                values=values,
                delta=delta,
                pct_change=_pct(values[0], values[-1]),
            ))

    # Scalars
    all_scalar_keys: list[str] = []
    for m in matrices:
        for k in m.get("scalars", {}):
            if k not in all_scalar_keys:
                all_scalar_keys.append(k)
    for k in all_scalar_keys:
        vals = [m.get("scalars", {}).get(k) for m in matrices]
        scalar_deltas[k] = vals

    return deltas, scalar_deltas


def _compute_counterfactual_deltas(
    snapshots: list[RunSnapshot],
) -> list[CounterfactualDelta]:
    """Compare counterfactual results across runs."""
    # Collect all test types
    all_tests: list[str] = []
    cf_by_run: list[dict[str, dict]] = []
    for snap in snapshots:
        cfs = snap.diagnostic.get("counterfactual_results", [])
        by_type = {cf["test_type"]: cf for cf in cfs}
        cf_by_run.append(by_type)
        for t in by_type:
            if t not in all_tests:
                all_tests.append(t)

    deltas: list[CounterfactualDelta] = []
    for test_type in all_tests:
        values: list[float | None] = []
        confirmed: list[bool | None] = []
        for by_type in cf_by_run:
            cf = by_type.get(test_type)
            if cf:
                values.append(cf.get("action_delta_l2", 0.0))
                confirmed.append(cf.get("confirmed", False))
            else:
                values.append(None)
                confirmed.append(None)
        if any(v is None for v in values):
            delta = None
            pct = None
        else:
            delta = values[-1] - values[0]
            pct = _pct(values[0], values[-1])
        deltas.append(CounterfactualDelta(
            test_type=test_type,
            values=values,
            delta=delta,
            pct_change=pct,
            confirmed=confirmed,
        ))
    return deltas


def _compute_anomaly_summary(
    snapshots: list[RunSnapshot],
) -> dict[str, list[str]]:
    """Map label → list of anomaly types present in that run."""
    summary: dict[str, list[str]] = {}
    for snap in snapshots:
        anomalies = snap.diagnostic.get("anomalies", [])
        summary[snap.label] = [a["type"] for a in anomalies]
    return summary


_ALPHA_STATUS_THRESHOLDS = [
    (2.0, "overcorrelated"),
    (4.0, "healthy"),
    (6.0, "undertrained"),
    (float("inf"), "severely undertrained"),
]


def _alpha_status(alpha: float) -> str:
    for threshold, label in _ALPHA_STATUS_THRESHOLDS:
        if alpha < threshold:
            return label
    return "severely undertrained"


def _compute_weight_alpha_deltas(
    snapshots: list[RunSnapshot],
) -> list[WeightAlphaDelta]:
    """Compare weight spectral alpha across runs."""
    # Collect all component names
    all_components: list[str] = []
    for snap in snapshots:
        if snap.internals:
            for comp in snap.internals:
                if comp not in all_components:
                    all_components.append(comp)

    deltas: list[WeightAlphaDelta] = []
    for comp in all_components:
        values = []
        statuses = []
        for snap in snapshots:
            if snap.internals and comp in snap.internals:
                alpha = snap.internals[comp]["mean_alpha"]
                values.append(alpha)
                # Use status from markdown parse or compute from alpha
                status = snap.internals[comp].get("status") or _alpha_status(alpha)
                statuses.append(status)
            else:
                values.append(0.0)
                statuses.append("N/A")

        if all(v == 0.0 for v in values):
            continue

        delta = values[-1] - values[0]
        deltas.append(WeightAlphaDelta(
            component=comp,
            values=values,
            delta=delta,
            pct_change=_pct(values[0], values[-1]),
            status=statuses,
        ))
    return deltas


def _generate_verdict(report: ComparisonReport) -> str:
    """Generate a short textual verdict summarising the comparison."""
    lines: list[str] = []

    # Check weight alpha improvement
    trainable_improved = []
    for wa in report.weight_alpha_deltas:
        if wa.delta < 0 and wa.values[0] > 4.0:
            trainable_improved.append(wa)

    if trainable_improved:
        comps = ", ".join(f"{wa.component} ({wa.delta:+.2f})" for wa in trainable_improved)
        lines.append(f"Weight structure improved for: {comps}.")

    # Check counterfactual changes
    for cd in report.counterfactual_deltas:
        if cd.delta is None:
            continue
        if cd.test_type == "background_substitution" and cd.delta < 0:
            lines.append(
                f"Background dependence decreased "
                f"({cd.values[0]:.4f} → {cd.values[-1]:.4f}, "
                f"{cd.pct_change:+.1f}%)."
            )
        elif cd.test_type == "object_recolor" and cd.delta > 0:
            lines.append(
                f"Object sensitivity increased "
                f"({cd.values[0]:.4f} → {cd.values[-1]:.4f}, "
                f"{cd.pct_change:+.1f}%)."
            )

    # Check anomaly resolution
    first_label = report.labels[0]
    last_label = report.labels[-1]
    first_anomalies = set(report.anomaly_summary.get(first_label, []))
    last_anomalies = set(report.anomaly_summary.get(last_label, []))
    resolved = first_anomalies - last_anomalies
    new = last_anomalies - first_anomalies
    if resolved:
        lines.append(f"Resolved anomalies: {', '.join(resolved)}.")
    if new:
        lines.append(f"New anomalies: {', '.join(new)}.")
    if first_anomalies == last_anomalies and first_anomalies:
        lines.append(f"Same anomalies persist across all runs: {', '.join(first_anomalies)}.")

    # Attribution changes for task objects (non-background, non-gripper)
    object_improvements = []
    object_regressions = []
    for ad in report.attribution_deltas:
        if ad.region == "background" or "gripper" in ad.region:
            continue
        if ad.signal in ("gradcam_siglip", "cross_attention", "saliency"):
            if ad.delta > 0.001:
                object_improvements.append(ad)
            elif ad.delta < -0.001:
                object_regressions.append(ad)

    if object_improvements:
        items = [f"{ad.signal}/{ad.region} ({ad.pct_change:+.1f}%)"
                 for ad in object_improvements if ad.pct_change is not None]
        if items:
            lines.append(f"Object attribution improved: {', '.join(items)}.")

    if not lines:
        lines.append("No significant differences detected between runs.")

    return "\n".join(lines)


def _generate_recommendations(
    report: ComparisonReport,
    snapshots: list[RunSnapshot],
) -> list[str]:
    """Generate actionable recommendations based on the comparison."""
    recs: list[str] = []
    first = snapshots[0]
    last = snapshots[-1]

    # --- Training progress ---------------------------------------------------
    expert_delta = None
    for wa in report.weight_alpha_deltas:
        if "expert" in wa.component.lower() and "trainable" in wa.component.lower():
            expert_delta = wa
            break

    if expert_delta:
        last_alpha = expert_delta.values[-1]
        if last_alpha > 6.0 and expert_delta.delta < 0:
            recs.append(
                f"**Continue training** — Expert alpha improved "
                f"({expert_delta.values[0]:.1f} → {last_alpha:.1f}) but is still "
                f"severely undertrained (healthy range: 2-4). The loss is decreasing "
                f"and weight structure is converging; more epochs should help."
            )
        elif last_alpha > 6.0 and expert_delta.delta >= 0:
            recs.append(
                f"**Adjust learning rate** — Expert alpha did not improve "
                f"({expert_delta.values[0]:.1f} → {last_alpha:.1f}). Consider "
                f"increasing the learning rate or number of training steps."
            )
        elif 4.0 < last_alpha <= 6.0:
            recs.append(
                f"**Almost there** — Expert alpha is {last_alpha:.1f} (undertrained, "
                f"target: 2-4). A few more epochs of training should bring it into "
                f"the healthy range."
            )

    # --- Background dependence -----------------------------------------------
    bg_cf = None
    for cd in report.counterfactual_deltas:
        if cd.test_type == "background_substitution":
            bg_cf = cd
            break

    # Get the latest background attribution across gradient signals
    bg_shares: list[float] = []
    for ad in report.attribution_deltas:
        if ad.region == "background" and ad.signal in ("gradcam_siglip", "saliency"):
            bg_shares.append(ad.values[-1])

    if bg_shares and max(bg_shares) > 0.70:
        if bg_cf and bg_cf.delta < -0.05:
            recs.append(
                f"**Background reliance decreasing but still high** — "
                f"Background still receives {max(bg_shares):.0%} of attribution. "
                f"Consider data augmentation (random background crops, color jitter, "
                f"background replacement) to accelerate the shift toward object features."
            )
        elif bg_cf and bg_cf.delta >= -0.05:
            recs.append(
                f"**High background reliance persists** — "
                f"Background receives {max(bg_shares):.0%} of attribution with minimal "
                f"improvement. Strongly recommend background augmentation during training "
                f"(random crops, Gaussian blur on non-object regions, or synthetic background "
                f"substitution) to force the model to learn object features."
            )

    # --- Object grounding ----------------------------------------------------
    obj_cf = None
    for cd in report.counterfactual_deltas:
        if cd.test_type == "object_recolor":
            obj_cf = cd
            break

    # Find task object attribution (non-gripper, non-background)
    low_objects: list[tuple[str, float]] = []
    for ad in report.attribution_deltas:
        if ad.region == "background" or "gripper" in ad.region:
            continue
        if ad.signal == "gradcam_siglip":
            if ad.values[-1] < 0.05:
                low_objects.append((ad.region, ad.values[-1]))

    if low_objects:
        obj_names = ", ".join(f"{name} ({val:.1%})" for name, val in low_objects)
        if obj_cf and obj_cf.delta > 0.05:
            recs.append(
                f"**Object grounding improving but still weak** — "
                f"Task objects have low attribution: {obj_names}. "
                f"Object sensitivity is increasing ({obj_cf.pct_change:+.0f}%), "
                f"which suggests the model is starting to attend to them. "
                f"More training data with varied object positions and appearances will help."
            )
        else:
            recs.append(
                f"**Weak object grounding** — Task objects have very low attribution: "
                f"{obj_names}. Consider adding training data with more varied object "
                f"positions, orientations, and lighting conditions."
            )

    # --- Dataset diversity ---------------------------------------------------
    last_diag = last.diagnostic
    dd = last_diag.get("dataset_diversity")
    if dd:
        bg_div = dd.get("background_diversity_score", 1.0)
        task_count = dd.get("task_string_diversity", {}).get("unique_count", 0)

        if bg_div < 0.1:
            recs.append(
                f"**Increase background diversity** — Background diversity score is "
                f"only {bg_div:.3f}. Collecting demonstrations in varied environments "
                f"(different tables, lighting, backgrounds) will reduce the model's "
                f"tendency to memorize scene-specific features."
            )

        if task_count == 1:
            recs.append(
                f"**Add more task variations** — Only 1 unique task string in the dataset. "
                f"Adding demonstrations for related tasks (different objects, target "
                f"containers) will improve the model's language grounding and "
                f"generalization."
            )

        # Check object position variance
        obj_stats = dd.get("object_position_stats", {})
        for obj_name, stats in obj_stats.items():
            if "gripper" in obj_name.lower():
                continue
            std_x = stats.get("std_x", 999)
            std_y = stats.get("std_y", 999)
            if std_x < 10 and std_y < 10:
                recs.append(
                    f"**Vary {obj_name} position** — Position standard deviation is "
                    f"only ({std_x:.1f}, {std_y:.1f}) pixels. The model may memorize "
                    f"this fixed position rather than learning to locate it visually. "
                    f"Collect demos with the {obj_name} in different locations."
                )

    # --- Anomaly resolution --------------------------------------------------
    first_anomalies = set(report.anomaly_summary.get(report.labels[0], []))
    last_anomalies = set(report.anomaly_summary.get(report.labels[-1], []))
    new_anomalies = last_anomalies - first_anomalies
    if new_anomalies:
        recs.append(
            f"**Investigate new anomalies** — The following anomalies appeared after "
            f"fine-tuning: {', '.join(new_anomalies)}. Check for overfitting or "
            f"training instability."
        )

    if not recs:
        recs.append(
            "No specific recommendations — the model appears to be progressing well. "
            "Continue monitoring diagnostics as training advances."
        )

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_runs(
    run_dirs: list[str],
    labels: list[str] | None = None,
) -> ComparisonReport:
    """Compare two or more diagnostic runs.

    Args:
        run_dirs: Paths to run directories (each must have diagnostic/report.json).
        labels: Human-readable labels for each run. Defaults to directory basenames.

    Returns:
        A :class:`ComparisonReport` ready for serialisation.
    """
    if len(run_dirs) < 2:
        raise ValueError("Need at least 2 run directories to compare.")

    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError(f"Got {len(labels)} labels but {len(run_dirs)} run dirs.")

    # Load snapshots
    snapshots = [load_run_snapshot(rd, lbl) for rd, lbl in zip(run_dirs, labels)]

    # Compute deltas
    attr_deltas, scalar_deltas = _compute_attribution_deltas(snapshots)
    cf_deltas = _compute_counterfactual_deltas(snapshots)
    anomaly_summary = _compute_anomaly_summary(snapshots)
    weight_deltas = _compute_weight_alpha_deltas(snapshots)

    report = ComparisonReport(
        labels=labels,
        run_dirs=run_dirs,
        attribution_deltas=attr_deltas,
        scalar_deltas=scalar_deltas,
        counterfactual_deltas=cf_deltas,
        anomaly_summary=anomaly_summary,
        weight_alpha_deltas=weight_deltas,
    )

    report.verdict = _generate_verdict(report)
    report.recommendations = _generate_recommendations(report, snapshots)

    return report

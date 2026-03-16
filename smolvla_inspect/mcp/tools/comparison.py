"""Comparison tools (5 tools)."""

from __future__ import annotations

import dataclasses
import json

from ..server import mcp
from ..context import resolve_run, get_base_dir
from ..helpers import success_response, error_response, _json_default


@mcp.tool()
def compare_diagnostic_runs(run_ids: list[str], labels: list[str] | None = None) -> str:
    """Compare diagnostic reports across two or more runs.

    Args:
        run_ids: List of run identifiers to compare (minimum 2).
        labels: Optional human-readable labels for each run.
    """
    try:
        if len(run_ids) < 2:
            return error_response("Need at least 2 run IDs to compare.")

        run_dirs = []
        for rid in run_ids:
            run_dir, _ = resolve_run(rid)
            run_dirs.append(str(run_dir))

        # Lazy import to avoid pulling torch at startup
        from smolvla_inspect.diagnostic.comparison import compare_runs

        report = compare_runs(run_dirs, labels=labels)
        # Convert dataclass to dict
        report_dict = dataclasses.asdict(report)
        return json.dumps({"success": True, "data": report_dict}, default=_json_default)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Comparison failed: {e}")


@mcp.tool()
def compare_run_configs(run_ids: list[str]) -> str:
    """Compare configuration differences between runs.

    Args:
        run_ids: List of run identifiers to compare.
    """
    try:
        if len(run_ids) < 2:
            return error_response("Need at least 2 run IDs to compare.")

        manifests = []
        for rid in run_ids:
            _, manifest = resolve_run(rid)
            manifests.append(manifest)

        from web.backend.services.comparison import compute_config_diff

        diffs = compute_config_diff(manifests)
        return success_response({
            "run_ids": run_ids,
            "config_diffs": diffs,
            "num_differing_fields": len(diffs),
        })
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Config comparison failed: {e}")


@mcp.tool()
def compare_component_weights(run_ids: list[str]) -> str:
    """Compare model weight statistics (WeightWatcher) between runs.

    Args:
        run_ids: List of run identifiers to compare.
    """
    try:
        if len(run_ids) < 2:
            return error_response("Need at least 2 run IDs to compare.")

        ww_data = []
        for rid in run_ids:
            run_dir, _ = resolve_run(rid)
            ww_path = run_dir / "data" / "model_internals" / "weightwatcher.json"
            if not ww_path.exists():
                return error_response(
                    f"No weightwatcher data for run {rid}.",
                    suggestion="Re-run with --model-health.",
                )
            with open(ww_path) as f:
                ww_data.append(json.load(f))

        # Compute deltas between first two runs
        deltas = {}
        keys_a = set(ww_data[0].keys()) if isinstance(ww_data[0], dict) else set()
        keys_b = set(ww_data[1].keys()) if isinstance(ww_data[1], dict) else set()
        for key in keys_a & keys_b:
            a_val = ww_data[0][key]
            b_val = ww_data[1][key]
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                deltas[key] = {
                    "run_0": a_val,
                    "run_1": b_val,
                    "delta": round(b_val - a_val, 6),
                }

        return success_response({
            "run_ids": run_ids,
            "per_run": ww_data,
            "deltas": deltas,
        })
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Weight comparison failed: {e}")


@mcp.tool()
def compare_heatmaps(
    run_ids: list[str],
    viz_types: list[str] | None = None,
    frame_indices: list[int] | None = None,
) -> str:
    """Compare heatmap visualizations across runs.

    Args:
        run_ids: List of run identifiers to compare.
        viz_types: Visualization types to compare (default: ["self_attention"]).
        frame_indices: Optional frame indices to include.
    """
    try:
        if len(run_ids) < 2:
            return error_response("Need at least 2 run IDs to compare.")

        viz_types = viz_types or ["self_attention"]
        run_dirs = []
        manifests = []
        for rid in run_ids:
            rd, mf = resolve_run(rid)
            run_dirs.append(rd)
            manifests.append(mf)

        from web.backend.services.comparison import compare_runs

        result = compare_runs(run_dirs, manifests, viz_types, frame_indices)
        return json.dumps({"success": True, "data": result}, default=_json_default)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Heatmap comparison failed: {e}")


@mcp.tool()
def compare_matrix(
    metric: str,
    model_ids: list[str] | None = None,
    dataset_ids: list[str] | None = None,
) -> str:
    """Build a comparison matrix of a diagnostic metric across model/dataset combinations.

    For each (model_id, dataset_id) pair, finds the most recent run and extracts
    the requested metric from its diagnostic report.

    Args:
        metric: Metric key path in the diagnostic report (e.g., "findings.0.severity",
                "symptoms.0.type", or a dot-separated path).
        model_ids: List of model IDs to include. If omitted, uses all found models.
        dataset_ids: List of dataset IDs to include. If omitted, uses all found datasets.
    """
    try:
        from web.backend.services.run_scanner import scan_directory
        from web.backend.services.diagnostic_service import load_diagnostic_report

        runs = scan_directory(get_base_dir())

        # Group runs by (model_id, dataset_id), keep most recent
        grouped: dict[tuple, list] = {}
        for r in runs:
            if not r.model_id or not r.dataset_id:
                continue
            if model_ids and r.model_id not in model_ids:
                continue
            if dataset_ids and r.dataset_id not in dataset_ids:
                continue
            key = (r.model_id, r.dataset_id)
            grouped.setdefault(key, []).append(r)

        # For each group, pick most recent (by created_at or first found)
        matrix: dict = {}
        found_models = set()
        found_datasets = set()

        for (mid, did), group_runs in grouped.items():
            # Sort by created_at descending
            group_runs.sort(key=lambda x: x.created_at or "", reverse=True)
            best = group_runs[0]

            try:
                run_dir, _ = resolve_run(best.id)
                report = load_diagnostic_report(run_dir)
                if not report:
                    continue

                # Extract metric by dot-path
                value = report
                for part in metric.split("."):
                    if isinstance(value, dict):
                        value = value.get(part)
                    elif isinstance(value, list):
                        try:
                            value = value[int(part)]
                        except (IndexError, ValueError):
                            value = None
                    else:
                        value = None
                    if value is None:
                        break

                matrix.setdefault(mid, {})[did] = value
                found_models.add(mid)
                found_datasets.add(did)
            except (ValueError, Exception):
                continue

        return success_response({
            "metric": metric,
            "models": sorted(found_models),
            "datasets": sorted(found_datasets),
            "matrix": matrix,
        })
    except Exception as e:
        return error_response(f"Matrix comparison failed: {e}")

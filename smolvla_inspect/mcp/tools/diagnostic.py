"""Diagnostic report read tools (11 tools)."""

from __future__ import annotations

from ..server import mcp
from ..context import resolve_run
from ..helpers import success_response, error_response, image_to_base64


def _load_report(run_id: str) -> tuple:
    """Helper: resolve run and load diagnostic report. Returns (run_dir, report)."""
    from web.backend.services.diagnostic_service import load_diagnostic_report

    run_dir, manifest = resolve_run(run_id)
    report = load_diagnostic_report(run_dir)
    if not report:
        raise ValueError(
            f"No diagnostic report found for run {run_id}. "
            "Run `smolvla-inspect diagnose` to generate one."
        )
    return run_dir, report


def _get_report_key(report: dict, key: str, legacy_key: str | None = None) -> list | dict | None:
    """Extract a key from a diagnostic report, handling legacy renames."""
    value = report.get(key)
    if value is None and legacy_key:
        value = report.get(legacy_key)
    return value


@mcp.tool()
def get_diagnostic_report(run_id: str) -> str:
    """Get the full diagnostic report for a run.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        return success_response(report)
    except ValueError as e:
        return error_response(str(e), suggestion="Run diagnose first, or use list_runs with has_diagnostic=true.")
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_symptoms(run_id: str) -> str:
    """Get detected symptoms from the diagnostic report.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        symptoms = _get_report_key(report, "symptoms", legacy_key="anomalies")
        if symptoms is None:
            return error_response("No symptoms section in diagnostic report.")
        return success_response(symptoms)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_hypotheses(run_id: str) -> str:
    """Get generated hypotheses from the diagnostic report.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        hypotheses = report.get("hypotheses")
        if hypotheses is None:
            return error_response("No hypotheses section in diagnostic report.")
        return success_response(hypotheses)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_findings(run_id: str) -> str:
    """Get diagnostic findings (conclusions) from the diagnostic report.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        findings = report.get("findings")
        if findings is None:
            return error_response("No findings section in diagnostic report.")
        return success_response(findings)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_evidence_chain(run_id: str) -> str:
    """Get the evidence chain linking symptoms to hypotheses to findings.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        chain = report.get("evidence_chain")
        if chain is None:
            # Build a lightweight chain from available data
            result = {
                "symptoms": _get_report_key(report, "symptoms", "anomalies") or [],
                "hypotheses": report.get("hypotheses", []),
                "findings": report.get("findings", []),
                "counterfactual_results": report.get("counterfactual_results", {}),
            }
            return success_response(result)
        return success_response(chain)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_diagnostic_matrix(run_id: str) -> str:
    """Get the diagnostic matrix (attribution scores across components and frames).

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services.diagnostic_service import load_diagnostic_matrix

        run_dir, _ = resolve_run(run_id)
        matrix = load_diagnostic_matrix(run_dir)
        if not matrix:
            return error_response("No diagnostic matrix found.")
        return success_response(matrix)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_scene_data(run_id: str) -> str:
    """Get scene understanding data (object detections, annotations).

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services.diagnostic_service import load_scene_data

        run_dir, _ = resolve_run(run_id)
        data = load_scene_data(run_dir)
        if not data:
            return error_response("No scene data found.")

        # Include annotated frame as base64 if available
        if data.get("has_annotated_frame"):
            ann_path = run_dir / "diagnostic" / "scene" / "annotated_frame.png"
            if ann_path.exists():
                data["annotated_frame_base64"] = image_to_base64(ann_path)

        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_counterfactual_result(run_id: str, test_name: str) -> str:
    """Get a specific counterfactual test result.

    Args:
        run_id: The run identifier.
        test_name: Name of the counterfactual test (e.g., "background_substitution").
    """
    try:
        from web.backend.services.diagnostic_service import load_counterfactual_result

        run_dir, _ = resolve_run(run_id)
        result = load_counterfactual_result(run_dir, test_name)
        if not result:
            return error_response(
                f"No counterfactual result for test '{test_name}'.",
                suggestion="Check the diagnostic report for available test names.",
            )

        # Include comparison image if available
        if result.get("has_comparison_image"):
            comp_path = run_dir / "diagnostic" / "counterfactuals" / test_name / "comparison.png"
            if comp_path.exists():
                result["comparison_image_base64"] = image_to_base64(comp_path)

        return success_response(result)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_semantic_probe(run_id: str) -> str:
    """Get semantic probe results from the diagnostic report.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        probe = report.get("semantic_probe")
        if probe is None:
            return error_response("No semantic probe data in diagnostic report.")
        return success_response(probe)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_qk_probe(run_id: str) -> str:
    """Get QK (query-key) probe results from the diagnostic report.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        probe = report.get("qk_probe")
        if probe is None:
            return error_response("No QK probe data in diagnostic report.")
        return success_response(probe)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_spatial_object_diagnosis(run_id: str) -> str:
    """Get spatial object-level diagnosis from the diagnostic report.

    Args:
        run_id: The run identifier.
    """
    try:
        _, report = _load_report(run_id)
        spatial = report.get("spatial_object_diagnosis")
        if spatial is None:
            return error_response("No spatial object diagnosis in diagnostic report.")
        return success_response(spatial)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")

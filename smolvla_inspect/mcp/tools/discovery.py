"""Run discovery and management tools (8 tools)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..server import mcp
from ..context import get_base_dir, resolve_run
from ..helpers import success_response, error_response


@mcp.tool()
def list_runs(
    model_id: str | None = None,
    dataset_id: str | None = None,
    tag: str | None = None,
    has_diagnostic: bool | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> str:
    """List all inspection runs, with optional filters.

    Args:
        model_id: Filter by model ID (substring match).
        dataset_id: Filter by dataset ID (substring match).
        tag: Filter by tag (exact match).
        has_diagnostic: If true, only runs with diagnostic reports.
        created_after: ISO date string — only runs created after this date.
        created_before: ISO date string — only runs created before this date.
    """
    try:
        from web.backend.services.run_scanner import scan_directory
        from web.backend.services.diagnostic_service import diagnostic_exists

        runs = scan_directory(get_base_dir())
        filtered = []
        for r in runs:
            if model_id and (not r.model_id or model_id.lower() not in r.model_id.lower()):
                continue
            if dataset_id and (not r.dataset_id or dataset_id.lower() not in r.dataset_id.lower()):
                continue
            if created_after and r.created_at:
                try:
                    if r.created_at < created_after:
                        continue
                except (TypeError, ValueError):
                    pass
            if created_before and r.created_at:
                try:
                    if r.created_at > created_before:
                        continue
                except (TypeError, ValueError):
                    pass

            run_dict = r.model_dump()

            # Check tags and diagnostic existence (requires resolving run dir)
            try:
                run_dir, manifest = resolve_run(r.id)
                if has_diagnostic is not None:
                    has_diag = diagnostic_exists(run_dir)
                    if has_diagnostic != has_diag:
                        continue
                run_dict["has_diagnostic"] = diagnostic_exists(run_dir)

                # Check tags
                tags = manifest.get("tags", [])
                notes_path = run_dir / "run_notes.json"
                if notes_path.exists():
                    with open(notes_path) as f:
                        notes_data = json.load(f)
                    tags = notes_data.get("tags", tags)
                run_dict["tags"] = tags
                if tag and tag not in tags:
                    continue
            except (ValueError, Exception):
                if has_diagnostic is True or tag:
                    continue
                run_dict["has_diagnostic"] = False
                run_dict["tags"] = []

            filtered.append(run_dict)

        return success_response(filtered)
    except Exception as e:
        return error_response(f"Failed to list runs: {e}")


@mcp.tool()
def search_runs(query: str) -> str:
    """Search runs by name, model ID, dataset ID, or task string (case-insensitive substring match).

    Args:
        query: Search string to match against run metadata.
    """
    try:
        from web.backend.services.run_scanner import scan_directory

        runs = scan_directory(get_base_dir())
        q = query.lower()
        matches = []
        for r in runs:
            searchable = " ".join(filter(None, [
                r.name, r.model_id, r.dataset_id,
            ])).lower()
            # Also check manifest for task_string
            try:
                run_dir, manifest = resolve_run(r.id)
                task_str = (manifest.get("dataset_info") or {}).get("task_string", "")
                searchable += " " + task_str.lower()
            except (ValueError, Exception):
                pass
            if q in searchable:
                matches.append(r.model_dump())

        return success_response(matches)
    except Exception as e:
        return error_response(f"Search failed: {e}")


@mcp.tool()
def get_run_detail(run_id: str) -> str:
    """Get full details for a specific run including manifest and metadata.

    Args:
        run_id: The run identifier (from list_runs).
    """
    try:
        from web.backend.services.diagnostic_service import diagnostic_exists

        run_dir, manifest = resolve_run(run_id)
        result = {
            "id": run_id,
            "name": run_dir.name,
            "path": str(run_dir),
            "manifest": manifest,
            "has_diagnostic": diagnostic_exists(run_dir),
        }
        # Include notes if present
        notes_path = run_dir / "run_notes.json"
        if notes_path.exists():
            with open(notes_path) as f:
                result["notes"] = json.load(f)

        return success_response(result)
    except ValueError as e:
        return error_response(str(e), suggestion="Use list_runs to see available run IDs.")
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def tag_run(run_id: str, tag: str) -> str:
    """Add a tag to a run.

    Args:
        run_id: The run identifier.
        tag: Tag string to add.
    """
    try:
        run_dir, manifest = resolve_run(run_id)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            return error_response("Cannot tag legacy runs without a manifest file.")

        tags = manifest.get("tags", [])
        if tag not in tags:
            tags.append(tag)
            manifest["tags"] = tags
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        return success_response({"run_id": run_id, "tags": tags})
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def untag_run(run_id: str, tag: str) -> str:
    """Remove a tag from a run.

    Args:
        run_id: The run identifier.
        tag: Tag string to remove.
    """
    try:
        run_dir, manifest = resolve_run(run_id)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            return error_response("Cannot untag legacy runs without a manifest file.")

        tags = manifest.get("tags", [])
        if tag in tags:
            tags.remove(tag)
            manifest["tags"] = tags
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        return success_response({"run_id": run_id, "tags": tags})
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def rescan_runs() -> str:
    """Re-scan the base directory for runs and return the updated count."""
    try:
        from web.backend.services.run_scanner import scan_directory

        runs = scan_directory(get_base_dir())
        return success_response({"count": len(runs), "run_ids": [r.id for r in runs]})
    except Exception as e:
        return error_response(f"Rescan failed: {e}")


@mcp.tool()
def validate_run(run_id: str) -> str:
    """Check the integrity of a run — verify expected files and directories exist.

    Args:
        run_id: The run identifier.
    """
    try:
        run_dir, manifest = resolve_run(run_id)
        issues: list[str] = []
        checks: list[str] = []

        # Check manifest
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            checks.append("run_manifest.json: OK")
        else:
            issues.append("run_manifest.json: MISSING")

        # Check data directory
        data_dir = run_dir / "data"
        if data_dir.is_dir():
            checks.append("data/: OK")
        else:
            issues.append("data/: MISSING")

        # Check images directory
        images_dir = run_dir / "images"
        if images_dir.is_dir():
            checks.append("images/: OK")
        else:
            issues.append("images/: MISSING (may be legacy run)")

        # Check available visualizations
        avail = manifest.get("available_visualizations", {})
        for viz_name, expected in avail.items():
            if not expected:
                continue
            # Spot-check a few data files
            if viz_name == "self_attention":
                path = data_dir / "self_attention" / "heatmaps.npz"
                if path.exists():
                    checks.append(f"{viz_name}: OK")
                else:
                    issues.append(f"{viz_name}: declared but data file missing")
            elif viz_name == "cross_attention":
                path = data_dir / "cross_attention" / "heatmaps.npz"
                if path.exists():
                    checks.append(f"{viz_name}: OK")
                else:
                    issues.append(f"{viz_name}: declared but data file missing")

        # Check diagnostic
        diag_dir = run_dir / "diagnostic"
        if diag_dir.is_dir():
            report_path = diag_dir / "report.json"
            if report_path.exists():
                checks.append("diagnostic/report.json: OK")
            else:
                issues.append("diagnostic/: directory exists but report.json missing")
        else:
            checks.append("diagnostic/: not present (run diagnose to generate)")

        return success_response({
            "run_id": run_id,
            "valid": len(issues) == 0,
            "checks": checks,
            "issues": issues,
        })
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def add_run(path: str) -> str:
    """Register an external run directory by path.

    Args:
        path: Absolute or relative path to a run directory containing run_manifest.json.
    """
    try:
        run_path = Path(path).resolve()
        if not run_path.is_dir():
            return error_response(f"Directory not found: {path}")

        manifest_path = run_path / "run_manifest.json"
        if not manifest_path.exists():
            return error_response(
                f"No run_manifest.json in {path}",
                suggestion="Ensure this is a valid smolvla-inspect output directory.",
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        from web.backend.services.run_scanner import _run_id_from_path

        run_id = _run_id_from_path(run_path)
        return success_response({
            "run_id": run_id,
            "name": run_path.name,
            "path": str(run_path),
            "model_id": (manifest.get("model_info") or {}).get("model_id"),
            "dataset_id": (manifest.get("dataset_info") or {}).get("dataset_id"),
        })
    except Exception as e:
        return error_response(f"Failed to add run: {e}")

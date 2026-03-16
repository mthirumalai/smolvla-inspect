"""MCP resources — static configs, registry data, and per-run data."""

from __future__ import annotations

from .server import mcp
from .context import get_base_dir, get_project_root, resolve_run


# -- Static config resources --

@mcp.resource("smolvla://config/defaults")
def config_defaults() -> str:
    """Default CLI configuration (configs/defaults.yaml)."""
    path = get_project_root() / "configs" / "defaults.yaml"
    if path.exists():
        return path.read_text()
    return "(defaults.yaml not found)"


@mcp.resource("smolvla://config/diagnostic")
def config_diagnostic() -> str:
    """Diagnostic pipeline configuration (configs/diagnostic.yaml)."""
    path = get_project_root() / "configs" / "diagnostic.yaml"
    if path.exists():
        return path.read_text()
    return "(diagnostic.yaml not found)"


# -- Registry resources (lazy imports to avoid torch) --

@mcp.resource("smolvla://registry/primitives")
def registry_primitives() -> str:
    """All registered diagnostic primitives."""
    import json
    from smolvla_inspect.diagnostic.registry import list_primitives

    specs = list_primitives()
    return json.dumps([{
        "name": s.name, "category": s.category, "cost": s.cost,
        "requires_gpu": s.requires_gpu, "description": s.description,
    } for s in specs], indent=2)


@mcp.resource("smolvla://registry/signals")
def registry_signals() -> str:
    """All registered expensive signal runners."""
    import json
    from smolvla_inspect.diagnostic.registry import SIGNAL_REGISTRY

    return json.dumps([{
        "name": name, "cost_description": spec.cost_description,
        "prompt_description": spec.prompt_description,
    } for name, spec in sorted(SIGNAL_REGISTRY.items())], indent=2)


@mcp.resource("smolvla://registry/hypothesis-templates")
def registry_hypothesis_templates() -> str:
    """All registered hypothesis templates."""
    import json
    from smolvla_inspect.diagnostic.registry import HYPOTHESIS_TEMPLATES

    return json.dumps([{
        "symptom_type": name,
        "description_template": t.description_template,
        "confidence": t.confidence,
        "test_type": t.test_type,
    } for name, t in sorted(HYPOTHESIS_TEMPLATES.items())], indent=2)


# -- Dynamic per-run resources --

@mcp.resource("smolvla://run/{run_id}/manifest")
def run_manifest(run_id: str) -> str:
    """Run manifest JSON."""
    import json
    _, manifest = resolve_run(run_id)
    return json.dumps(manifest, indent=2)


@mcp.resource("smolvla://run/{run_id}/diagnostic-report")
def run_diagnostic_report(run_id: str) -> str:
    """Diagnostic report JSON for a run."""
    import json
    from web.backend.services.diagnostic_service import load_diagnostic_report

    run_dir, _ = resolve_run(run_id)
    report = load_diagnostic_report(run_dir)
    if not report:
        return '{"error": "No diagnostic report found"}'
    return json.dumps(report, indent=2)


@mcp.resource("smolvla://run/{run_id}/notes")
def run_notes(run_id: str) -> str:
    """Run notes JSON."""
    import json

    run_dir, _ = resolve_run(run_id)
    notes_path = run_dir / "run_notes.json"
    if not notes_path.exists():
        return '{"notes": "", "updated_at": null}'
    return notes_path.read_text()

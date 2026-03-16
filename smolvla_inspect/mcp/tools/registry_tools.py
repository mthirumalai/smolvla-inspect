"""Registry tools — list diagnostic primitives, signals, and templates (7 tools)."""

from __future__ import annotations

from ..server import mcp
from ..helpers import success_response, error_response


@mcp.tool()
def list_primitives(category: str | None = None) -> str:
    """List all registered diagnostic primitives.

    Args:
        category: Optional filter by category ("scene", "model", "composite",
                  "counterfactual", "dataset").
    """
    try:
        from smolvla_inspect.diagnostic.registry import list_primitives as _list

        specs = _list(category)
        return success_response([{
            "name": s.name,
            "category": s.category,
            "cost": s.cost,
            "requires_gpu": s.requires_gpu,
            "requires_model": s.requires_model,
            "description": s.description,
            "param_schema": s.param_schema,
        } for s in specs])
    except Exception as e:
        return error_response(f"Failed to list primitives: {e}")


@mcp.tool()
def list_signals() -> str:
    """List all registered expensive signal runners."""
    try:
        from smolvla_inspect.diagnostic.registry import SIGNAL_REGISTRY

        return success_response([{
            "name": name,
            "cost_description": spec.cost_description,
            "prompt_description": spec.prompt_description,
            "requires_model": spec.requires_model,
        } for name, spec in sorted(SIGNAL_REGISTRY.items())])
    except Exception as e:
        return error_response(f"Failed to list signals: {e}")


@mcp.tool()
def list_hypothesis_templates() -> str:
    """List all registered hypothesis templates (symptom → hypothesis mapping)."""
    try:
        from smolvla_inspect.diagnostic.registry import HYPOTHESIS_TEMPLATES

        return success_response([{
            "symptom_type": name,
            "description_template": t.description_template,
            "confidence": t.confidence,
            "test_type": t.test_type,
            "expected_if_true": t.expected_if_true,
            "expected_if_false": t.expected_if_false,
        } for name, t in sorted(HYPOTHESIS_TEMPLATES.items())])
    except Exception as e:
        return error_response(f"Failed to list hypothesis templates: {e}")


@mcp.tool()
def list_symptom_detectors() -> str:
    """List primitives that detect symptoms (category 'scene' or 'model' with detection role)."""
    try:
        from smolvla_inspect.diagnostic.registry import list_primitives as _list

        # Return all primitives that act as detectors (scene + model categories)
        specs = _list("scene") + _list("model")
        return success_response([{
            "name": s.name,
            "category": s.category,
            "cost": s.cost,
            "description": s.description,
        } for s in specs])
    except Exception as e:
        return error_response(f"Failed to list symptom detectors: {e}")


@mcp.tool()
def get_primitive_detail(name: str) -> str:
    """Get full details for a specific diagnostic primitive.

    Args:
        name: Primitive name (e.g., "counterfactual.background_substitution").
    """
    try:
        from smolvla_inspect.diagnostic.registry import get_primitive

        spec = get_primitive(name)
        return success_response({
            "name": spec.name,
            "category": spec.category,
            "cost": spec.cost,
            "requires_gpu": spec.requires_gpu,
            "requires_model": spec.requires_model,
            "requires_scene_models": spec.requires_scene_models,
            "description": spec.description,
            "prompt_description": spec.prompt_description,
            "param_schema": spec.param_schema,
        })
    except KeyError:
        return error_response(
            f"Primitive '{name}' not found.",
            suggestion="Use list_primitives to see all available primitives.",
        )
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_counterfactual_tests() -> str:
    """List available counterfactual test types from the registry."""
    try:
        from smolvla_inspect.diagnostic.registry import format_counterfactual_prompt_section

        section = format_counterfactual_prompt_section()
        return success_response({"available_tests": section})
    except Exception as e:
        return error_response(f"Failed to list counterfactual tests: {e}")


@mcp.tool()
def get_signal_descriptions() -> str:
    """Get formatted descriptions of all expensive signal runners (for prompt injection)."""
    try:
        from smolvla_inspect.diagnostic.registry import format_signals_prompt_section

        section = format_signals_prompt_section()
        return success_response({"signal_descriptions": section})
    except Exception as e:
        return error_response(f"Failed to get signal descriptions: {e}")

"""Statistics tools (3 tools)."""

from __future__ import annotations

from ..server import mcp
from ..context import resolve_run
from ..helpers import success_response, error_response


@mcp.tool()
def get_viz_stats(
    run_id: str,
    viz_type: str,
    frame_indices: list[int] | None = None,
) -> str:
    """Compute numerical statistics for a specific visualization type.

    Args:
        run_id: The run identifier.
        viz_type: Visualization type — one of "self_attention", "cross_attention",
                  "saliency", "gradcam_siglip", "gradcam_connector", "per_head",
                  "vision_vs_state", "per_action_dim".
        frame_indices: Optional list of frame indices to include.
    """
    try:
        from web.backend.services import data_loader, stats_engine

        run_dir, _ = resolve_run(run_id)

        if viz_type in ("self_attention", "cross_attention", "saliency",
                        "gradcam_siglip", "gradcam_connector"):
            if viz_type == "self_attention":
                heatmaps = data_loader.load_self_attention_heatmaps(run_dir)
            elif viz_type == "cross_attention":
                heatmaps = data_loader.load_cross_attention_heatmaps(run_dir)
            else:
                heatmaps = data_loader.load_gradient_data(run_dir, viz_type)

            if not heatmaps:
                return error_response(f"No {viz_type} data found.")
            if frame_indices:
                idx_set = set(frame_indices)
                heatmaps = [h for i, h in enumerate(heatmaps) if i in idx_set]
            return success_response(stats_engine.compute_heatmap_stats(heatmaps))

        elif viz_type == "per_head":
            data = data_loader.load_per_head_data(run_dir)
            if not data:
                return error_response("No per-head data found.")
            return success_response(
                stats_engine.compute_per_head_stats(data.get("heads", []), data.get("entropies"))
            )

        elif viz_type == "vision_vs_state":
            data = data_loader.load_vision_vs_state(run_dir)
            if not data:
                return error_response("No vision vs state data found.")
            return success_response(stats_engine.compute_vision_vs_state_stats(data))

        elif viz_type == "per_action_dim":
            data = data_loader.load_per_action_dim(run_dir)
            if not data:
                return error_response("No per-action-dimension data found.")
            return success_response(stats_engine.compute_per_action_dim_stats(data))

        else:
            return error_response(
                f"Unknown viz_type: {viz_type}",
                suggestion="Valid types: self_attention, cross_attention, saliency, "
                           "gradcam_siglip, gradcam_connector, per_head, vision_vs_state, per_action_dim.",
            )
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_run_summary_stats(run_id: str) -> str:
    """Compute summary statistics across all available visualization types for a run.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import stats_engine

        run_dir, manifest = resolve_run(run_id)
        stats = stats_engine.compute_run_summary_stats(run_dir, manifest)
        if not stats:
            return error_response("No visualization data found for statistics.")
        return success_response(stats)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def format_stats_for_prompt(run_id: str, viz_type: str | None = None) -> str:
    """Get human-readable statistics text, suitable for including in an LLM prompt.

    Args:
        run_id: The run identifier.
        viz_type: Optional specific viz type. If omitted, returns summary stats for all types.
    """
    try:
        from web.backend.services import data_loader, stats_engine

        run_dir, manifest = resolve_run(run_id)

        if viz_type:
            # Compute stats for specific viz type, then format
            # Reuse get_viz_stats logic
            if viz_type in ("self_attention", "cross_attention", "saliency",
                            "gradcam_siglip", "gradcam_connector"):
                if viz_type == "self_attention":
                    hm = data_loader.load_self_attention_heatmaps(run_dir)
                elif viz_type == "cross_attention":
                    hm = data_loader.load_cross_attention_heatmaps(run_dir)
                else:
                    hm = data_loader.load_gradient_data(run_dir, viz_type)
                stats = stats_engine.compute_heatmap_stats(hm) if hm else {}
            elif viz_type == "per_head":
                data = data_loader.load_per_head_data(run_dir)
                stats = stats_engine.compute_per_head_stats(
                    data.get("heads", []), data.get("entropies")) if data else {}
            elif viz_type == "vision_vs_state":
                data = data_loader.load_vision_vs_state(run_dir)
                stats = stats_engine.compute_vision_vs_state_stats(data) if data else {}
            elif viz_type == "per_action_dim":
                data = data_loader.load_per_action_dim(run_dir)
                stats = stats_engine.compute_per_action_dim_stats(data) if data else {}
            else:
                stats = {}
        else:
            stats = stats_engine.compute_run_summary_stats(run_dir, manifest)

        formatted = stats_engine.format_stats_for_prompt(stats)
        return success_response({"formatted_text": formatted})
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")

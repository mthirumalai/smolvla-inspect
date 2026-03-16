"""Inspection tools — load and return visualization data (12 tools)."""

from __future__ import annotations

from ..server import mcp
from ..context import resolve_run
from ..helpers import success_response, error_response, filter_frames, image_to_base64


@mcp.tool()
def get_attention_heatmaps(
    run_id: str,
    attention_type: str = "self",
    frame_indices: list[int] | None = None,
) -> str:
    """Get attention heatmaps for a run.

    Args:
        run_id: The run identifier.
        attention_type: "self" or "cross".
        frame_indices: Optional list of frame indices to return (default: all, capped at 20).
    """
    try:
        from web.backend.services import data_loader

        run_dir, manifest = resolve_run(run_id)
        if attention_type == "cross":
            heatmaps = data_loader.load_cross_attention_heatmaps(run_dir)
        else:
            heatmaps = data_loader.load_self_attention_heatmaps(run_dir)

        if not heatmaps:
            return error_response(
                f"No {attention_type}-attention data found for run {run_id}.",
                suggestion=f"Re-run with --{'cross-attention' if attention_type == 'cross' else 'method last-layer'}.",
            )

        heatmaps = filter_frames(heatmaps, frame_indices)
        return success_response({
            "attention_type": attention_type,
            "num_frames": len(heatmaps),
            "heatmaps": heatmaps,
        })
    except ValueError as e:
        return error_response(str(e), suggestion="Use list_runs to see available run IDs.")
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_gradient_attribution(
    run_id: str,
    grad_type: str = "saliency",
    frame_indices: list[int] | None = None,
) -> str:
    """Get gradient-based attribution maps.

    Args:
        run_id: The run identifier.
        grad_type: One of "saliency", "gradcam_siglip", "gradcam_connector".
        frame_indices: Optional list of frame indices to return.
    """
    try:
        from web.backend.services import data_loader

        run_dir, manifest = resolve_run(run_id)
        data = data_loader.load_gradient_data(run_dir, grad_type)
        if not data:
            return error_response(
                f"No {grad_type} data found.",
                suggestion="Re-run with --gradient.",
            )

        data = filter_frames(data, frame_indices)
        return success_response({
            "grad_type": grad_type,
            "num_frames": len(data),
            "heatmaps": data,
        })
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_per_head_attention(run_id: str) -> str:
    """Get per-head attention data for the first frame.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        data = data_loader.load_per_head_data(run_dir)
        if not data:
            return error_response(
                "No per-head data found.",
                suggestion="Re-run with --show-heads.",
            )
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_per_step_cross_attention(run_id: str) -> str:
    """Get per-denoising-step cross-attention data.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        data = data_loader.load_per_step_cross_attention(run_dir)
        if not data:
            return error_response(
                "No per-step cross-attention data found.",
                suggestion="Re-run with --cross-attention.",
            )
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_vlm_layers(run_id: str) -> str:
    """Get VLM layer-wise gradient attribution data.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        data = data_loader.load_vlm_layers(run_dir)
        if not data:
            return error_response(
                "No VLM layer data found.",
                suggestion="Re-run with --gradient.",
            )
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_per_action_dim(run_id: str) -> str:
    """Get per-action-dimension GradCAM data.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        data = data_loader.load_per_action_dim(run_dir)
        if not data:
            return error_response(
                "No per-action-dimension data found.",
                suggestion="Re-run with --gradient.",
            )
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_language_diff(run_id: str) -> str:
    """Get language variant comparison data.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        data = data_loader.load_language_diff(run_dir)
        if not data:
            return error_response(
                "No language diff data found.",
                suggestion="Re-run with --gradient.",
            )
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_vision_vs_state(run_id: str) -> str:
    """Get vision vs proprioceptive state balance chart data.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        data = data_loader.load_vision_vs_state(run_dir)
        if not data:
            return error_response(
                "No vision vs state data found.",
                suggestion="Re-run with --gradient.",
            )
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_model_internals(run_id: str) -> str:
    """Get model internals data (WeightWatcher, entropy, redundancy).

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        data = data_loader.load_model_internals_data(run_dir)
        if not data:
            return error_response(
                "No model internals data found.",
                suggestion="Re-run with --model-health.",
            )
        return success_response(data)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_frame_image(run_id: str, frame_index: int = 0) -> str:
    """Get a base64-encoded frame image from a run.

    Args:
        run_id: The run identifier.
        frame_index: Frame index to retrieve (default: 0).
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        images = data_loader.list_images(run_dir)
        # Look for frame-specific images
        target = f"frame_{frame_index:03d}"
        for img_path in images:
            if target in img_path and "original" in img_path.lower():
                full_path = run_dir / img_path
                return success_response({
                    "frame_index": frame_index,
                    "image": image_to_base64(full_path),
                })

        # Fallback: return first matching frame image
        for img_path in images:
            if target in img_path:
                full_path = run_dir / img_path
                return success_response({
                    "frame_index": frame_index,
                    "image": image_to_base64(full_path),
                })

        return error_response(f"No image found for frame {frame_index}.")
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def list_run_images(run_id: str) -> str:
    """List all available images for a run.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        images = data_loader.list_images(run_dir)
        return success_response({"run_id": run_id, "images": images})
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def get_frames(run_id: str) -> str:
    """List all available data frames for a run.

    Args:
        run_id: The run identifier.
    """
    try:
        from web.backend.services import data_loader

        run_dir, _ = resolve_run(run_id)
        frames = data_loader.load_frames(run_dir)
        # Strip npz_path for cleaner output
        return success_response([
            {"index": f["index"]} for f in frames
        ])
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")

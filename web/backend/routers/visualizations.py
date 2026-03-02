"""Visualization data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import Settings
from ..models.schemas import VizData
from ..services import data_loader, run_scanner

router = APIRouter(prefix="/api/runs", tags=["visualizations"])


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


def _load_run(run_id: str):
    settings = _settings()
    try:
        return run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")


@router.get("/{run_id}/viz/{viz_type}", response_model=VizData)
async def get_viz_data(run_id: str, viz_type: str):
    run_dir, manifest = _load_run(run_id)
    avail = manifest.get("available_visualizations", {})
    is_legacy = manifest.get("_legacy", False)

    # For legacy runs, return image URLs only
    if is_legacy:
        images = data_loader.list_images(run_dir)
        matching = [img for img in images
                    if _image_matches_viz(img, viz_type)]
        return VizData(viz_type=viz_type, image_urls=matching)

    if not avail.get(viz_type, False):
        raise HTTPException(404, f"Visualization {viz_type} not available in this run")

    if viz_type == "self_attention":
        heatmaps = data_loader.load_self_attention_heatmaps(run_dir)
        return VizData(viz_type=viz_type, heatmaps=heatmaps,
                       frames=list(range(len(heatmaps))))

    if viz_type == "per_head":
        per_head = data_loader.load_per_head_data(run_dir)
        return VizData(viz_type=viz_type,
                       metadata=per_head or {},
                       frames=[0])

    if viz_type == "cross_attention":
        heatmaps = data_loader.load_cross_attention_heatmaps(run_dir)
        return VizData(viz_type=viz_type, heatmaps=heatmaps,
                       frames=list(range(len(heatmaps))))

    if viz_type == "per_step_cross_attention":
        per_step = data_loader.load_per_step_cross_attention(run_dir)
        return VizData(viz_type=viz_type,
                       metadata=per_step or {},
                       frames=list((per_step or {}).keys()))

    if viz_type == "saliency":
        heatmaps = data_loader.load_gradient_data(run_dir, "saliency")
        if heatmaps is None:
            raise HTTPException(404)
        return VizData(viz_type=viz_type, heatmaps=heatmaps,
                       frames=list(range(len(heatmaps))))

    if viz_type == "gradcam_siglip":
        heatmaps = data_loader.load_gradient_data(run_dir, "gradcam_siglip")
        if heatmaps is None:
            raise HTTPException(404)
        return VizData(viz_type=viz_type, heatmaps=heatmaps,
                       frames=list(range(len(heatmaps))))

    if viz_type == "gradcam_connector":
        heatmaps = data_loader.load_gradient_data(run_dir, "gradcam_connector")
        if heatmaps is None:
            raise HTTPException(404)
        return VizData(viz_type=viz_type, heatmaps=heatmaps,
                       frames=list(range(len(heatmaps))))

    if viz_type == "gradcam_vlm_layers":
        vlm = data_loader.load_vlm_layers(run_dir)
        images = _matching_images(run_dir, viz_type)
        if vlm is None and not images:
            raise HTTPException(404)
        return VizData(viz_type=viz_type, metadata=vlm or {},
                       image_urls=images,
                       frames=list(vlm.keys()) if vlm else [])

    if viz_type == "per_action_dim":
        dims = data_loader.load_per_action_dim(run_dir)
        images = _matching_images(run_dir, viz_type)
        if dims is None and not images:
            raise HTTPException(404)
        return VizData(viz_type=viz_type, metadata=dims or {},
                       image_urls=images,
                       frames=list(dims.keys()) if dims else [])

    if viz_type == "language_diff":
        diff = data_loader.load_language_diff(run_dir)
        images = _matching_images(run_dir, viz_type)
        if diff is None and not images:
            raise HTTPException(404)
        return VizData(viz_type=viz_type,
                       metadata=diff or {},
                       image_urls=images,
                       frames=list(diff.keys()) if diff else [])

    if viz_type == "vision_vs_state":
        vs = data_loader.load_vision_vs_state(run_dir)
        if vs is None:
            raise HTTPException(404)
        return VizData(viz_type=viz_type, chart_data=vs,
                       frames=list(range(len(vs))))

    raise HTTPException(404, f"Unknown viz type: {viz_type}")


def _matching_images(run_dir, viz_type: str) -> list[str]:
    """Return image URLs from the run that match the given viz type."""
    images = data_loader.list_images(run_dir)
    return [img for img in images if _image_matches_viz(img, viz_type)]


def _image_matches_viz(image_path: str, viz_type: str) -> bool:
    """Check if an image filename matches a viz type."""
    name = image_path.split("/")[-1].lower()
    mapping = {
        "self_attention": ["episode_dashboard", "attention_grid"],
        "per_head": ["per_head"],
        "cross_attention": ["cross_attn", "episode_dashboard"],
        "per_step_cross_attention": ["per_step_cross_attn"],
        "gradcam_siglip": ["episode_dashboard"],
        "gradcam_connector": ["episode_dashboard"],
        "gradcam_vlm_layers": ["vlm_layers"],
        "per_action_dim": ["per_action_dim"],
        "language_diff": ["language_diff"],
        "vision_vs_state": ["vision_vs_state"],
        "model_health": ["model_health"],
    }
    prefixes = mapping.get(viz_type, [])
    return any(name.startswith(p) for p in prefixes)

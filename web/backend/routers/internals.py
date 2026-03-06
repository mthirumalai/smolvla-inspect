"""Model internals endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import Settings
from ..models.schemas import ModelInternalsData
from ..services import data_loader, run_scanner

router = APIRouter(prefix="/api/runs", tags=["model_internals"])


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


@router.get("/{run_id}/internals", response_model=ModelInternalsData)
@router.get("/{run_id}/health", response_model=ModelInternalsData)
async def get_model_internals(run_id: str):
    settings = _settings()
    try:
        run_dir, manifest = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404)

    avail = manifest.get("available_visualizations", {})
    if not avail.get("model_internals", False):
        raise HTTPException(404, "Model internals data not available for this run")

    data = data_loader.load_model_internals_data(run_dir)
    if not data:
        raise HTTPException(404, "No model internals data files found")

    return ModelInternalsData(**data)

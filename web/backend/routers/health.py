"""Health metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import Settings
from ..models.schemas import HealthData
from ..services import data_loader, run_scanner

router = APIRouter(prefix="/api/runs", tags=["health"])


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


@router.get("/{run_id}/health", response_model=HealthData)
async def get_health(run_id: str):
    settings = _settings()
    try:
        run_dir, manifest = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404)

    avail = manifest.get("available_visualizations", {})
    if not avail.get("model_health", False):
        raise HTTPException(404, "Health data not available for this run")

    data = data_loader.load_health_data(run_dir)
    if not data:
        raise HTTPException(404, "No health data files found")

    return HealthData(**data)

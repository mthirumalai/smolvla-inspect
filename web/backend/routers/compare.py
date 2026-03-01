"""Cross-run comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import Settings
from ..models.schemas import CompareRequest, CompareResult
from ..services import comparison, run_scanner

router = APIRouter(prefix="/api", tags=["compare"])


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


@router.post("/compare", response_model=CompareResult)
async def compare_runs(req: CompareRequest):
    settings = _settings()
    if len(req.run_ids) < 2:
        raise HTTPException(400, "Need at least 2 run IDs")

    run_dirs = []
    manifests = []
    for rid in req.run_ids:
        try:
            rd, mf = run_scanner.load_manifest(settings.base_dir, rid)
            run_dirs.append(rd)
            manifests.append(mf)
        except FileNotFoundError:
            raise HTTPException(404, f"Run {rid} not found")

    result = comparison.compare_runs(
        run_dirs, manifests, req.viz_types, req.frame_indices)
    return CompareResult(runs=result["runs"], diffs=result["diffs"])

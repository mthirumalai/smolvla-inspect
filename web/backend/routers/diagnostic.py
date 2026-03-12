"""API endpoints for diagnostic runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/diagnostic", tags=["diagnostic"])


class DiagnosticRequest(BaseModel):
    dataset_id: str | None = None
    model_id: str | None = None
    max_counterfactuals: int = 3
    skip_counterfactuals: bool = False


def _resolve_run_dir(run_id: str, request: Request) -> Path:
    """Resolve run_id to a directory path."""
    from ..main import get_settings
    from ..services.diagnostic_service import get_run_dir
    settings = get_settings()
    run_dir = get_run_dir(run_id, settings)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run_dir


@router.get("/{run_id}")
async def get_diagnostic_report(run_id: str, request: Request) -> dict:
    """Load saved diagnostic report for a run."""
    run_dir = _resolve_run_dir(run_id, request)
    from ..services.diagnostic_service import load_diagnostic_report
    report = load_diagnostic_report(run_dir)
    if report is None:
        raise HTTPException(status_code=404, detail="No diagnostic report found for this run")
    return report


@router.post("/{run_id}/run")
async def run_diagnostic(run_id: str, req: DiagnosticRequest, request: Request):
    """Trigger diagnostic analysis on an existing run.

    Returns SSE stream with progress events.
    """
    run_dir = _resolve_run_dir(run_id, request)
    from ..main import get_settings
    settings = get_settings()

    async def generate():
        yield _sse_event("started", {"run_id": run_id})

        from ..services.diagnostic_service import run_diagnostic_async
        report, error = await run_diagnostic_async(
            run_dir=run_dir,
            dataset_id=req.dataset_id,
            model_id=req.model_id,
            max_counterfactuals=req.max_counterfactuals,
            skip_counterfactuals=req.skip_counterfactuals,
            settings=settings,
        )

        if error:
            yield _sse_event("error", {"message": error})
        else:
            yield _sse_event("complete", {"report": report})

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/{run_id}/matrix")
async def get_diagnostic_matrix(run_id: str, request: Request) -> dict:
    """Load the diagnostic matrix."""
    run_dir = _resolve_run_dir(run_id, request)
    from ..services.diagnostic_service import load_diagnostic_matrix
    matrix = load_diagnostic_matrix(run_dir)
    if matrix is None:
        raise HTTPException(status_code=404, detail="No diagnostic matrix found")
    return matrix


@router.get("/{run_id}/scene")
async def get_scene_data(run_id: str, request: Request) -> dict:
    """Load scene understanding data."""
    run_dir = _resolve_run_dir(run_id, request)
    from ..services.diagnostic_service import load_scene_data
    scene = load_scene_data(run_dir)
    if scene is None:
        raise HTTPException(status_code=404, detail="No scene data found")
    return scene


@router.get("/{run_id}/scene/annotated-frame")
async def get_annotated_frame(run_id: str, request: Request):
    """Serve the annotated scene frame image."""
    run_dir = _resolve_run_dir(run_id, request)
    img_path = run_dir / "diagnostic" / "scene" / "annotated_frame.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="No annotated frame")
    from fastapi.responses import FileResponse
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/{run_id}/counterfactual/{test_name}")
async def get_counterfactual_result(run_id: str, test_name: str, request: Request) -> dict:
    """Load a specific counterfactual result."""
    run_dir = _resolve_run_dir(run_id, request)
    from ..services.diagnostic_service import load_counterfactual_result
    result = load_counterfactual_result(run_dir, test_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No counterfactual result: {test_name}")
    return result


@router.get("/{run_id}/counterfactual/{test_name}/comparison")
async def get_counterfactual_comparison(run_id: str, test_name: str, request: Request):
    """Serve the counterfactual comparison image."""
    run_dir = _resolve_run_dir(run_id, request)
    img_path = run_dir / "diagnostic" / "counterfactuals" / test_name / "comparison.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="No comparison image")
    from fastapi.responses import FileResponse
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/{run_id}/status")
async def get_diagnostic_status(run_id: str, request: Request) -> dict:
    """Check whether a diagnostic report exists for this run."""
    run_dir = _resolve_run_dir(run_id, request)
    from ..services.diagnostic_service import diagnostic_exists
    return {
        "run_id": run_id,
        "has_diagnostic": diagnostic_exists(run_dir),
    }


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

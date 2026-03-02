"""Run management endpoints."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from ..config import Settings
from ..models.schemas import RunDetail, RunSummary
from ..services import data_loader, run_scanner

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


@router.get("", response_model=list[RunSummary])
async def list_runs(base_dir: str | None = Query(None)):
    settings = _settings()
    scan_dir = Path(base_dir) if base_dir else settings.base_dir
    return run_scanner.scan_directory(scan_dir)


@router.post("/scan", response_model=list[RunSummary])
async def scan_runs(base_dir: str | None = None):
    settings = _settings()
    scan_dir = Path(base_dir) if base_dir else settings.base_dir
    return run_scanner.scan_directory(scan_dir)


@router.post("/add", response_model=RunSummary)
async def add_run(path: str):
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(404, f"Directory not found: {path}")
    runs = run_scanner.scan_directory(p.parent)
    for r in runs:
        if r.name == p.name:
            return r
    raise HTTPException(404, f"No valid run found at {path}")


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str):
    settings = _settings()
    try:
        run_dir, manifest = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")
    return RunDetail(id=run_id, name=run_dir.name, manifest=manifest)


@router.get("/{run_id}/frames")
async def get_frames(run_id: str):
    settings = _settings()
    try:
        run_dir, manifest = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404)

    frames = data_loader.load_frames(run_dir)
    result = []
    for f in frames:
        result.append({
            "index": f["index"],
            "image_url": f"/api/runs/{run_id}/data/frames/frame_{f['index']:03d}.npz",
        })

    # Also include rendered frame images if available
    images = data_loader.list_images(run_dir)
    return {"frames": result, "images": images}


@router.get("/{run_id}/frame/{frame_idx}")
async def serve_frame_png(run_id: str, frame_idx: int):
    """Serve a stored frame .npz as a PNG image."""
    settings = _settings()
    try:
        run_dir, _ = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404)

    npz_path = run_dir / "data" / "frames" / f"frame_{frame_idx:03d}.npz"
    if not npz_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")

    data = np.load(npz_path, allow_pickle=False)
    arr = data[list(data.keys())[0]]  # uint8 RGB (H, W, 3)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/{run_id}/image/{path:path}")
async def serve_image(run_id: str, path: str):
    settings = _settings()
    try:
        run_dir, _ = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404)

    resolved = data_loader.resolve_image_path(run_dir, path)
    if resolved is None:
        raise HTTPException(404, f"Image not found: {path}")
    return FileResponse(resolved, media_type="image/png")

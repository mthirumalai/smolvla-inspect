"""Run notes endpoints — sidecar file approach."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import Settings
from ..services import run_scanner

router = APIRouter(prefix="/api/runs", tags=["notes"])

NOTES_FILENAME = "run_notes.json"


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


class NotesBody(BaseModel):
    notes: str


class NotesResponse(BaseModel):
    notes: str
    updated_at: str | None = None


@router.get("/{run_id}/notes", response_model=NotesResponse)
async def get_notes(run_id: str):
    settings = _settings()
    try:
        run_dir, _ = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")

    notes_path = run_dir / NOTES_FILENAME
    if not notes_path.exists():
        return NotesResponse(notes="", updated_at=None)

    with open(notes_path) as f:
        data = json.load(f)
    return NotesResponse(
        notes=data.get("notes", ""),
        updated_at=data.get("updated_at"),
    )


@router.put("/{run_id}/notes", response_model=NotesResponse)
async def set_notes(run_id: str, body: NotesBody):
    settings = _settings()
    try:
        run_dir, _ = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")

    now = datetime.now(timezone.utc).isoformat()
    data = {"notes": body.notes, "updated_at": now}

    notes_path = run_dir / NOTES_FILENAME
    with open(notes_path, "w") as f:
        json.dump(data, f, indent=2)

    return NotesResponse(notes=body.notes, updated_at=now)

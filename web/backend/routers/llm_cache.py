"""LLM analysis cache endpoints — sidecar file approach."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import Settings
from ..models.schemas import (
    LLMAnalysesResponse,
    LLMAnalysisEntry,
    SaveLLMAnalysisRequest,
)
from ..services import run_scanner

router = APIRouter(prefix="/api/runs", tags=["llm-cache"])

CACHE_FILENAME = "llm_analyses.json"


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


def _cache_path(run_dir: Path) -> Path:
    return run_dir / CACHE_FILENAME


def _read_cache(run_dir: Path) -> dict:
    path = _cache_path(run_dir)
    if not path.exists():
        return {"version": 1, "analyses": {}}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data.get("analyses"), dict):
            return {"version": 1, "analyses": {}}
        return data
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "analyses": {}}


def _write_cache(run_dir: Path, data: dict) -> None:
    """Atomic write via temp file + rename."""
    path = _cache_path(run_dir)
    fd, tmp = tempfile.mkstemp(dir=run_dir, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@router.get("/{run_id}/llm-analyses", response_model=LLMAnalysesResponse)
async def get_llm_analyses(run_id: str):
    settings = _settings()
    try:
        run_dir, _ = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")

    data = _read_cache(run_dir)
    analyses = {}
    for key, entry in data.get("analyses", {}).items():
        analyses[key] = LLMAnalysisEntry(**entry)
    return LLMAnalysesResponse(analyses=analyses)


@router.put("/{run_id}/llm-analyses", response_model=LLMAnalysisEntry)
async def save_llm_analysis(run_id: str, body: SaveLLMAnalysisRequest):
    settings = _settings()
    try:
        run_dir, _ = run_scanner.load_manifest(settings.base_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")

    now = datetime.now(timezone.utc).isoformat()
    data = _read_cache(run_dir)

    existing = data["analyses"].get(body.analysis_type)
    created_at = existing["created_at"] if existing else now

    entry = {
        "response": body.response,
        "model": body.model or settings.llm_model,
        "provider": body.provider or settings.llm_provider,
        "custom_prompt": body.custom_prompt,
        "created_at": created_at,
        "updated_at": now,
    }
    data["analyses"][body.analysis_type] = entry
    _write_cache(run_dir, data)

    return LLMAnalysisEntry(**entry)

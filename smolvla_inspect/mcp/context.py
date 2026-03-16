"""Shared context singleton — base_dir and import helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_base_dir: Path | None = None
_project_root: Path = Path(__file__).resolve().parent.parent.parent


def init_context(base_dir: str) -> None:
    """Set the base directory for run scanning."""
    global _base_dir
    _base_dir = Path(base_dir).resolve()

    # Ensure web.backend is importable (same pattern as serve.py)
    try:
        from web.backend.config import Settings  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(_project_root))


def get_base_dir() -> Path:
    if _base_dir is None:
        raise RuntimeError("init_context() has not been called")
    return _base_dir


def get_project_root() -> Path:
    return _project_root


def resolve_run(run_id: str) -> tuple[Path, dict]:
    """Find the run directory and load its manifest.

    Returns (run_dir, manifest).
    Raises ValueError with a helpful message if not found.
    """
    from web.backend.services.run_scanner import load_manifest

    try:
        return load_manifest(get_base_dir(), run_id)
    except FileNotFoundError:
        raise ValueError(
            f"Run '{run_id}' not found under {get_base_dir()}. "
            "Use list_runs to see available runs, or rescan_runs to refresh."
        )

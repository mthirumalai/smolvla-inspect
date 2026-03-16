"""Shared helpers — JSON serialisation, base64 encoding, response envelopes."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


def _json_default(obj):
    """Custom JSON serialiser for numpy/pathlib types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def success_response(data) -> str:
    """Wrap *data* in a standard success envelope."""
    return json.dumps({"success": True, "data": data}, default=_json_default)


def error_response(error: str, suggestion: str | None = None) -> str:
    """Wrap an error string in a standard error envelope."""
    payload: dict = {"success": False, "error": error}
    if suggestion:
        payload["suggestion"] = suggestion
    return json.dumps(payload)


def image_to_base64(path: Path) -> str:
    """Read an image file and return its base64-encoded content."""
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def filter_frames(data: list, frame_indices: list[int] | None, max_frames: int = 20) -> list:
    """Subset a list by frame indices, with a default cap."""
    if frame_indices is not None:
        idx_set = set(frame_indices)
        data = [d for i, d in enumerate(data) if i in idx_set]
    if len(data) > max_frames:
        data = data[:max_frames]
    return data

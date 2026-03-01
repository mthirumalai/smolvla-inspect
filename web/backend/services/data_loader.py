"""Load .npz / .json data from run directories."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    return dict(np.load(path, allow_pickle=False))


def load_frames(run_dir: Path) -> list[dict]:
    """Return list of {index, path} for each frame npz."""
    frames_dir = run_dir / "data" / "frames"
    if not frames_dir.is_dir():
        return []
    result = []
    for f in sorted(frames_dir.glob("frame_*.npz")):
        idx = int(f.stem.split("_")[1])
        result.append({"index": idx, "npz_path": str(f)})
    return result


def load_self_attention_heatmaps(run_dir: Path) -> list[list[list[float]]]:
    """Load self-attention heatmaps as nested lists (JSON-serialisable)."""
    data = _load_npz(run_dir / "data" / "self_attention" / "heatmaps.npz")
    result = []
    for key in sorted(data.keys()):
        result.append(data[key].tolist())
    return result


def load_per_head_data(run_dir: Path) -> dict | None:
    """Load per-head data for first frame."""
    data = _load_npz(run_dir / "data" / "self_attention" / "per_head_frame0.npz")
    if not data:
        return None
    heads = []
    entropies = None
    for key in sorted(data.keys()):
        if key.startswith("head_"):
            heads.append(data[key].tolist())
        elif key == "entropies":
            entropies = data[key].tolist()
    return {"heads": heads, "entropies": entropies}


def load_cross_attention_heatmaps(run_dir: Path) -> list[list[list[float]]]:
    data = _load_npz(run_dir / "data" / "cross_attention" / "heatmaps.npz")
    return [data[k].tolist() for k in sorted(data.keys())]


def load_per_step_cross_attention(run_dir: Path) -> dict | None:
    data = _load_npz(run_dir / "data" / "cross_attention" / "per_step.npz")
    if not data:
        return None
    # Organize by frame → step
    frames: dict[int, list] = {}
    for key in sorted(data.keys()):
        parts = key.split("_")  # frame000_step000
        fi = int(parts[0].replace("frame", ""))
        frames.setdefault(fi, []).append(data[key].tolist())
    return frames


def load_gradient_data(run_dir: Path, grad_type: str) -> list[list[list[float]]] | None:
    """Load a gradient npz (saliency, gradcam_siglip, gradcam_connector)."""
    path = run_dir / "data" / "gradient" / f"{grad_type}.npz"
    data = _load_npz(path)
    if not data:
        return None
    return [data[k].tolist() for k in sorted(data.keys())]


def load_vlm_layers(run_dir: Path) -> dict | None:
    data = _load_npz(run_dir / "data" / "gradient" / "vlm_layers.npz")
    if not data:
        return None
    # Organize into frame → layer → modality
    result: dict = {}
    for key in sorted(data.keys()):
        parts = key.split("_")  # frame000_layer00_vision
        fi = int(parts[0].replace("frame", ""))
        li = int(parts[1].replace("layer", ""))
        modality = "_".join(parts[2:])
        result.setdefault(fi, {}).setdefault(li, {})[modality] = data[key].tolist()
    return result


def load_per_action_dim(run_dir: Path) -> dict | None:
    data = _load_npz(run_dir / "data" / "gradient" / "per_action_dim.npz")
    if not data:
        return None
    result: dict = {}
    for key in sorted(data.keys()):
        parts = key.split("_")
        fi = int(parts[0].replace("frame", ""))
        di = int(parts[1].replace("dim", ""))
        result.setdefault(fi, {})[di] = data[key].tolist()
    return result


def load_language_diff(run_dir: Path) -> dict | None:
    data = _load_npz(run_dir / "data" / "gradient" / "language_diff.npz")
    if not data:
        return None
    result: dict = {}
    for key in sorted(data.keys()):
        parts = key.split("_")
        fi = int(parts[0].replace("frame", ""))
        variant = "_".join(parts[1:])  # original, alternative, diff
        result.setdefault(fi, {})[variant] = data[key].tolist()
    return result


def load_vision_vs_state(run_dir: Path) -> list[dict] | None:
    path = run_dir / "data" / "gradient" / "vision_vs_state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_health_data(run_dir: Path) -> dict:
    result = {}
    health_dir = run_dir / "data" / "health"
    if not health_dir.is_dir():
        return result
    for name in ("weightwatcher", "entropy", "redundancy"):
        path = health_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                result[name] = json.load(f)
    return result


def list_images(run_dir: Path) -> list[str]:
    """List all PNG image paths relative to run_dir."""
    images_dir = run_dir / "images"
    if not images_dir.is_dir():
        # Legacy: images are in run_dir root
        images_dir = run_dir
    paths = []
    for f in sorted(images_dir.rglob("*.png")):
        paths.append(str(f.relative_to(run_dir)))
    return paths


def resolve_image_path(run_dir: Path, image_path: str) -> Path | None:
    """Resolve an image path securely within run_dir."""
    full = (run_dir / image_path).resolve()
    if not full.is_relative_to(run_dir.resolve()):
        return None
    if not full.exists():
        return None
    return full

"""Scan directories for run_manifest.json and parse run metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..models.schemas import RunSummary


# Known PNGs that indicate a legacy (pre-manifest) run
_LEGACY_MARKERS = {
    "episode_dashboard_ep",
    "per_head_ep",
    "per_step_cross_attn_ep",
    "vlm_layers_ep",
    "per_action_dim_ep",
    "language_diff_ep",
    "vision_vs_state_ep",
    "model_health_report",
    "positional_baseline",
}


def _run_id_from_path(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]


def _is_legacy_run(directory: Path) -> bool:
    """Check if directory contains known PNG filenames without a manifest."""
    if (directory / "run_manifest.json").exists():
        return False
    for f in directory.iterdir():
        if f.suffix == ".png":
            for marker in _LEGACY_MARKERS:
                if f.name.startswith(marker):
                    return True
    return False


def _build_legacy_manifest(directory: Path) -> dict:
    """Construct a partial manifest for legacy runs (image-only mode)."""
    images = sorted(
        str(f.relative_to(directory))
        for f in directory.rglob("*.png")
    )
    # Infer available viz from filenames
    viz = {}
    for img in images:
        name = os.path.basename(img)
        if name.startswith("episode_dashboard"):
            viz["self_attention"] = True
        elif name.startswith("per_head"):
            viz["per_head"] = True
        elif name.startswith("per_step_cross_attn"):
            viz["per_step_cross_attention"] = True
            viz["cross_attention"] = True
        elif name.startswith("vlm_layers"):
            viz["gradcam_vlm_layers"] = True
        elif name.startswith("per_action_dim"):
            viz["per_action_dim"] = True
        elif name.startswith("language_diff"):
            viz["language_diff"] = True
        elif name.startswith("vision_vs_state"):
            viz["vision_vs_state"] = True
        elif name.startswith("model_health"):
            viz["model_health"] = True

    return {
        "version": 0,
        "created_at": None,
        "cli_args": {},
        "model_info": {},
        "dataset_info": {},
        "available_visualizations": viz,
        "images": images,
        "_legacy": True,
        "_dir": str(directory),
    }


def scan_directory(base_dir: Path) -> list[RunSummary]:
    """Scan *base_dir* (non-recursively) for run folders.

    A run folder is any child directory containing ``run_manifest.json``
    or known legacy PNGs.
    """
    runs: list[RunSummary] = []
    base = Path(base_dir)
    if not base.is_dir():
        return runs

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        manifest_path = child / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            runs.append(RunSummary(
                id=_run_id_from_path(child),
                name=child.name,
                created_at=manifest.get("created_at"),
                model_id=(manifest.get("model_info") or {}).get("model_id"),
                dataset_id=(manifest.get("dataset_info") or {}).get("dataset_id"),
                episode_idx=(manifest.get("dataset_info") or {}).get("episode_idx"),
                num_frames=(manifest.get("dataset_info") or {}).get("num_frames"),
                available_visualizations=manifest.get("available_visualizations", {}),
                is_legacy=False,
            ))
        elif _is_legacy_run(child):
            runs.append(RunSummary(
                id=_run_id_from_path(child),
                name=child.name,
                is_legacy=True,
                available_visualizations=_build_legacy_manifest(child).get(
                    "available_visualizations", {}),
            ))

    # Also check if base_dir itself is a legacy run (flat layout)
    if not runs and _is_legacy_run(base):
        runs.append(RunSummary(
            id=_run_id_from_path(base),
            name=base.name,
            is_legacy=True,
            available_visualizations=_build_legacy_manifest(base).get(
                "available_visualizations", {}),
        ))

    return runs


def load_manifest(base_dir: Path, run_id: str) -> tuple[Path, dict]:
    """Find the run folder matching *run_id* and return (path, manifest)."""
    base = Path(base_dir)
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if _run_id_from_path(child) == run_id:
            manifest_path = child / "run_manifest.json"
            if manifest_path.exists():
                return child, json.loads(manifest_path.read_text())
            if _is_legacy_run(child):
                return child, _build_legacy_manifest(child)

    # Check if base itself matches
    if _run_id_from_path(base) == run_id:
        if (base / "run_manifest.json").exists():
            return base, json.loads((base / "run_manifest.json").read_text())
        if _is_legacy_run(base):
            return base, _build_legacy_manifest(base)

    raise FileNotFoundError(f"Run {run_id} not found under {base_dir}")

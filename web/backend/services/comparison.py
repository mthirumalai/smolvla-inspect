"""Cross-run diff computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data_loader


def _diff_heatmaps(a: list[list[list[float]]],
                   b: list[list[list[float]]]) -> list[list[list[float]]]:
    """Compute element-wise difference between two sets of heatmaps."""
    result = []
    for i in range(min(len(a), len(b))):
        arr_a = np.array(a[i], dtype=np.float32)
        arr_b = np.array(b[i], dtype=np.float32)
        if arr_a.shape != arr_b.shape:
            continue
        diff = (arr_a - arr_b).tolist()
        result.append(diff)
    return result


def compute_config_diff(manifests: list[dict]) -> dict:
    """Compare configs across runs, returning only fields that differ."""
    if len(manifests) < 2:
        return {}

    def _flatten(d: dict, prefix: str = "") -> dict:
        flat = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(_flatten(v, key))
            else:
                flat[key] = v
        return flat

    # Flatten the config-relevant sections of each manifest
    config_keys = ("cli_args", "model_info", "dataset_info")
    flattened = []
    for mf in manifests:
        flat: dict = {}
        for ck in config_keys:
            section = mf.get(ck, {})
            if isinstance(section, dict):
                flat.update(_flatten(section, ck))
        flattened.append(flat)

    # Find fields that differ across any pair of runs
    all_keys = set()
    for f in flattened:
        all_keys.update(f.keys())

    diffs: dict = {}
    for key in sorted(all_keys):
        values = [f.get(key) for f in flattened]
        # Check if all values are the same
        if len(set(str(v) for v in values)) > 1:
            diffs[key] = values

    return diffs


def compare_runs(run_dirs: list[Path],
                 manifests: list[dict],
                 viz_types: list[str],
                 frame_indices: list[int] | None = None) -> dict:
    """Compute side-by-side data and diffs for the requested viz types."""
    runs_data = []
    for rd, mf in zip(run_dirs, manifests):
        entry = {
            "id": mf.get("_id", rd.name),
            "name": rd.name,
            "manifest": mf,
            "viz": {},
        }
        avail = mf.get("available_visualizations", {})
        for vt in viz_types:
            canonical_vt = "model_internals" if vt in ("model_internals", "model_health") else vt
            if not avail.get(canonical_vt, False):
                continue
            if canonical_vt == "self_attention":
                entry["viz"][canonical_vt] = data_loader.load_self_attention_heatmaps(rd)
            elif canonical_vt == "cross_attention":
                entry["viz"][canonical_vt] = data_loader.load_cross_attention_heatmaps(rd)
            elif canonical_vt == "vision_vs_state":
                entry["viz"][canonical_vt] = data_loader.load_vision_vs_state(rd)
            elif canonical_vt == "model_internals":
                entry["viz"][canonical_vt] = data_loader.load_model_internals_data(rd)
        runs_data.append(entry)

    # Compute diffs where both runs have the same viz type
    diffs: dict = {}
    if len(runs_data) >= 2:
        for vt in viz_types:
            canonical_vt = "model_internals" if vt in ("model_internals", "model_health") else vt
            a_data = runs_data[0].get("viz", {}).get(canonical_vt)
            b_data = runs_data[1].get("viz", {}).get(canonical_vt)
            if a_data is None or b_data is None:
                continue
            if canonical_vt in ("self_attention", "cross_attention"):
                diffs[canonical_vt] = _diff_heatmaps(a_data, b_data)

    return {"runs": runs_data, "diffs": diffs}

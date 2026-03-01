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
            if not avail.get(vt, False):
                continue
            if vt == "self_attention":
                entry["viz"][vt] = data_loader.load_self_attention_heatmaps(rd)
            elif vt == "cross_attention":
                entry["viz"][vt] = data_loader.load_cross_attention_heatmaps(rd)
            elif vt == "vision_vs_state":
                entry["viz"][vt] = data_loader.load_vision_vs_state(rd)
            elif vt == "model_health":
                entry["viz"][vt] = data_loader.load_health_data(rd)
        runs_data.append(entry)

    # Compute diffs where both runs have the same viz type
    diffs: dict = {}
    if len(runs_data) >= 2:
        for vt in viz_types:
            a_data = runs_data[0].get("viz", {}).get(vt)
            b_data = runs_data[1].get("viz", {}).get(vt)
            if a_data is None or b_data is None:
                continue
            if vt in ("self_attention", "cross_attention"):
                diffs[vt] = _diff_heatmaps(a_data, b_data)

    return {"runs": runs_data, "diffs": diffs}

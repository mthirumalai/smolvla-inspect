"""Report generation and export for diagnostic results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .models import DiagnosticReport, SceneSegmentation


def save_diagnostic_report(report: DiagnosticReport, run_dir: str) -> str:
    """Save a complete diagnostic report to the run directory.

    Creates:
        run_dir/diagnostic/report.json
        run_dir/diagnostic/report.md
        run_dir/diagnostic/matrix.json
        run_dir/diagnostic/scene/detections.json
        run_dir/diagnostic/scene/segmentation.npz
        run_dir/diagnostic/scene/annotated_frame.png
        run_dir/diagnostic/counterfactuals/<test_name>/comparison.png
        run_dir/diagnostic/counterfactuals/<test_name>/result.json
        run_dir/diagnostic/evidence_chain.json

    Returns:
        Path to the diagnostic/ directory.
    """
    diag_dir = os.path.join(run_dir, "diagnostic")
    os.makedirs(diag_dir, exist_ok=True)

    # 1. Full report as JSON
    report_json = report.to_json()
    with open(os.path.join(diag_dir, "report.json"), "w") as f:
        f.write(report_json)

    # 2. Full report as Markdown
    with open(os.path.join(diag_dir, "report.md"), "w") as f:
        f.write(report.to_markdown())

    # 3. Matrix as standalone JSON
    with open(os.path.join(diag_dir, "matrix.json"), "w") as f:
        json.dump(report.matrix.to_dict(), f, indent=2, default=_json_default)

    # 4. Scene data
    scene_dir = os.path.join(diag_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)
    _save_scene_data(report.scene, scene_dir)

    # 5. Counterfactual results
    for cf in report.counterfactual_results:
        cf_dir = os.path.join(diag_dir, "counterfactuals", cf.test_type)
        os.makedirs(cf_dir, exist_ok=True)

        # Result JSON
        cf_data = {
            "hypothesis_id": cf.hypothesis_id,
            "test_type": cf.test_type,
            "action_delta_l2": cf.action_delta_l2,
            "action_delta_per_dim": cf.action_delta_per_dim,
            "gradcam_shift": cf.gradcam_shift,
            "attribution_shift_per_region": cf.attribution_shift_per_region,
            "confirmed": cf.confirmed,
            "metrics": cf.metrics,
        }
        with open(os.path.join(cf_dir, "result.json"), "w") as f:
            json.dump(cf_data, f, indent=2, default=_json_default)

        # Comparison image
        if cf.visual_comparison is not None:
            _save_image(cf.visual_comparison, os.path.join(cf_dir, "comparison.png"))

    # 6. Evidence chain
    evidence_data = []
    # Evidence entries aren't stored on the report directly, but we can
    # reconstruct from the report's components
    for symptom in report.symptoms:
        evidence_data.append({
            "phase": "matrix",
            "type": "symptom",
            "symptom_type": symptom.type,
            "severity": symptom.severity,
            "description": symptom.description,
        })
    for h in report.hypotheses:
        evidence_data.append({
            "phase": "hypothesize",
            "type": "hypothesis",
            "id": h.id,
            "description": h.description,
            "confidence": h.confidence,
            "test_type": h.test_type,
        })
    for cf in report.counterfactual_results:
        evidence_data.append({
            "phase": "counterfactual",
            "type": "counterfactual_result",
            "hypothesis_id": cf.hypothesis_id,
            "test_type": cf.test_type,
            "action_delta_l2": cf.action_delta_l2,
            "confirmed": cf.confirmed,
            "metrics": cf.metrics,
        })

    if report.semantic_probe is not None:
        evidence_data.append({
            "phase": "representation",
            "type": "semantic_probe",
            "target_object": report.semantic_probe.target_object,
            "summary": report.semantic_probe.summary,
        })

    if report.qk_probe is not None:
        evidence_data.append({
            "phase": "representation",
            "type": "qk_probe",
            "layer": report.qk_probe.layer,
            "summary": report.qk_probe.summary,
            "dominant_head_type": report.qk_probe.dominant_head_type,
        })

    if report.spatial_object_diagnosis is not None:
        diag = report.spatial_object_diagnosis
        evidence_data.append({
            "phase": "disambiguation",
            "type": "spatial_object_diagnosis",
            "target_object": diag.target_object,
            "verdict": diag.verdict,
            "confidence": diag.confidence,
            "spatial_score": diag.spatial_score,
            "object_score": diag.object_score,
        })

    with open(os.path.join(diag_dir, "evidence_chain.json"), "w") as f:
        json.dump(evidence_data, f, indent=2, default=_json_default)

    print(f"  Diagnostic report saved to {diag_dir}")
    return diag_dir


def regenerate_report_markdown(run_dir: str) -> str:
    """Re-render ``report.md`` from the saved ``report.json``.

    Use this after changing the markdown template in
    :pyclass:`DiagnosticReport.to_markdown` to refresh an existing report
    without re-running the diagnostic pipeline.

    Returns the path to the regenerated ``report.md``.
    """
    data = load_diagnostic_report(run_dir)
    if data is None:
        raise FileNotFoundError(
            f"No diagnostic report found in {run_dir}/diagnostic/report.json"
        )
    report = DiagnosticReport.from_dict(data)
    md_path = os.path.join(run_dir, "diagnostic", "report.md")
    with open(md_path, "w") as f:
        f.write(report.to_markdown())
    print(f"  Regenerated {md_path}")
    return md_path


def load_diagnostic_report(run_dir: str) -> dict | None:
    """Load a saved diagnostic report from a run directory.

    Returns the report JSON dict, or None if no diagnostic exists.
    Handles legacy double-encoded files transparently.
    """
    report_path = os.path.join(run_dir, "diagnostic", "report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        data = json.load(f)
    # Handle legacy double-encoded files (string instead of dict)
    if isinstance(data, str):
        data = json.loads(data)
    return data


def load_diagnostic_matrix(run_dir: str) -> dict | None:
    """Load the diagnostic matrix from a run directory."""
    matrix_path = os.path.join(run_dir, "diagnostic", "matrix.json")
    if not os.path.exists(matrix_path):
        return None
    with open(matrix_path) as f:
        return json.load(f)


def load_scene_data(run_dir: str) -> dict | None:
    """Load scene understanding data from a run directory."""
    scene_dir = os.path.join(run_dir, "diagnostic", "scene")
    det_path = os.path.join(scene_dir, "detections.json")
    if not os.path.exists(det_path):
        return None
    with open(det_path) as f:
        data = json.load(f)
    # Add image URL info
    annotated_path = os.path.join(scene_dir, "annotated_frame.png")
    data["has_annotated_frame"] = os.path.exists(annotated_path)
    return data


def load_counterfactual_result(run_dir: str, test_name: str) -> dict | None:
    """Load a specific counterfactual result."""
    result_path = os.path.join(run_dir, "diagnostic", "counterfactuals",
                               test_name, "result.json")
    if not os.path.exists(result_path):
        return None
    with open(result_path) as f:
        data = json.load(f)
    # Check for comparison image
    comp_path = os.path.join(run_dir, "diagnostic", "counterfactuals",
                             test_name, "comparison.png")
    data["has_comparison_image"] = os.path.exists(comp_path)
    return data


def _save_scene_data(scene: SceneSegmentation, scene_dir: str):
    """Save scene understanding data."""
    # Detections JSON
    detections = []
    for obj in scene.objects:
        det = {
            "label": obj.label,
            "box": list(obj.box),
            "score": float(obj.score),
            "has_mask": obj.mask is not None,
        }
        detections.append(det)

    with open(os.path.join(scene_dir, "detections.json"), "w") as f:
        json.dump({"objects": detections, "image_shape": list(scene.image_shape)}, f, indent=2)

    # Segmentation masks as NPZ
    masks = {}
    for obj in scene.objects:
        if obj.mask is not None:
            masks[obj.label.replace(" ", "_")] = obj.mask.astype(np.uint8)
    if scene.background_mask is not None:
        masks["background"] = scene.background_mask.astype(np.uint8)
    if masks:
        np.savez_compressed(os.path.join(scene_dir, "segmentation.npz"), **masks)

    # Annotated frame with detection boxes
    _save_annotated_frame(scene, scene_dir)


def _save_annotated_frame(scene: SceneSegmentation, scene_dir: str):
    """Draw detection boxes on a blank canvas (or we skip if no PIL)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    h, w = scene.image_shape
    # Create a simple visualization showing colored masks
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
        (255, 100, 100),  # red
        (100, 255, 100),  # green
        (100, 100, 255),  # blue
        (255, 255, 100),  # yellow
        (255, 100, 255),  # magenta
        (100, 255, 255),  # cyan
    ]

    for i, obj in enumerate(scene.objects):
        color = colors[i % len(colors)]
        if obj.mask is not None:
            mask = obj.mask
            if mask.shape != (h, w):
                # Resize mask
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize((w, h), PILImage.NEAREST)
                mask = np.array(mask_img) > 127
            for c in range(3):
                canvas[:, :, c] = np.where(mask, color[c], canvas[:, :, c])

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    for i, obj in enumerate(scene.objects):
        x1, y1, x2, y2 = obj.box
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{obj.label} ({obj.score:.2f})"
        draw.text((x1, max(y1 - 12, 0)), label, fill=color)

    img.save(os.path.join(scene_dir, "annotated_frame.png"))


def _save_image(arr: np.ndarray, path: str):
    """Save a numpy array as PNG image."""
    try:
        from PIL import Image
    except ImportError:
        return
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

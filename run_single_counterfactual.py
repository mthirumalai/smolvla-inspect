#!/usr/bin/env python3
"""Run a single counterfactual test on an existing run directory.

Loads model, dataset, and scene data from a previous run, then executes
one counterfactual test and saves the result + comparison image.
Much faster than re-running the full diagnostic pipeline.

Usage:
    python run_single_counterfactual.py <run_dir> <test_name> [--param key=value ...]
    python run_single_counterfactual.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Run a single counterfactual test on an existing run directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s outputs/mthirumalai/finetuned_model background_substitution
  %(prog)s outputs/mthirumalai/finetuned_model distractor_insertion --param position='[200,200]'
  %(prog)s outputs/mthirumalai/finetuned_model task_string_swap --viz-only
  %(prog)s --list
""",
    )
    parser.add_argument("run_dir", nargs="?", help="Path to existing run directory")
    parser.add_argument("test_name", nargs="?", help="Counterfactual test name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--param", action="append", default=[],
                        help="Test parameter as key=value (repeatable)")
    parser.add_argument("--viz-only", action="store_true",
                        help="Only regenerate visualization from existing result.json")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode index (default: from manifest)")

    args = parser.parse_args()

    # Import here so --list/--help are fast
    _ensure_imports()

    if args.list:
        _list_tests()
        return

    if not args.run_dir or not args.test_name:
        parser.error("run_dir and test_name are required (or use --list)")

    run_dir = args.run_dir
    test_name = args.test_name

    # Validate run directory
    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: No run_manifest.json in {run_dir}")
        sys.exit(1)

    # Parse test params
    test_params = _parse_params(args.param)

    if args.viz_only:
        _regenerate_viz(run_dir, test_name)
        return

    # Load manifest for model/dataset info
    with open(manifest_path) as f:
        manifest = json.load(f)

    cli_args = manifest.get("cli_args", {})
    model_id = cli_args.get("model")
    dataset_id = manifest.get("dataset_info", {}).get("dataset_id") or cli_args.get("dataset")
    image_key = cli_args.get("image_key")
    image_map_str = cli_args.get("image_map")
    episode_idx = args.episode if args.episode is not None else cli_args.get("episode", 0)

    if not model_id:
        print("ERROR: Cannot determine model from manifest. Specify --model?")
        sys.exit(1)
    if not dataset_id:
        print("ERROR: Cannot determine dataset from manifest.")
        sys.exit(1)

    # Device
    import torch
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n{'=' * 50}")
    print(f"  Quick Counterfactual Test")
    print(f"{'=' * 50}")
    print(f"  Run dir:  {run_dir}")
    print(f"  Test:     {test_name}")
    print(f"  Model:    {model_id}")
    print(f"  Dataset:  {dataset_id}")
    print(f"  Device:   {device}")
    if test_params:
        print(f"  Params:   {test_params}")
    print(f"{'=' * 50}\n")

    # ── Load model ──
    t0 = time.time()
    print("  Loading model...", end="", flush=True)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()
    print(f" done ({time.time() - t0:.1f}s)")

    # ── Load dataset ──
    t0 = time.time()
    print("  Loading dataset...", end="", flush=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(dataset_id)
    print(f" done ({time.time() - t0:.1f}s)")

    # Resolve image key
    if image_key is None:
        from smolvla_inspect.data import find_image_keys
        image_keys = find_image_keys(dataset)
        image_key = image_keys[0] if image_keys else "observation.images.top"

    # Parse image map
    image_map = None
    if image_map_str:
        from smolvla_inspect.data import parse_image_map
        image_map = parse_image_map(image_map_str)

    # ── Load scene data ──
    print("  Loading scene data...", end="", flush=True)
    scene = _load_scene(run_dir)
    if scene is None:
        print("\n  WARNING: No scene data found — running scene detection...")
        scene = _detect_scene(dataset, episode_idx, image_key, device)
    else:
        print(" done")

    # ── Get sample ──
    print("  Loading sample...", end="", flush=True)
    first_frame_idx = _get_first_frame_idx(dataset, episode_idx)
    sample = dataset[first_frame_idx]
    print(f" done (frame {first_frame_idx})")

    # ── Validate test name ──
    from smolvla_inspect.diagnostic.registry import REGISTRY
    primitive_name = f"counterfactual.{test_name}"
    if primitive_name not in REGISTRY:
        print(f"\n  ERROR: Unknown test '{test_name}'")
        print(f"  Available: {', '.join(n.removeprefix('counterfactual.') for n in REGISTRY if n.startswith('counterfactual.'))}")
        sys.exit(1)

    # ── Apply default params if not provided ──
    test_params = _apply_defaults(test_name, test_params, scene)

    # ── Run the test ──
    spec = REGISTRY[primitive_name]
    params = dict(test_params)
    params.update({
        "policy": policy,
        "sample": sample,
        "dataset": dataset,
        "image_key": image_key,
        "device": device,
        "image_map": image_map,
    })
    if "segmentation" in spec.fn.__code__.co_varnames:
        params["segmentation"] = scene
    if "episode_idx" in spec.fn.__code__.co_varnames:
        params["episode_idx"] = episode_idx

    print(f"\n  Running {test_name}...", flush=True)
    t0 = time.time()
    result = spec.fn(**params)
    elapsed = time.time() - t0

    # ── Save result ──
    import numpy as np

    cf_dir = os.path.join(run_dir, "diagnostic", "counterfactuals", test_name)
    os.makedirs(cf_dir, exist_ok=True)

    # Save result.json
    result_dict = {
        "hypothesis_id": result.hypothesis_id,
        "test_type": result.test_type,
        "action_delta_l2": float(result.action_delta_l2),
        "action_delta_per_dim": [float(x) for x in result.action_delta_per_dim],
        "gradcam_shift": float(result.gradcam_shift) if result.gradcam_shift else 0.0,
        "attribution_shift_per_region": result.attribution_shift_per_region or {},
        "confirmed": result.confirmed,
        "metrics": result.metrics or {},
    }
    result_path = os.path.join(cf_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    # Save comparison image
    if result.visual_comparison is not None:
        from PIL import Image
        comp_path = os.path.join(cf_dir, "comparison.png")
        Image.fromarray(result.visual_comparison).save(comp_path)
        print(f"  Saved: {comp_path}")

    print(f"\n  Result ({elapsed:.1f}s):")
    print(f"    Action delta (L2): {result.action_delta_l2:.4f}")
    print(f"    Confirmed: {result.confirmed}")
    print(f"    Saved to: {result_path}")
    print(f"{'=' * 50}\n")


def _ensure_imports():
    """Check that the package is importable."""
    try:
        import smolvla_inspect  # noqa: F401
    except ImportError:
        # Try adding the project root to sys.path
        root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, root)


def _list_tests():
    """Print available counterfactual tests."""
    _ensure_imports()
    # Force registry population by importing the counterfactual module
    import smolvla_inspect.diagnostic.counterfactual  # noqa: F401
    from smolvla_inspect.diagnostic.registry import list_primitives

    print("\nAvailable counterfactual tests:\n")
    for spec in list_primitives(category="counterfactual"):
        name = spec.name.removeprefix("counterfactual.")
        print(f"  {name}")
        print(f"    {spec.description}")
        if spec.param_schema:
            print(f"    Params: {spec.param_schema}")
        print()


def _parse_params(param_list: list[str]) -> dict:
    """Parse --param key=value arguments into a dict."""
    params = {}
    for p in param_list:
        if "=" not in p:
            print(f"ERROR: Invalid param '{p}' — expected key=value")
            sys.exit(1)
        key, val = p.split("=", 1)
        # Try JSON parsing for lists, numbers, bools
        try:
            params[key] = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            params[key] = val
    return params


def _load_scene(run_dir: str):
    """Reconstruct SceneSegmentation from saved files."""
    import numpy as np
    from smolvla_inspect.diagnostic.models import SceneSegmentation, DetectedObject

    scene_dir = os.path.join(run_dir, "diagnostic", "scene")
    det_path = os.path.join(scene_dir, "detections.json")
    seg_path = os.path.join(scene_dir, "segmentation.npz")

    if not os.path.exists(det_path):
        return None

    with open(det_path) as f:
        det_data = json.load(f)

    seg_data = {}
    if os.path.exists(seg_path):
        seg_data = dict(np.load(seg_path))

    objects = []
    for det in det_data["objects"]:
        mask_key = det["label"].replace(" ", "_")
        mask = seg_data.get(mask_key)
        if mask is not None:
            mask = mask.astype(bool)
        objects.append(DetectedObject(
            label=det["label"],
            box=tuple(det["box"]),
            score=det["score"],
            mask=mask,
        ))

    bg_mask = seg_data.get("background")
    if bg_mask is not None:
        bg_mask = bg_mask.astype(bool)

    h, w = det_data["image_shape"]
    return SceneSegmentation(
        objects=objects,
        background_mask=bg_mask,
        image_shape=(h, w),
    )


def _get_first_frame_idx(dataset, episode_idx: int) -> int:
    """Get the dataset index of the first frame in an episode."""
    try:
        return dataset.meta.episodes["dataset_from_index"][episode_idx]
    except (AttributeError, KeyError):
        try:
            return dataset.episode_data_index["from"][episode_idx].item()
        except (AttributeError, KeyError):
            return episode_idx * 200


def _detect_scene(dataset, episode_idx: int, image_key: str, device: str):
    """Run scene detection from scratch (fallback when no saved scene)."""
    from smolvla_inspect.diagnostic.scene import detect_scene

    first_idx = _get_first_frame_idx(dataset, episode_idx)
    sample = dataset[first_idx]
    return detect_scene(sample, image_key, device)


def _apply_defaults(test_name: str, params: dict, scene) -> dict:
    """Fill in sensible defaults for test params that weren't provided."""
    if test_name == "background_substitution":
        params.setdefault("replacement", "gray")
    elif test_name == "object_relocation":
        if "target_object" not in params and scene:
            params["target_object"] = _pick_target(scene)
        params.setdefault("shift_pixels", [100, -80])
    elif test_name == "object_recolor":
        if "target_object" not in params and scene:
            params["target_object"] = _pick_target(scene)
        params.setdefault("hue_shift", 0.5)
    elif test_name == "occlusion_targeted":
        if "target_object" not in params and scene:
            params["target_object"] = _pick_target(scene)
        params.setdefault("fill", "gray")
    elif test_name == "distractor_insertion":
        params.setdefault("position", [256, 256])
        params.setdefault("distractor_size", 80)
    elif test_name == "task_string_swap":
        params.setdefault("replacement_task", "do nothing")
    elif test_name == "lighting_perturbation":
        params.setdefault("brightness_delta", 0.3)
        params.setdefault("contrast_delta", 0.3)
    elif test_name == "temporal_consistency":
        params.setdefault("perturbation_type", "background_substitution")
        params.setdefault("num_frames", 5)
    return params


def _pick_target(scene) -> str:
    """Pick the most likely manipulation target from scene objects."""
    skip = {"robot gripper", "robot arm", "gripper", "arm"}
    for obj in scene.objects:
        if obj.label.lower() not in skip and obj.mask is not None:
            return obj.label
    # Fallback to first object with a mask
    for obj in scene.objects:
        if obj.mask is not None:
            return obj.label
    return scene.objects[0].label if scene.objects else "object"


def _regenerate_viz(run_dir: str, test_name: str):
    """Regenerate only the visualization from an existing result.json."""
    import numpy as np

    cf_dir = os.path.join(run_dir, "diagnostic", "counterfactuals", test_name)
    result_path = os.path.join(cf_dir, "result.json")

    if not os.path.exists(result_path):
        print(f"ERROR: No result.json at {result_path}")
        print(f"  Run the test first (without --viz-only)")
        sys.exit(1)

    with open(result_path) as f:
        result_data = json.load(f)

    # For task_string_swap, regenerate the action delta chart
    if test_name == "task_string_swap":
        from smolvla_inspect.diagnostic.counterfactual import _make_action_delta_chart
        metrics = result_data.get("metrics", {})
        baseline = metrics.get("baseline_actions")
        modified = metrics.get("modified_actions")

        if baseline is None or modified is None:
            # Reconstruct from deltas (approximate — modified = baseline + delta)
            # but we don't have absolute values, so the grouped-bar top panel
            # won't render. Re-run the test without --viz-only instead.
            print("  WARNING: result.json does not contain baseline/modified actions.")
            print("  Re-run the test without --viz-only to get the two-panel chart.")
            print("  (Older results lack this data; only delta bars will be shown.)")
            return

        baseline = np.array(baseline)
        modified = np.array(modified)
        original_task = metrics.get("original_task", "original task")
        replacement_task = metrics.get("replacement_task", "replacement task")

        chart = _make_action_delta_chart(baseline, modified, original_task, replacement_task)
        from PIL import Image
        comp_path = os.path.join(cf_dir, "comparison.png")
        Image.fromarray(chart).save(comp_path)
        print(f"  Regenerated chart: {comp_path}")
    else:
        print(f"  --viz-only currently supports: task_string_swap")
        print(f"  For image-based tests, re-run the test (model needed for comparison).")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Structured data export — save intermediate arrays as .npz and metadata as JSON.

Called by cli.py when --export-data is enabled (default: true).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MANIFEST_VERSION = 1


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def create_run_dir(base_output_dir, run_name=None):
    """Create a named run directory and return its path."""
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
    return run_dir


def save_frames(run_dir, frames):
    """Save raw frame images as npz.  frames: list of (C,H,W) tensors."""
    frames_dir = os.path.join(run_dir, "data", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
        # Store as (H,W,C) uint8 for easy loading
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        np.savez_compressed(os.path.join(frames_dir, f"frame_{i:03d}.npz"), image=arr)


def save_self_attention(run_dir, heatmaps, per_head_data=None,
                        baseline_heatmap=None):
    """Save self-attention heatmaps and optional per-head / baseline data."""
    sa_dir = os.path.join(run_dir, "data", "self_attention")
    os.makedirs(sa_dir, exist_ok=True)

    # Main heatmaps — one per frame
    arrays = {f"heatmap_{i:03d}": hm.astype(np.float32) for i, hm in enumerate(heatmaps)}
    np.savez_compressed(os.path.join(sa_dir, "heatmaps.npz"), **arrays)

    # Per-head data (first frame only in current CLI)
    if per_head_data is not None:
        head_arrays = {}
        if "heads" in per_head_data:
            for j, h in enumerate(per_head_data["heads"]):
                arr = h.numpy() if hasattr(h, "numpy") else np.asarray(h)
                head_arrays[f"head_{j:02d}"] = arr.astype(np.float32)
        if "entropies" in per_head_data:
            ent = per_head_data["entropies"]
            head_arrays["entropies"] = (ent.numpy() if hasattr(ent, "numpy")
                                        else np.asarray(ent)).astype(np.float32)
        if head_arrays:
            np.savez_compressed(os.path.join(sa_dir, "per_head_frame0.npz"),
                                **head_arrays)

    # Positional baseline
    if baseline_heatmap is not None:
        np.savez_compressed(os.path.join(sa_dir, "positional_baseline.npz"),
                            baseline=baseline_heatmap.astype(np.float32))


def save_cross_attention(run_dir, heatmaps, per_step_data=None,
                         centroids=None):
    """Save cross-attention heatmaps and optional per-step / centroid data."""
    ca_dir = os.path.join(run_dir, "data", "cross_attention")
    os.makedirs(ca_dir, exist_ok=True)

    arrays = {f"heatmap_{i:03d}": hm.astype(np.float32)
              for i, hm in enumerate(heatmaps)}
    np.savez_compressed(os.path.join(ca_dir, "heatmaps.npz"), **arrays)

    if per_step_data is not None:
        step_arrays = {}
        for fi, frame_steps in enumerate(per_step_data):
            for si, step in enumerate(frame_steps):
                arr = step.numpy() if hasattr(step, "numpy") else np.asarray(step)
                step_arrays[f"frame{fi:03d}_step{si:03d}"] = arr.astype(np.float32)
        if step_arrays:
            np.savez_compressed(os.path.join(ca_dir, "per_step.npz"),
                                **step_arrays)

    if centroids is not None:
        np.savez_compressed(os.path.join(ca_dir, "centroids.npz"),
                            centroids=np.asarray(centroids, dtype=np.float32))


def save_gradient_data(run_dir, saliency_maps=None, gradcam_maps=None,
                       connector_maps=None, vlm_layer_results=None,
                       per_action_dim_maps=None, per_action_dim_mags=None,
                       language_diff_results=None,
                       vision_vs_state_results=None):
    """Save all gradient-based attribution data."""
    grad_dir = os.path.join(run_dir, "data", "gradient")
    os.makedirs(grad_dir, exist_ok=True)

    if saliency_maps:
        arrays = {f"saliency_{i:03d}": m.astype(np.float32)
                  for i, m in enumerate(saliency_maps)}
        np.savez_compressed(os.path.join(grad_dir, "saliency.npz"), **arrays)

    if gradcam_maps:
        arrays = {f"gradcam_{i:03d}": m.astype(np.float32)
                  for i, m in enumerate(gradcam_maps)}
        np.savez_compressed(os.path.join(grad_dir, "gradcam_siglip.npz"),
                            **arrays)

    if connector_maps:
        arrays = {f"connector_{i:03d}": m.astype(np.float32)
                  for i, m in enumerate(connector_maps)}
        np.savez_compressed(os.path.join(grad_dir, "gradcam_connector.npz"),
                            **arrays)

    if vlm_layer_results:
        arrays = {}
        # vlm_layer_results is {layer_idx: [list of per-frame dicts]}
        for layer_idx, frame_list in vlm_layer_results.items():
            for fi, layer_data in enumerate(frame_list):
                if layer_data is None:
                    continue
                for modality, data in layer_data.items():
                    if data is None:
                        continue
                    arr = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
                    arrays[f"frame{fi:03d}_layer{layer_idx:02d}_{modality}"] = arr.astype(np.float32)
        if arrays:
            np.savez_compressed(os.path.join(grad_dir, "vlm_layers.npz"),
                                **arrays)

    if per_action_dim_maps:
        arrays = {}
        for fi, frame_dims in enumerate(per_action_dim_maps):
            if frame_dims is None:
                continue
            for di, dim_map in enumerate(frame_dims):
                if dim_map is not None:
                    arr = dim_map.numpy() if hasattr(dim_map, "numpy") else np.asarray(dim_map)
                    arrays[f"frame{fi:03d}_dim{di:02d}"] = arr.astype(np.float32)
        if arrays:
            np.savez_compressed(os.path.join(grad_dir, "per_action_dim.npz"),
                                **arrays)
        if per_action_dim_mags:
            mag_arrays = {}
            for fi, mags in enumerate(per_action_dim_mags):
                if mags is not None:
                    arr = np.asarray(mags, dtype=np.float32)
                    mag_arrays[f"magnitudes_{fi:03d}"] = arr
            if mag_arrays:
                np.savez_compressed(
                    os.path.join(grad_dir, "per_action_dim_magnitudes.npz"),
                    **mag_arrays)

    if language_diff_results:
        arrays = {}
        for fi, result in enumerate(language_diff_results):
            if result is None:
                continue
            for key in ("original", "alternative", "diff"):
                if key in result:
                    arr = result[key]
                    if hasattr(arr, "numpy"):
                        arr = arr.numpy()
                    arrays[f"frame{fi:03d}_{key}"] = np.asarray(arr, dtype=np.float32)
        if arrays:
            np.savez_compressed(os.path.join(grad_dir, "language_diff.npz"),
                                **arrays)

    if vision_vs_state_results:
        vs_data = []
        for r in vision_vs_state_results:
            if r is not None:
                vs_data.append({
                    "vision_norm": float(r["vision_norm"]),
                    "state_norm": float(r["state_norm"]),
                    "vision_share": float(r["vision_share"]),
                })
            else:
                vs_data.append(None)
        with open(os.path.join(grad_dir, "vision_vs_state.json"), "w") as f:
            json.dump(vs_data, f, indent=2)


def save_health_data(run_dir, health_results):
    """Save model health report data as JSON files."""
    health_dir = os.path.join(run_dir, "data", "health")
    os.makedirs(health_dir, exist_ok=True)

    if "weightwatcher" in health_results:
        with open(os.path.join(health_dir, "weightwatcher.json"), "w") as f:
            json.dump(health_results["weightwatcher"], f, indent=2)

    if "entropy" in health_results:
        with open(os.path.join(health_dir, "entropy.json"), "w") as f:
            json.dump(health_results["entropy"], f, indent=2)

    if "redundancy" in health_results:
        with open(os.path.join(health_dir, "redundancy.json"), "w") as f:
            json.dump(health_results["redundancy"], f, indent=2)


def build_manifest(run_dir, args, model_info=None, dataset_info=None,
                   available_viz=None, image_paths=None):
    """Build and save run_manifest.json."""
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cli_args": {k: v for k, v in vars(args).items()
                     if not k.startswith("_")},
        "model_info": model_info or {},
        "dataset_info": dataset_info or {},
        "available_visualizations": available_viz or {},
        "images": image_paths or [],
    }

    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def collect_model_info(policy):
    """Extract model metadata for the manifest."""
    info = {}
    try:
        info["model_id"] = getattr(policy.config, "pretrained_model_name_or_path",
                                   str(type(policy).__name__))
    except Exception:
        info["model_id"] = str(type(policy).__name__)

    # SigLIP vision encoder info
    try:
        ve = policy.model.vlm_with_expert.vlm.model.vision_model
        vc = ve.config
        info["vision_encoder"] = {
            "image_size": getattr(vc, "image_size", None),
            "patch_size": getattr(vc, "patch_size", None),
            "num_heads": getattr(vc, "num_attention_heads", None),
            "num_layers": getattr(vc, "num_hidden_layers", None),
        }
        if info["vision_encoder"]["image_size"] and info["vision_encoder"]["patch_size"]:
            gs = info["vision_encoder"]["image_size"] // info["vision_encoder"]["patch_size"]
            info["vision_encoder"]["grid_size"] = gs
            info["vision_encoder"]["num_patches"] = gs * gs
    except Exception:
        pass

    # VLM info
    try:
        vlm = policy.model.vlm_with_expert
        tc = vlm.get_vlm_model().text_model.config
        info["vlm"] = {
            "num_heads": getattr(tc, "num_attention_heads", None),
            "num_layers": getattr(tc, "num_hidden_layers", None),
        }
    except Exception:
        pass

    # Expert info
    try:
        ec = policy.model.vlm_with_expert.lm_expert.config
        info["expert"] = {
            "num_heads": getattr(ec, "num_attention_heads", None),
            "num_layers": getattr(ec, "num_hidden_layers", None),
        }
    except Exception:
        pass

    # Action dimensions
    try:
        info["action_dim"] = policy.config.output_shapes["action"][0]
    except Exception:
        pass

    # Total params
    try:
        info["total_params_m"] = round(
            sum(p.numel() for p in policy.parameters()) / 1e6, 1)
    except Exception:
        pass

    return info


def collect_dataset_info(dataset, episode_idx, num_frames, task_str,
                         action_dim_names=None, image_keys=None):
    """Extract dataset metadata for the manifest."""
    info = {
        "dataset_id": getattr(dataset, "repo_id", str(type(dataset).__name__)),
        "episode_idx": episode_idx,
        "num_frames": num_frames,
        "task_string": task_str,
    }
    try:
        info["total_frames"] = len(dataset)
    except Exception:
        pass

    if action_dim_names:
        info["action_dim_names"] = action_dim_names

    if image_keys:
        info["image_keys"] = image_keys

    return info


def collect_image_paths(run_dir):
    """Scan the images/ subdirectory for all generated PNGs."""
    images_dir = os.path.join(run_dir, "images")
    if not os.path.isdir(images_dir):
        return []
    paths = []
    for root, _dirs, files in os.walk(images_dir):
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                rel = os.path.relpath(os.path.join(root, f), run_dir)
                paths.append(rel)
    return paths


def build_available_viz(args, heatmaps=None, cross_attn_heatmaps=None,
                        saliency_maps=None, gradcam_maps=None,
                        connector_maps=None, vlm_layer_results=None,
                        per_step_cross_attn=None, per_action_dim_maps=None,
                        language_diff_results=None,
                        vision_vs_state_results=None, health_ran=False):
    """Build a boolean map of which viz types exist in this run."""
    viz = {
        "self_attention": heatmaps is not None and len(heatmaps) > 0,
        "cross_attention": (cross_attn_heatmaps is not None
                            and len(cross_attn_heatmaps) > 0),
        "per_head": getattr(args, "show_heads", False),
        "per_step_cross_attention": (per_step_cross_attn is not None
                                     and any(len(s) > 0 for s in per_step_cross_attn)),
        "saliency": saliency_maps is not None and len(saliency_maps) > 0,
        "gradcam_siglip": gradcam_maps is not None and len(gradcam_maps) > 0,
        "gradcam_connector": (connector_maps is not None
                              and len(connector_maps) > 0),
        "gradcam_vlm_layers": vlm_layer_results is not None and len(vlm_layer_results) > 0,
        "per_action_dim": (per_action_dim_maps is not None
                           and any(m is not None for m in per_action_dim_maps)),
        "language_diff": (language_diff_results is not None
                          and any(r is not None for r in language_diff_results)),
        "vision_vs_state": (vision_vs_state_results is not None
                            and len(vision_vs_state_results) > 0),
        "model_health": health_ran,
    }
    return viz

"""
CLI entry point — extract_attention_maps, gradient_attention_map, load_defaults, main.

Imports from all other package modules.
"""

import argparse
import logging
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from ._compat import resize_with_pad
from .capture import SigLIPAttentionCapture, ActionVisionAttentionCapture
from .heatmap import (
    compute_padding_patches,
    attention_to_heatmap,
    compute_patch_attention_scores,
    compute_attention_rollout,
    compute_positional_baseline,
)
from .data import (
    find_vision_encoder,
    find_image_keys,
    _resolve_task_string,
    build_policy_batch_from_sample,
    get_episode_frames,
    parse_image_map,
)
from .viz import (
    create_visualization_grid,
    save_individual_frames,
    create_per_head_grid,
    create_per_step_cross_attn_grid,
    create_vlm_layer_grid,
    create_per_action_dim_grid,
    create_language_diff_grid,
    create_vision_vs_state_chart,
    overlay_heatmap,
)
from .health import run_model_health_report
from .gradient import (
    compute_gradient_maps,
    compute_saliency_map,
    compute_gradcam_connector_maps,
    compute_gradcam_vlm_layers_maps,
    compute_vision_vs_state_maps,
    compute_per_action_dim_maps,
    compute_language_conditional_maps,
)
from .export import (
    create_run_dir,
    save_frames,
    save_self_attention,
    save_cross_attention,
    save_gradient_data,
    save_health_data,
    build_manifest,
    collect_model_info,
    collect_dataset_info,
    collect_image_paths,
    build_available_viz,
)


def extract_attention_maps(policy, dataset, episode_idx=0, num_frames=8,
                           image_key=None, device="cpu",
                           method="last-layer", cross_attention=False,
                           show_heads=False, output_dir="./outputs",
                           raw_attention=False, attn_threshold=0.5,
                           task_override=None, per_step_cross_attention=False,
                           image_map=None):
    """
    Core function: Run inference and extract attention heatmaps.

    Args:
        method: ``"last-layer"`` (default), ``"rollout"``, or
                ``"all-layers"`` — how to aggregate SigLIP self-attention.
        cross_attention: If *True* also capture action-expert → vision
                         cross-attention (requires full ``select_action``
                         forward pass, slower).
        show_heads: If *True* save a per-head grid for the first frame.
        output_dir: Where to save per-head grids (only used when
                    *show_heads* is True).
        raw_attention: If *True* skip positional baseline subtraction
                       (show raw, uncorrected attention).
        attn_threshold: Percentile (0–1) below which attention values are
                        zeroed to suppress residual positional noise.
        task_override: If not *None*, use this string as the language
                       instruction instead of the one from the dataset.
        per_step_cross_attention: If *True*, also return per-denoising-step
                                  cross-attention maps.

    Returns:
        frames: list of image tensors (C, H, W)
        heatmaps: list of numpy heatmaps (H, W) in [0, 1]
        actions: list of predicted actions (or None)
        cross_attn_heatmaps: list of numpy heatmaps or *None*
        per_step_cross_attn: list of (list of Tensors per step) per frame, or *None*
    """

    # --- Find vision encoder ---
    print("\n[1/4] Locating vision encoder...")
    vision_encoder = find_vision_encoder(policy)

    if vision_encoder is None:
        print("  Could not find vision encoder. Dumping model structure:")
        for name, module in policy.named_modules():
            print(f"    {name}: {type(module).__name__}")
        raise RuntimeError("Cannot find SigLIP vision encoder in model")

    # --- Force eager attention so we get attention weights (SDPA/Flash return None) ---
    eager_count = 0
    for mod in vision_encoder.modules():
        if getattr(mod, "config", None) is not None and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "eager"
            eager_count += 1
    if eager_count:
        print("  Using eager attention to capture weights")

    # --- Register attention hooks ---
    print("\n[2/4] Registering attention hooks...")
    attn_capture = SigLIPAttentionCapture()
    attn_capture.register_hooks(vision_encoder)

    # --- Optionally register cross-attention capture ---
    cross_capture = None
    vision_start = vision_end = n_vision_tokens = 0
    if cross_attention:
        try:
            vlm_with_expert = policy.model.vlm_with_expert
            cross_capture = ActionVisionAttentionCapture()
            cross_capture.register(vlm_with_expert)
            vision_start, vision_end, n_vision_tokens = cross_capture.get_vision_token_range(policy)
            print(f"  Cross-attention capture enabled (vision tokens {vision_start}..{vision_end}, "
                  f"n={n_vision_tokens})")
        except Exception as e:
            print(f"  WARNING: Could not set up cross-attention capture: {e}")
            cross_capture = None

    # --- Find image key in dataset ---
    if image_key is None:
        image_keys = find_image_keys(dataset)
        if not image_keys:
            raise ValueError("No image keys found in dataset. Available keys: " +
                           str(list(dataset[0].keys())))
        image_key = image_keys[0]
        print(f"  Using image key: {image_key}")
        if len(image_keys) > 1:
            print(f"  Other available image keys: {image_keys[1:]}")

    # --- Get frames ---
    print(f"\n[3/4] Loading {num_frames} frames from episode {episode_idx}...")
    frame_pairs = get_episode_frames(dataset, episode_idx, num_frames, image_key)

    # --- Log task string being used ---
    first_sample = dataset[frame_pairs[0][0]]
    task_str = _resolve_task_string(first_sample, dataset, task_override=task_override)
    print(f"  Task string: \"{task_str}\"")

    # --- Compute positional baseline (once, after frames are loaded so we
    #     know the input aspect ratio for a properly padded baseline) ---
    first_img = frame_pairs[0][1]  # (C, H, W)
    input_hw = (first_img.shape[1], first_img.shape[2])

    baseline_scores = None
    per_head_baseline = None
    if not raw_attention:
        baseline_scores, per_head_baseline = compute_positional_baseline(
            vision_encoder, attn_capture, device, method, input_hw=input_hw,
        )
        if baseline_scores is not None:
            print("  Positional baseline computed (subtracting to reveal content-dependent attention)")
        else:
            print("  Could not compute positional baseline, using raw attention")

    # --- Compute padding-patch crop so heatmaps exclude pad regions ---
    target_size = getattr(getattr(vision_encoder, "config", None), "image_size", None) or 384
    patch_size_cfg = getattr(vision_encoder, "patch_size", None) or getattr(
        getattr(vision_encoder, "config", None), "patch_size", 14)
    content_crop = compute_padding_patches(input_hw, target_size, patch_size_cfg)
    if content_crop != (0, 0):
        print(f"  Padding crop: {content_crop[0]} top rows, {content_crop[1]} left cols of patches")

    # --- Save positional baseline diagnostic heatmap ---
    if baseline_scores is not None:
        n_bl = baseline_scores.shape[0]
        bl_side = int(math.sqrt(n_bl))
        grid_h_bl = target_size // patch_size_cfg if target_size and patch_size_cfg else bl_side
        grid_w_bl = grid_h_bl
        img_h_bl, img_w_bl = first_img.shape[1], first_img.shape[2]
        bl_heatmap = attention_to_heatmap(
            baseline_scores, (grid_h_bl, grid_w_bl), (img_h_bl, img_w_bl),
            content_crop=content_crop,
        )
        bl_path = os.path.join(output_dir, "positional_baseline.png")
        fig_bl, ax_bl = plt.subplots(figsize=(6, 6))
        ax_bl.imshow(bl_heatmap, cmap="jet")
        ax_bl.set_title("Positional baseline (gray-image attention)", fontsize=11)
        ax_bl.axis("off")
        fig_bl.savefig(bl_path, dpi=100, bbox_inches="tight")
        plt.close(fig_bl)
        print(f"  Saved positional baseline diagnostic: {bl_path}")

    # --- Run inference and collect attention ---
    print(f"\n[4/4] Running forward passes and extracting attention (method={method})...")
    frames = []
    heatmaps = []
    cross_attn_heatmaps = [] if cross_capture else None
    per_step_cross_attn = [] if (per_step_cross_attention and cross_capture) else None
    actions = []

    policy.eval()

    # We need a full policy forward pass when cross-attention is requested
    # (the vision encoder hooks still fire during select_action too)
    use_full_forward = cross_attention and cross_capture is not None

    raw_heads_attn = None   # populated when show_heads is True (first frame only)
    n_patches_h = n_patches_w = 0  # set during vision-encoder-only forward

    for i, (frame_idx, img_tensor) in enumerate(tqdm(frame_pairs, desc="Attention extraction", unit="frame")):
        frames.append(img_tensor.clone())

        # Reset captured attention maps
        attn_capture.reset_maps()
        if cross_capture:
            cross_capture.reset_maps()

        sample = dataset[frame_idx]

        # ----- forward pass -----
        try:
            with torch.no_grad():
                if use_full_forward:
                    # Full policy forward — needed for cross-attention capture.
                    # Reset the action queue so each frame triggers a real
                    # forward pass (otherwise cached actions are returned).
                    policy.reset()
                    batch, _ = build_policy_batch_from_sample(
                        sample, policy, device, batch_size=1,
                        image_key_for_grad=None, dataset=dataset,
                        task_override=task_override, image_map=image_map,
                    )
                    try:
                        policy.select_action(batch)
                    except Exception as e:
                        print(f"    select_action error: {e}")
                else:
                    # Vision-encoder-only forward (faster)
                    img = img_tensor.unsqueeze(0).to(device)
                    if img.max() > 1.0:
                        img = img.float() / 255.0

                    target_size = getattr(
                        getattr(vision_encoder, "config", None), "image_size", None,
                    ) or getattr(vision_encoder, "image_size", 384)
                    if img.shape[-1] != target_size or img.shape[-2] != target_size:
                        img_resized = resize_with_pad(img, target_size, target_size, pad_value=0)
                    else:
                        img_resized = img
                    img_resized = img_resized * 2.0 - 1.0  # normalize to [-1, 1] matching SigLIP

                    try:
                        enc_dtype = next(vision_encoder.parameters()).dtype
                        img_resized = img_resized.to(enc_dtype)
                    except StopIteration:
                        pass

                    patch_size = getattr(vision_encoder, "patch_size", None) or getattr(
                        getattr(vision_encoder, "config", None), "patch_size", 14
                    )
                    n_patches_h = img_resized.size(2) // patch_size
                    n_patches_w = img_resized.size(3) // patch_size
                    patch_mask = torch.ones(1, n_patches_h, n_patches_w, dtype=torch.bool, device=device)

                    try:
                        if hasattr(vision_encoder, 'embeddings') and hasattr(vision_encoder, 'encoder'):
                            embeddings = vision_encoder.embeddings(img_resized, patch_mask)
                            encoder_out = vision_encoder.encoder(embeddings)
                        elif hasattr(vision_encoder, 'forward'):
                            encoder_out = vision_encoder(img_resized)
                        else:
                            encoder_out = vision_encoder(pixel_values=img_resized,
                                                          patch_attention_mask=patch_mask)
                    except Exception as e:
                        print(f"    Direct vision forward failed ({e}), trying full policy...")
                        batch, _ = build_policy_batch_from_sample(
                            sample, policy, device, batch_size=1,
                            image_key_for_grad=None, dataset=dataset,
                            task_override=task_override, image_map=image_map,
                        )
                        try:
                            policy.select_action(batch)
                        except Exception:
                            pass

        except Exception as e:
            print(f"    Frame {i} forward pass error: {e}")

        # ----- per-head capture (independent of summary method) -----
        if show_heads and i == 0 and raw_heads_attn is None:
            attn_for_heads = attn_capture.get_last_layer_attention()
            if attn_for_heads is not None:
                while attn_for_heads.dim() > 3:
                    attn_for_heads = attn_for_heads[0]
                if attn_for_heads.dim() == 3:
                    raw_heads_attn = attn_for_heads.clone()

        # ----- self-attention heatmap -----
        if method == "rollout":
            all_layers = attn_capture.get_all_layer_attentions()
            rollout_mat = compute_attention_rollout(all_layers)
            if rollout_mat is not None:
                patch_scores = rollout_mat.mean(dim=0)  # per-patch importance
            else:
                patch_scores = None
        elif method == "all-layers":
            # For "all-layers" we still produce a single summary heatmap
            # (averaging last-layer scores) but also save individual layer
            # grids elsewhere; here fall through to last-layer for the
            # summary heatmap.
            attn = attn_capture.get_last_layer_attention()
            patch_scores = None
            if attn is not None:
                while attn.dim() > 3:
                    attn = attn[0]
                if attn.dim() == 3:
                    attn = attn.mean(dim=0)
                patch_scores = compute_patch_attention_scores(attn, method="mean")
        else:
            # "last-layer" (default)
            attn = attn_capture.get_last_layer_attention()
            patch_scores = None
            if attn is not None:
                while attn.dim() > 3:
                    attn = attn[0]
                if attn.dim() == 3:
                    attn = attn.mean(dim=0)
                patch_scores = compute_patch_attention_scores(attn, method="mean")

        # Subtract positional baseline to isolate content-dependent signal
        if patch_scores is not None and baseline_scores is not None:
            patch_scores = torch.clamp(patch_scores - baseline_scores, min=0)

        if patch_scores is not None:
            n_patches = patch_scores.shape[0]
            grid_side = int(math.sqrt(n_patches))
            if grid_side * grid_side != n_patches:
                grid_h = n_patches_h if n_patches_h > 0 else grid_side
                grid_w = n_patches_w if n_patches_w > 0 else grid_side
            else:
                grid_h = grid_w = grid_side

            img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
            heatmap = attention_to_heatmap(patch_scores, (grid_h, grid_w), (img_h, img_w),
                                         content_crop=content_crop,
                                         threshold_pct=attn_threshold)
            heatmaps.append(heatmap)

            print(f"    Frame {i}: {n_patches} patches \u2192 "
                  f"{grid_h}x{grid_w} grid \u2192 {img_h}x{img_w} heatmap ({method})")

            # Per-head grid for first frame
            if show_heads and i == 0 and raw_heads_attn is not None:
                head_path = os.path.join(output_dir, f"per_head_ep{episode_idx:03d}.png")
                create_per_head_grid(
                    img_tensor, raw_heads_attn,
                    (grid_h, grid_w), (img_h, img_w),
                    output_path=head_path,
                    content_crop=content_crop,
                    baseline_per_head=per_head_baseline if not raw_attention else None,
                    threshold_pct=attn_threshold,
                )
                raw_heads_attn = None  # only once
        else:
            print(f"    Frame {i}: No attention captured, using uniform heatmap")
            img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
            heatmaps.append(np.ones((img_h, img_w)) * 0.5)

        # ----- cross-attention heatmap -----
        if cross_capture is not None:
            cross_scores = cross_capture.get_mean_cross_attention(vision_start, vision_end)
            if cross_scores is not None:
                # cross_scores: (n_vision_tokens,)
                # Vision tokens come from SigLIP → connector (pixel shuffle).
                # After pixel shuffle the spatial grid is halved in each dim.
                n_vis = cross_scores.shape[0]
                cs_side = int(math.sqrt(n_vis))
                if cs_side * cs_side != n_vis:
                    cs_h = cs_w = cs_side
                else:
                    cs_h = cs_w = cs_side

                img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
                cross_hm = attention_to_heatmap(cross_scores, (cs_h, cs_w), (img_h, img_w),
                                               threshold_pct=attn_threshold)
                cross_attn_heatmaps.append(cross_hm)
                print(f"    Frame {i}: Cross-attention captured ({n_vis} vision tokens)")
            else:
                img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
                cross_attn_heatmaps.append(np.ones((img_h, img_w)) * 0.5)
                print(f"    Frame {i}: No cross-attention captured, using uniform")

        # ----- per-step cross-attention -----
        if per_step_cross_attn is not None and cross_capture is not None:
            step_maps = cross_capture.get_per_step_cross_attention(
                vision_start, vision_end,
            )
            if step_maps is not None:
                per_step_cross_attn.append(step_maps)
                print(f"    Frame {i}: Per-step cross-attention: {len(step_maps)} steps")
            else:
                per_step_cross_attn.append([])

    # Cleanup
    attn_capture.clear()
    if cross_capture:
        cross_capture.clear()

    return frames, heatmaps, actions, cross_attn_heatmaps, per_step_cross_attn


def gradient_attention_map(policy, dataset, frame_idx, image_key, device="cpu",
                           task_override=None, image_map=None):
    """
    Compute input-gradient saliency map as a fallback.

    Delegates to :func:`gradient.compute_saliency_map` which bypasses the
    ``@torch.no_grad()`` on ``select_action()`` by calling internal model
    methods directly under ``torch.enable_grad()``.
    """
    sample = dataset[frame_idx]
    return compute_saliency_map(
        policy, sample, dataset, image_key, device,
        task_override=task_override, image_map=image_map,
    )


def load_defaults(config_path=None):
    """Load defaults from a YAML config file.

    Resolution order:
      1. Explicit *config_path* argument (from ``--config``)
      2. ``configs/defaults.yaml`` in the project root
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "defaults.yaml"
    else:
        config_path = Path(config_path)
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    elif config_path != Path(__file__).resolve().parent.parent / "configs" / "defaults.yaml":
        # Only error if the user explicitly asked for a config that doesn't exist
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    return {}


def main():
    # Pre-parse --config so we can load defaults before building the full parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="Path to YAML config file (default: configs/defaults.yaml)")
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="See what SmolVLA's vision encoder is looking at.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_attention.py
  python inspect_attention.py --episode 3 --num-frames 12 --device cuda
  python inspect_attention.py --model path/to/finetuned_checkpoint
  python inspect_attention.py --config configs/gpu.yaml
        """
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (default: configs/defaults.yaml)")
    parser.add_argument("--model", type=str,
                        default=defaults.get("model", "lerobot/smolvla_base"))
    parser.add_argument("--dataset", type=str,
                        default=defaults.get("dataset", "lerobot/svla_so101_pickplace"))
    parser.add_argument("--episode", type=int,
                        default=defaults.get("episode", 0))
    parser.add_argument("--num-frames", type=int,
                        default=defaults.get("num_frames", 8))
    parser.add_argument("--image-key", type=str, default=None)
    parser.add_argument("--image-map", type=str, default=None,
                        help="Explicit dataset→policy image key mapping. "
                             "Comma-separated pairs using = delimiter. "
                             "Accepts full keys or short suffixes. "
                             "Example: front=camera2,side=camera3")
    parser.add_argument("--task", type=str, default=defaults.get("task", None),
                        help="Override the task/language instruction (default: from dataset)")
    parser.add_argument("--output-dir", type=str,
                        default=defaults.get("output_dir", "./outputs"))
    parser.add_argument("--device", type=str,
                        default=defaults.get("device", "auto"),
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--save-individual", action="store_true",
                        default=defaults.get("save_individual", False))
    parser.add_argument("--export-data", action="store_true",
                        default=defaults.get("export_data", True),
                        help="Save structured .npz + JSON alongside PNGs (default: true)")
    parser.add_argument("--no-export-data", action="store_false", dest="export_data",
                        help="Disable structured data export")
    parser.add_argument("--run-name", type=str,
                        default=defaults.get("run_name", None),
                        help="Run folder name (default: timestamped run_YYYY-MM-DD_HH-MM-SS)")
    parser.add_argument("--method", type=str,
                        default=defaults.get("method", "last-layer"),
                        choices=["last-layer", "rollout", "all-layers"],
                        help="Self-attention aggregation method")
    parser.add_argument("--cross-attention", action="store_true",
                        default=defaults.get("cross_attention", False),
                        help="Capture action-expert \u2192 vision cross-attention (slower)")
    parser.add_argument("--show-heads", action="store_true",
                        default=defaults.get("show_heads", False),
                        help="Save a per-head attention grid for the first frame")
    parser.add_argument("--raw-attention", action="store_true",
                        default=defaults.get("raw_attention", False),
                        help="Skip positional baseline subtraction (show raw attention)")
    parser.add_argument("--attn-threshold", type=float,
                        default=defaults.get("attn_threshold", 0.5),
                        help="Percentile threshold (0-1) below which attention values are zeroed")

    # Model health diagnostics
    parser.add_argument("--model-health", action="store_true",
                        default=defaults.get("model_health", False),
                        help="Run health diagnostics instead of attention heatmaps")
    parser.add_argument("--health-frames", type=int,
                        default=defaults.get("health_frames", 5),
                        help="Number of sample frames for entropy/redundancy (default: 5)")
    parser.add_argument("--entropy-warn", type=float,
                        default=defaults.get("entropy_warn", 0.8),
                        help="Entropy ratio threshold for 'unfocused' warning (default: 0.8)")
    parser.add_argument("--entropy-critical", type=float,
                        default=defaults.get("entropy_critical", 0.95),
                        help="Entropy ratio threshold for 'uniform/dead' (default: 0.95)")
    parser.add_argument("--entropy-low", type=float,
                        default=defaults.get("entropy_low", 0.1),
                        help="Entropy ratio threshold for 'collapsed' (default: 0.1)")
    parser.add_argument("--redundancy-warn", type=float,
                        default=defaults.get("redundancy_warn", 0.7),
                        help="Cosine similarity threshold for 'high redundancy' (default: 0.7)")
    parser.add_argument("--redundancy-critical", type=float,
                        default=defaults.get("redundancy_critical", 0.9),
                        help="Cosine similarity threshold for 'collapsed' (default: 0.9)")

    # Gradient-based attribution
    parser.add_argument("--gradient", nargs="?", const="both",
                        default=defaults.get("gradient", None),
                        choices=["saliency", "gradcam", "both"],
                        help="Gradient attribution method (default: off; bare --gradient means 'both')")
    parser.add_argument("--gradient-device", type=str,
                        default=defaults.get("gradient_device", None),
                        choices=["cpu", "cuda", "mps"],
                        help="Device for gradient attribution (default: same as --device)")
    parser.add_argument("--gradient-seed", type=int,
                        default=defaults.get("gradient_seed", 42),
                        help="Fixed noise seed for reproducible gradient attribution (default: 42)")
    parser.add_argument("--smooth-grad", type=int,
                        default=defaults.get("smooth_grad", 1),
                        help="SmoothGrad samples for saliency (1 = vanilla, >1 = averaged over N noisy inputs)")
    parser.add_argument("--smooth-grad-sigma", type=float,
                        default=defaults.get("smooth_grad_sigma", 0.15),
                        help="Gaussian noise std for SmoothGrad (default: 0.15)")

    # Extended attribution features
    parser.add_argument("--per-step-cross-attention", action="store_true",
                        default=defaults.get("per_step_cross_attention", False),
                        help="Visualize cross-attention at each denoising step")
    parser.add_argument("--gradcam-connector", action="store_true",
                        default=defaults.get("gradcam_connector", False),
                        help="GradCAM on VLM connector output (post-pixel-shuffle)")
    parser.add_argument("--gradcam-vlm-layers", nargs="?", const="4,8,12,16",
                        default=defaults.get("gradcam_vlm_layers", None),
                        help="GradCAM on VLM intermediate layers (comma-separated 1-indexed, default: 4,8,12,16)")
    parser.add_argument("--vision-vs-state", action="store_true",
                        default=defaults.get("vision_vs_state", False),
                        help="Compare gradient attribution between vision and state inputs")
    parser.add_argument("--per-action-dim", action="store_true",
                        default=defaults.get("per_action_dim", False),
                        help="Per-action-dimension GradCAM (uses retain_graph — GPU recommended)")
    parser.add_argument("--language-diff", nargs="?", const="auto",
                        default=defaults.get("language_diff", None),
                        help="Language-conditional comparison (auto-select or provide alt task string)")

    args = parser.parse_args()

    # Auto-enable cross-attention if per-step is requested
    if args.per_step_cross_attention and not args.cross_attention:
        args.cross_attention = True
        print("  NOTE: --per-step-cross-attention auto-enables --cross-attention")

    # Validate language-diff requires gradient gradcam
    if args.language_diff and not args.gradient:
        args.gradient = "gradcam"
        print("  NOTE: --language-diff auto-enables --gradient gradcam")

    # --- Set up run directory ---
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)

    if args.export_data:
        run_dir = create_run_dir(base_output_dir, args.run_name)
        images_dir = os.path.join(run_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        # Redirect output_dir so all PNGs go into images/ under the run folder
        args.output_dir = images_dir
    else:
        run_dir = None

    # Auto-detect device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)

    # Resolve gradient device (defaults to main device)
    grad_device_str = args.gradient_device or args.device
    grad_device = torch.device(grad_device_str)

    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SmolVLA Attention Visualizer")
    print("=" * 70)
    print(f"  Device: {args.device}")
    if args.gradient and grad_device_str != args.device:
        print(f"  Gradient device: {grad_device_str}")

    # --- Load model ---
    print(f"\n[Step 1] Loading model: {args.model}")
    print("  This may download ~1GB on first run...")

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        # Suppress noisy warnings from HF/lerobot during model loading:
        #  - "Device 'cuda' is not available. Switching to 'mps'"
        #  - "`torch_dtype` is deprecated! Use `dtype` instead!"
        #  - "Loading ... weights ..."
        _suppressed_loggers = {
            name: logging.getLogger(name)
            for name in ("lerobot.configs.policies", "lerobot", "transformers")
        }
        _saved_levels = {name: lg.level for name, lg in _suppressed_loggers.items()}
        for lg in _suppressed_loggers.values():
            lg.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
                policy = SmolVLAPolicy.from_pretrained(args.model)
        finally:
            for name, lg in _suppressed_loggers.items():
                lg.setLevel(_saved_levels[name])
        policy.to(device)
        policy.eval()
        print(f"  Model loaded successfully ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params)")
    except ImportError:
        print("\n  ERROR: LeRobot not installed. Run:")
        print('    pip install "lerobot[smolvla]"')
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR loading model: {e}")
        print("  Make sure the model ID is correct and you have internet access.")
        sys.exit(1)

    # --- Load dataset ---
    print(f"\n[Step 2] Loading dataset: {args.dataset}")
    print("  This may download several GB on first run...")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        dataset = LeRobotDataset(args.dataset)
        print(f"  Dataset loaded: {len(dataset)} frames")

        # Show dataset info
        sample = dataset[0]
        print(f"  Available keys: {list(sample.keys())}")
        image_keys = find_image_keys(dataset)
        print(f"  Image keys found: {image_keys}")
    except Exception as e:
        print(f"\n  ERROR loading dataset: {e}")
        sys.exit(1)

    # --- Parse image map ---
    image_map = parse_image_map(getattr(args, "image_map", None))
    if image_map:
        print(f"\n  Image map overrides: {image_map}")

    # --- Model health mode (early exit) ---
    if args.model_health:
        run_model_health_report(policy, dataset, args)
        return

    # --- Extract attention maps ---
    print(f"\n[Step 3] Extracting attention maps...")

    cross_attn_heatmaps = None
    per_step_cross_attn = None
    try:
        frames, heatmaps, actions, cross_attn_heatmaps, per_step_cross_attn = extract_attention_maps(
            policy=policy,
            dataset=dataset,
            episode_idx=args.episode,
            num_frames=args.num_frames,
            image_key=args.image_key,
            device=args.device,
            method=args.method,
            cross_attention=args.cross_attention,
            show_heads=args.show_heads,
            output_dir=args.output_dir,
            raw_attention=args.raw_attention,
            attn_threshold=args.attn_threshold,
            task_override=args.task,
            per_step_cross_attention=args.per_step_cross_attention,
            image_map=image_map,
        )
    except Exception as e:
        print(f"\n  Attention extraction failed: {e}")
        print("  Falling back to input-gradient saliency maps...")

        # Fallback to gradient-based saliency
        image_key = args.image_key or find_image_keys(dataset)[0]
        frame_pairs = get_episode_frames(dataset, args.episode, args.num_frames, image_key)

        frames = []
        heatmaps = []
        for frame_idx, img_tensor in frame_pairs:
            frames.append(img_tensor)
            saliency = gradient_attention_map(policy, dataset, frame_idx, image_key, args.device,
                                                task_override=args.task, image_map=image_map)
            if saliency is not None:
                heatmaps.append(saliency)
            else:
                heatmaps.append(np.ones((img_tensor.shape[1], img_tensor.shape[2])) * 0.5)

        actions = []

    if not frames:
        print("\nERROR: No frames extracted. Check episode index and dataset.")
        sys.exit(1)

    # --- Export attention data ---
    if run_dir:
        save_frames(run_dir, frames)
        save_self_attention(run_dir, heatmaps)
        if cross_attn_heatmaps:
            save_cross_attention(run_dir, cross_attn_heatmaps,
                                per_step_data=per_step_cross_attn)

    # --- Gradient-based attribution (after attention hooks are cleaned up) ---
    saliency_maps = None
    gradcam_maps = None
    if args.gradient:
        image_key_for_grad = args.image_key or find_image_keys(dataset)[0]

        # Move model to gradient device if different from main device
        if grad_device != device:
            print(f"\n  Moving model from {device} to {grad_device} for gradient computation...")
            policy.to(grad_device)

        print(f"\n[Step 3b] Computing gradient attribution (method={args.gradient}, device={grad_device_str})...")
        saliency_maps, gradcam_maps = compute_gradient_maps(
            policy=policy,
            dataset=dataset,
            episode_idx=args.episode,
            num_frames=args.num_frames,
            image_key=image_key_for_grad,
            device=grad_device_str,
            method=args.gradient,
            noise_seed=args.gradient_seed,
            task_override=args.task,
            smooth_n=args.smooth_grad,
            smooth_sigma=args.smooth_grad_sigma,
            image_map=image_map,
        )
        sal_label = f"SmoothGrad (N={args.smooth_grad})" if args.smooth_grad > 1 else "Saliency"
        if saliency_maps:
            print(f"  {sal_label} maps: {len(saliency_maps)} frames")
        if gradcam_maps:
            print(f"  GradCAM maps: {len(gradcam_maps)} frames")

    # --- Extended attribution features ---
    connector_gradcam_maps = None
    vlm_layer_results = None
    vlm_layer_indices = None
    vision_vs_state_results = None
    per_action_dim_maps = None
    per_action_dim_mags = None
    language_diff_results = None
    action_dim_names = None

    has_extended = any([
        args.gradcam_connector,
        args.gradcam_vlm_layers,
        args.vision_vs_state,
        args.per_action_dim,
        args.language_diff,
    ])

    if has_extended:
        image_key_for_grad = args.image_key or find_image_keys(dataset)[0]

        # Ensure model is on gradient device
        if grad_device != device and not args.gradient:
            print(f"\n  Moving model from {device} to {grad_device} for gradient computation...")
            policy.to(grad_device)

        enabled_features = []
        if args.gradcam_connector:
            enabled_features.append("Connector GradCAM")
        if args.gradcam_vlm_layers:
            enabled_features.append("VLM layer GradCAM")
        if args.vision_vs_state:
            enabled_features.append("Vision vs State")
        if args.per_action_dim:
            enabled_features.append("Per-action-dim")
        if args.language_diff:
            enabled_features.append("Language diff")
        print(f"\n[Step 3c] Extended attribution ({len(enabled_features)} features: "
              f"{', '.join(enabled_features)})...")

        feat_idx = 0
        # F1: Connector GradCAM
        if args.gradcam_connector:
            feat_idx += 1
            print(f"\n  [{feat_idx}/{len(enabled_features)}] Connector GradCAM...")
            connector_gradcam_maps = compute_gradcam_connector_maps(
                policy=policy, dataset=dataset,
                episode_idx=args.episode, num_frames=args.num_frames,
                image_key=image_key_for_grad, device=grad_device_str,
                noise_seed=args.gradient_seed, task_override=args.task,
                image_map=image_map,
            )
            if connector_gradcam_maps:
                print(f"  Connector GradCAM: {len(connector_gradcam_maps)} frames")

        # F2: VLM layer GradCAM
        if args.gradcam_vlm_layers:
            feat_idx += 1
            layer_str = args.gradcam_vlm_layers
            vlm_layer_indices = [int(x.strip()) - 1 for x in layer_str.split(",")]

            # Auto-detect layer count and filter out-of-range indices
            try:
                text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
                num_layers = len(text_model.layers)
                invalid = [i + 1 for i in vlm_layer_indices if i >= num_layers]
                if invalid:
                    print(f"  WARNING: Model has {num_layers} VLM layers — "
                          f"skipping out-of-range layers {invalid} (1-indexed)")
                    vlm_layer_indices = [i for i in vlm_layer_indices if i < num_layers]
                if not vlm_layer_indices:
                    print(f"  WARNING: No valid VLM layers to compute, skipping")
                    vlm_layer_results = None
                    # skip past the compute call
            except AttributeError:
                pass

            if vlm_layer_indices:
                valid_str = ",".join(str(i + 1) for i in vlm_layer_indices)
                print(f"\n  [{feat_idx}/{len(enabled_features)}] VLM layer GradCAM (layers {valid_str})...")
                vlm_layer_results = compute_gradcam_vlm_layers_maps(
                    policy=policy, dataset=dataset,
                    episode_idx=args.episode, num_frames=args.num_frames,
                    image_key=image_key_for_grad, device=grad_device_str,
                    layer_indices=vlm_layer_indices,
                    noise_seed=args.gradient_seed, task_override=args.task,
                    image_map=image_map,
                )

        # F4: Vision vs. State
        if args.vision_vs_state:
            feat_idx += 1
            print(f"\n  [{feat_idx}/{len(enabled_features)}] Vision vs. state attribution...")
            vision_vs_state_results = compute_vision_vs_state_maps(
                policy=policy, dataset=dataset,
                episode_idx=args.episode, num_frames=args.num_frames,
                image_key=image_key_for_grad, device=grad_device_str,
                noise_seed=args.gradient_seed, task_override=args.task,
                image_map=image_map,
            )

        # F6: Per-action-dim GradCAM
        if args.per_action_dim:
            feat_idx += 1
            print(f"\n  [{feat_idx}/{len(enabled_features)}] Per-action-dim GradCAM (retain_graph — GPU recommended)...")
            try:
                action_dim_names = list(dataset.meta.names.get("action", []))
            except (AttributeError, TypeError):
                action_dim_names = None
            print(f"  Computing per-action-dim GradCAM...")
            per_action_dim_maps, per_action_dim_mags = compute_per_action_dim_maps(
                policy=policy, dataset=dataset,
                episode_idx=args.episode, num_frames=args.num_frames,
                image_key=image_key_for_grad, device=grad_device_str,
                noise_seed=args.gradient_seed, task_override=args.task,
                action_dim_names=action_dim_names,
                image_map=image_map,
            )

        # F5: Language-conditional comparison
        if args.language_diff:
            feat_idx += 1
            alt_task = None if args.language_diff == "auto" else args.language_diff
            print(f"\n  [{feat_idx}/{len(enabled_features)}] Language-conditional comparison...")
            language_diff_results = compute_language_conditional_maps(
                policy=policy, dataset=dataset,
                episode_idx=args.episode, num_frames=args.num_frames,
                image_key=image_key_for_grad, device=grad_device_str,
                noise_seed=args.gradient_seed, task_override=args.task,
                alt_task=alt_task, image_map=image_map,
            )

    # --- Export gradient data ---
    if run_dir and (saliency_maps or gradcam_maps or connector_gradcam_maps
                    or vlm_layer_results or per_action_dim_maps
                    or language_diff_results or vision_vs_state_results):
        save_gradient_data(
            run_dir,
            saliency_maps=saliency_maps,
            gradcam_maps=gradcam_maps,
            connector_maps=connector_gradcam_maps,
            vlm_layer_results=vlm_layer_results,
            per_action_dim_maps=per_action_dim_maps,
            per_action_dim_mags=per_action_dim_mags,
            language_diff_results=language_diff_results,
            vision_vs_state_results=vision_vs_state_results,
        )

    # --- Generate visualizations ---
    print(f"\n[Step 4] Generating visualizations...")

    grid_path = os.path.join(args.output_dir, f"episode_dashboard_ep{args.episode:03d}.png")
    create_visualization_grid(
        frames=frames,
        heatmaps=heatmaps,
        actions=actions,
        cross_attn_heatmaps=cross_attn_heatmaps,
        saliency_maps=saliency_maps,
        gradcam_maps=gradcam_maps,
        connector_gradcam_maps=connector_gradcam_maps,
        language_diff_maps=language_diff_results,
        episode_idx=args.episode,
        output_path=grid_path,
        smooth_n=args.smooth_grad,
    )

    # F3: Per-step cross-attention grid
    if per_step_cross_attn and any(len(s) > 0 for s in per_step_cross_attn):
        psc_path = os.path.join(args.output_dir, f"per_step_cross_attn_ep{args.episode:03d}.png")
        create_per_step_cross_attn_grid(
            frames=frames,
            per_step_maps=per_step_cross_attn,
            episode_idx=args.episode,
            output_path=psc_path,
        )

    # F2: VLM layer GradCAM grid
    if vlm_layer_results and vlm_layer_indices:
        vlm_path = os.path.join(args.output_dir, f"vlm_layers_ep{args.episode:03d}.png")
        # Extract language tokens for bar chart labels
        vlm_lang_tokens = None
        try:
            tokenizer = policy.model.vlm_with_expert.processor.tokenizer
            _img_key = args.image_key or find_image_keys(dataset)[0]
            first_sample = dataset[get_episode_frames(dataset, args.episode, 1, _img_key)[0][0]]
            task_str = _resolve_task_string(first_sample, dataset, task_override=args.task)
            token_ids = tokenizer.encode(task_str, add_special_tokens=False)
            vlm_lang_tokens = [tokenizer.decode([tid]) for tid in token_ids]
        except Exception:
            pass
        create_vlm_layer_grid(
            frames=frames,
            vlm_layer_results=vlm_layer_results,
            layer_indices=vlm_layer_indices,
            episode_idx=args.episode,
            output_path=vlm_path,
            lang_tokens=vlm_lang_tokens,
        )

    # F6: Per-action-dim GradCAM grid
    if per_action_dim_maps and any(m is not None for m in per_action_dim_maps):
        pad_path = os.path.join(args.output_dir, f"per_action_dim_ep{args.episode:03d}.png")
        create_per_action_dim_grid(
            frames=frames,
            per_dim_maps=per_action_dim_maps,
            action_dim_names=action_dim_names,
            episode_idx=args.episode,
            output_path=pad_path,
            magnitudes=per_action_dim_mags,
        )

    # F5: Language-conditional comparison grid
    if language_diff_results and any(r is not None for r in language_diff_results):
        ld_path = os.path.join(args.output_dir, f"language_diff_ep{args.episode:03d}.png")
        create_language_diff_grid(
            frames=frames,
            lang_diff_results=language_diff_results,
            episode_idx=args.episode,
            output_path=ld_path,
        )

    # F4: Vision vs. State — save text report
    if vision_vs_state_results:
        vs_path = os.path.join(args.output_dir, f"vision_vs_state_ep{args.episode:03d}.txt")
        lines = []
        v_total = s_total = 0
        n_valid = 0
        for fi, r in enumerate(vision_vs_state_results):
            if r is not None:
                lines.append(f"Frame {fi}: vision={r['vision_norm']:.2f} "
                             f"({r['vision_share']:.0%}), "
                             f"state={r['state_norm']:.2f} "
                             f"({1-r['vision_share']:.0%})")
                v_total += r['vision_share']
                s_total += 1 - r['vision_share']
                n_valid += 1
            else:
                lines.append(f"Frame {fi}: failed")
        if n_valid > 0:
            avg_v = v_total / n_valid
            lines.append(f"Average: {avg_v:.0%} vision / {1-avg_v:.0%} state")
        report = "\n".join(lines)
        with open(vs_path, "w") as f:
            f.write(report + "\n")
        print(f"\n  Vision vs. State report:\n    " + "\n    ".join(lines))
        print(f"  Saved: {vs_path}")

        # Stacked bar chart
        vs_chart_path = os.path.join(args.output_dir, f"vision_vs_state_ep{args.episode:03d}.png")
        create_vision_vs_state_chart(
            vision_vs_state_results=vision_vs_state_results,
            episode_idx=args.episode,
            output_path=vs_chart_path,
        )

    if args.save_individual:
        save_individual_frames(
            frames=frames,
            heatmaps=heatmaps,
            output_dir=os.path.join(args.output_dir, f"episode_{args.episode:03d}"),
            episode_idx=args.episode,
        )

    # --- Build and save manifest ---
    if run_dir:
        # Resolve task string for manifest
        _manifest_task = args.task
        if _manifest_task is None:
            try:
                _img_key = args.image_key or find_image_keys(dataset)[0]
                _first = dataset[get_episode_frames(dataset, args.episode, 1, _img_key)[0][0]]
                _manifest_task = _resolve_task_string(_first, dataset)
            except Exception:
                _manifest_task = ""
        try:
            _action_names = list(dataset.meta.names.get("action", []))
        except (AttributeError, TypeError):
            _action_names = None
        _image_keys = find_image_keys(dataset) if dataset else None

        model_info = collect_model_info(policy)
        dataset_info_dict = collect_dataset_info(
            dataset, args.episode, args.num_frames, _manifest_task,
            action_dim_names=_action_names, image_keys=_image_keys,
        )
        available_viz = build_available_viz(
            args,
            heatmaps=heatmaps,
            cross_attn_heatmaps=cross_attn_heatmaps,
            saliency_maps=saliency_maps,
            gradcam_maps=gradcam_maps,
            connector_maps=connector_gradcam_maps,
            vlm_layer_results=vlm_layer_results,
            per_step_cross_attn=per_step_cross_attn,
            per_action_dim_maps=per_action_dim_maps,
            language_diff_results=language_diff_results,
            vision_vs_state_results=vision_vs_state_results,
        )
        image_paths = collect_image_paths(run_dir)
        build_manifest(run_dir, args, model_info=model_info,
                       dataset_info=dataset_info_dict,
                       available_viz=available_viz,
                       image_paths=image_paths)
        print(f"\n  Run manifest saved: {os.path.join(run_dir, 'run_manifest.json')}")

    # --- Summary ---
    output_label = run_dir if run_dir else args.output_dir
    print(f"\n{'=' * 70}")
    print("DONE!")
    print(f"{'=' * 70}")
    print(f"\nOutputs saved to: {output_label}/")
    print(f"  Episode dashboard:  {grid_path}")
    if args.save_individual:
        print(f"  Individual frames:  {args.output_dir}/episode_{args.episode:03d}/")
    if per_step_cross_attn and any(len(s) > 0 for s in per_step_cross_attn):
        print(f"  Per-step cross-attn: per_step_cross_attn_ep{args.episode:03d}.png")
        print(f"  Centroid trajectory: per_step_cross_attn_ep{args.episode:03d}_trajectory.png")
    if vlm_layer_results and vlm_layer_indices:
        print(f"  VLM layer GradCAM:   vlm_layers_ep{args.episode:03d}.png")
        print(f"  Lang token attrib:   vlm_layers_ep{args.episode:03d}_lang_tokens.png")
    if per_action_dim_maps and any(m is not None for m in per_action_dim_maps):
        print(f"  Per-action-dim:      per_action_dim_ep{args.episode:03d}.png")
    if language_diff_results and any(r is not None for r in language_diff_results):
        print(f"  Language diff:       language_diff_ep{args.episode:03d}.png")
    if vision_vs_state_results:
        print(f"  Vision vs state:     vision_vs_state_ep{args.episode:03d}.txt")
        print(f"  Vision vs state chart: vision_vs_state_ep{args.episode:03d}.png")
    if run_dir:
        print(f"\n  Run directory: {run_dir}")
        print(f"  Export data:   {os.path.join(run_dir, 'data')}/")

    print()

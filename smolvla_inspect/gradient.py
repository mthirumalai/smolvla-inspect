"""
Gradient-based attribution — saliency maps and GradCAM for SmolVLA.

Bypasses ``select_action()``'s ``@torch.no_grad()`` decorator by calling
internal methods (``prepare_images``, ``prepare_state``, ``sample_actions``)
directly under ``torch.enable_grad()``.
"""

import contextlib
import warnings

import numpy as np
import torch

from .data import build_policy_batch_from_sample, get_episode_frames, _resolve_task_string


# ---------------------------------------------------------------------------
# Monkey-patch: cast Long attention masks to bool for torch.where under grad
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_eager_attention_bool_mask(policy):
    """
    ``eager_attention_forward`` uses ``torch.where(mask, ...)`` where
    *mask* is Long.  Under ``torch.no_grad()`` PyTorch silently casts it,
    but with gradients enabled the dtype mismatch raises.  This context
    manager monkey-patches the method to add ``.bool()`` on the mask.
    """
    vlm_expert = getattr(policy.model, "vlm_with_expert", None)
    if vlm_expert is None:
        yield
        return

    orig_fn = vlm_expert.eager_attention_forward

    def _patched(attention_mask, *args, **kwargs):
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        return orig_fn(attention_mask, *args, **kwargs)

    vlm_expert.eager_attention_forward = _patched
    try:
        yield
    finally:
        vlm_expert.eager_attention_forward = orig_fn


# ---------------------------------------------------------------------------
# Internal: forward pass with gradient graph retained
# ---------------------------------------------------------------------------

def _run_forward_with_grad(policy, batch, device, noise_seed=42):
    """
    Replicate ``SmolVLAPolicy._get_action_chunk()`` without the
    ``@torch.no_grad()`` wrapper so that gradients flow back to inputs.

    Returns:
        action_scalar: scalar tensor suitable for ``.backward()``
        (the sum of the first predicted action step)
    """
    with torch.enable_grad(), _patch_eager_attention_bool_mask(policy):
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)

        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        # Fixed noise for reproducibility across frames
        bsize = state.shape[0]
        actions_shape = (
            bsize,
            policy.model.config.chunk_size,
            policy.model.config.max_action_dim,
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(noise_seed)
        noise = torch.randn(actions_shape, device=device, generator=gen, dtype=torch.float32)

        actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )

        # Backprop target: sum of the immediate next action (step 0)
        action_scalar = actions[:, 0, :].sum()

    return action_scalar


# ---------------------------------------------------------------------------
# Saliency map (vanilla input-gradient)
# ---------------------------------------------------------------------------

def compute_saliency_map(policy, sample, dataset, image_key, device,
                         noise_seed=42, task_override=None,
                         smooth_n=1, smooth_sigma=0.15):
    """
    Compute ``|d(action)/d(pixel)|`` at pixel resolution.

    When *smooth_n* > 1, uses SmoothGrad: averages saliency over *N*
    copies of the input with additive Gaussian noise (std = *smooth_sigma*).

    Returns:
        numpy array ``(H, W)`` in ``[0, 1]``, or *None* on failure.
    """
    accumulated = None
    n_success = 0

    for k in range(smooth_n):
        batch, grad_pkey = build_policy_batch_from_sample(
            sample, policy, device, batch_size=1,
            image_key_for_grad=image_key, dataset=dataset,
            task_override=task_override,
        )
        if grad_pkey is None:
            print("    WARNING: Could not identify gradient image tensor")
            return None

        grad_tensor = batch[grad_pkey]

        # Add Gaussian noise for SmoothGrad (skip for k=0 when N=1)
        if smooth_n > 1:
            noise = torch.randn_like(grad_tensor) * smooth_sigma
            grad_tensor.data.add_(noise)

        try:
            policy.reset()
            action_scalar = _run_forward_with_grad(policy, batch, device, noise_seed)
            action_scalar.backward()

            if grad_tensor.grad is None:
                if k == 0:
                    print("    WARNING: grad is None — gradient did not flow to input pixels")
                    return None
                continue

            sal = grad_tensor.grad.abs().squeeze(0)  # (C, H, W)
            if sal.dim() == 3:
                sal = sal.mean(dim=0)  # (H, W)

            if accumulated is None:
                accumulated = sal.detach().clone()
            else:
                accumulated += sal.detach()
            n_success += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("    WARNING: CUDA OOM during saliency — skipping this frame")
            return None
        except RuntimeError as e:
            if "MPS" in str(e) or "mps" in str(e):
                print(f"    WARNING: MPS backward error: {e}")
                print("    Consider using --gradient-device cpu")
                return None
            else:
                if k == 0:
                    print(f"    WARNING: Saliency computation failed: {e}")
                    return None
                # For SmoothGrad, tolerate occasional failures
                continue

    if accumulated is None or n_success == 0:
        return None

    accumulated /= n_success
    accumulated = accumulated / (accumulated.max() + 1e-8)
    return accumulated.cpu().numpy()


# ---------------------------------------------------------------------------
# GradCAM on SigLIP last encoder layer
# ---------------------------------------------------------------------------

def compute_gradcam_map(policy, sample, dataset, image_key, device,
                        noise_seed=42, task_override=None):
    """
    Gradient-weighted class activation map at patch resolution (32x32).

    Hooks the last SigLIP encoder layer to capture activations and their
    gradients, then computes GradCAM weights.

    Returns:
        numpy array ``(grid_h, grid_w)`` in ``[0, 1]``, or *None* on failure.
    """
    from .data import find_vision_encoder

    vision_encoder = find_vision_encoder(policy)
    if vision_encoder is None:
        print("    WARNING: Could not find vision encoder for GradCAM")
        return None

    # Hook last encoder layer
    try:
        last_layer = vision_encoder.encoder.layers[-1]
    except (AttributeError, IndexError):
        print("    WARNING: Could not access last encoder layer for GradCAM")
        return None

    activations = {}
    gradients = {}

    def fwd_hook(module, input, output):
        # output is typically a tuple; first element is hidden states
        out = output[0] if isinstance(output, tuple) else output
        activations["value"] = out

    def bwd_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fwd_handle = last_layer.register_forward_hook(fwd_hook)
    bwd_handle = last_layer.register_full_backward_hook(bwd_hook)

    try:
        batch, _ = build_policy_batch_from_sample(
            sample, policy, device, batch_size=1,
            image_key_for_grad=image_key, dataset=dataset,
            task_override=task_override,
        )

        policy.reset()
        action_scalar = _run_forward_with_grad(policy, batch, device, noise_seed)
        action_scalar.backward()

        if "value" not in activations or "value" not in gradients:
            print("    WARNING: GradCAM hooks did not fire")
            return None

        A = activations["value"]   # (B, n_patches, hidden_dim)
        dA = gradients["value"]    # same shape

        # GAP over patches → per-channel weight
        alpha = dA.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)

        # Weighted combination + ReLU
        cam = (alpha * A).sum(dim=-1)  # (B, n_patches)
        cam = torch.relu(cam)
        cam = cam.squeeze(0)  # (n_patches,)

        # Reshape to spatial grid
        n_patches = cam.shape[0]
        grid_side = int(n_patches ** 0.5)
        if grid_side * grid_side != n_patches:
            # Non-square — try to infer from vision encoder config
            img_size = getattr(getattr(vision_encoder, "config", None), "image_size", None) or 512
            patch_size = getattr(getattr(vision_encoder, "config", None), "patch_size", None) or 16
            grid_h = img_size // patch_size
            grid_w = grid_h
        else:
            grid_h = grid_w = grid_side

        cam_2d = cam[:grid_h * grid_w].reshape(grid_h, grid_w)
        cam_2d = cam_2d / (cam_2d.max() + 1e-8)
        return cam_2d.detach().float().cpu().numpy()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("    WARNING: CUDA OOM during GradCAM — skipping this frame")
        return None
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            print(f"    WARNING: MPS backward error: {e}")
            print("    Consider using --gradient-device cpu")
        else:
            print(f"    WARNING: GradCAM computation failed: {e}")
        return None
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


# ---------------------------------------------------------------------------
# Top-level: compute gradient maps for all frames
# ---------------------------------------------------------------------------

def compute_gradient_maps(policy, dataset, episode_idx, num_frames, image_key,
                          device, method="both", noise_seed=42,
                          task_override=None, smooth_n=1, smooth_sigma=0.15):
    """
    Compute saliency and/or GradCAM maps for a set of episode frames.

    Args:
        method: ``"saliency"``, ``"gradcam"``, or ``"both"``
        smooth_n: Number of SmoothGrad samples (1 = vanilla saliency).
        smooth_sigma: Gaussian noise std for SmoothGrad.

    Returns:
        ``(saliency_maps, gradcam_maps)`` — each is a list of numpy arrays
        or *None* if that method was not requested / all frames failed.
    """
    # MPS warning
    if str(device) == "mps":
        print("  NOTE: MPS backward support is limited. If gradient computation "
              "fails, try --gradient-device cpu")

    do_saliency = method in ("saliency", "both")
    do_gradcam = method in ("gradcam", "both")

    frame_pairs = get_episode_frames(dataset, episode_idx, num_frames, image_key)

    saliency_maps = [] if do_saliency else None
    gradcam_maps = [] if do_gradcam else None

    for i, (frame_idx, img_tensor) in enumerate(frame_pairs):
        sample = dataset[frame_idx]

        if do_saliency:
            policy.zero_grad()
            label = f"SmoothGrad (N={smooth_n})" if smooth_n > 1 else "Saliency"
            smap = compute_saliency_map(
                policy, sample, dataset, image_key, device,
                noise_seed=noise_seed, task_override=task_override,
                smooth_n=smooth_n, smooth_sigma=smooth_sigma,
            )
            if smap is not None:
                saliency_maps.append(smap)
                print(f"    Frame {i}: {label} computed ({smap.shape})")
            else:
                h, w = img_tensor.shape[1], img_tensor.shape[2]
                saliency_maps.append(np.ones((h, w)) * 0.5)
                print(f"    Frame {i}: {label} failed, using uniform")

        if do_gradcam:
            policy.zero_grad()
            gcam = compute_gradcam_map(
                policy, sample, dataset, image_key, device,
                noise_seed=noise_seed, task_override=task_override,
            )
            if gcam is not None:
                gradcam_maps.append(gcam)
                print(f"    Frame {i}: GradCAM computed ({gcam.shape})")
            else:
                h, w = img_tensor.shape[1], img_tensor.shape[2]
                gradcam_maps.append(np.ones((h, w)) * 0.5)
                print(f"    Frame {i}: GradCAM failed, using uniform")

    # If all frames failed, return None instead of list of uniforms
    if saliency_maps is not None and all(
        (m == 0.5).all() if isinstance(m, np.ndarray) else False for m in saliency_maps
    ):
        saliency_maps = None
    if gradcam_maps is not None and all(
        (m == 0.5).all() if isinstance(m, np.ndarray) else False for m in gradcam_maps
    ):
        gradcam_maps = None

    return saliency_maps, gradcam_maps

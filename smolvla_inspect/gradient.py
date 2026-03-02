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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .data import (
    build_policy_batch_from_sample,
    get_episode_frames,
    get_alternative_task_string,
    _resolve_task_string,
    find_vision_encoder,
)


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


def _run_forward_with_grad_raw(policy, batch, device, noise_seed=42):
    """
    Like ``_run_forward_with_grad`` but returns the raw ``actions`` tensor
    (with grad graph) instead of a scalar.  Used by F6 (per-action-dim)
    which needs separate backward passes per dimension.
    """
    with torch.enable_grad(), _patch_eager_attention_bool_mask(policy):
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)

        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

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

    return actions


# ---------------------------------------------------------------------------
# Unified multi-hook forward pass (F1 + F2 + F4 combined)
# ---------------------------------------------------------------------------

def _run_forward_with_multi_hooks(policy, sample, dataset, image_key, device,
                                   noise_seed=42, task_override=None,
                                   hook_specs=None, state_requires_grad=False,
                                   image_map=None):
    """
    Register multiple forward/backward hooks, run one forward+backward pass,
    and return captured activations/gradients organized by feature key.

    Args:
        hook_specs: list of dicts, each with:
            - ``"key"``: string identifier (e.g. ``"connector"``, ``"vlm_layer_4"``)
            - ``"module"``: the nn.Module to hook
        state_requires_grad: if True, enable grad on the state tensor

    Returns:
        dict mapping each key to ``{"activations": Tensor, "gradients": Tensor}``,
        plus ``"batch"`` containing the batch dict (for accessing state grad).
    """
    if hook_specs is None:
        hook_specs = []

    batch, grad_pkey = build_policy_batch_from_sample(
        sample, policy, device, batch_size=1,
        image_key_for_grad=image_key, dataset=dataset,
        task_override=task_override,
        state_requires_grad=state_requires_grad,
        image_map=image_map,
    )

    captured = {}
    handles = []

    for spec in hook_specs:
        key = spec["key"]
        module = spec["module"]
        store = {"activations": None, "gradients": None}
        captured[key] = store

        def _make_fwd(s):
            def fwd_hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                # Detach + clone for activations (used in GradCAM formula).
                # We capture gradients via a separate tensor that stays in
                # the graph: clone without detach, with retain_grad().
                s["activations"] = o.detach().clone()
                grad_tap = o.clone()
                grad_tap.retain_grad()
                s["_grad_tap"] = grad_tap
                # Return the clone so downstream inplace ops don't conflict
                # with the backward hook wrapper.
                if isinstance(out, tuple):
                    return (grad_tap,) + out[1:]
                return grad_tap
            return fwd_hook

        handles.append(module.register_forward_hook(_make_fwd(store)))

    try:
        policy.reset()
        action_scalar = _run_forward_with_grad(policy, batch, device, noise_seed)
        action_scalar.backward()
    finally:
        for h in handles:
            h.remove()

    captured["batch"] = batch
    captured["grad_pkey"] = grad_pkey
    return captured


def _gradcam_from_captured(activations, gradients, spatial_shape=None):
    """
    Compute GradCAM from pre-captured activations and gradients.

    Args:
        activations: ``(B, tokens, hidden_dim)``
        gradients: same shape
        spatial_shape: optional ``(h, w)`` — if *None*, infer square grid.

    Returns:
        numpy array ``(h, w)`` in ``[0, 1]``, or *None*.
    """
    if activations is None or gradients is None:
        return None

    alpha = gradients.mean(dim=1, keepdim=True)
    cam = (alpha * activations).sum(dim=-1)
    cam = torch.relu(cam).squeeze(0)

    n = cam.shape[0]
    if spatial_shape is not None:
        h, w = spatial_shape
    else:
        side = int(n ** 0.5)
        h = w = side

    cam = cam[:h * w].reshape(h, w)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Saliency map (vanilla input-gradient)
# ---------------------------------------------------------------------------

def compute_saliency_map(policy, sample, dataset, image_key, device,
                         noise_seed=42, task_override=None,
                         smooth_n=1, smooth_sigma=0.15, image_map=None):
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
            task_override=task_override, image_map=image_map,
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
                        noise_seed=42, task_override=None, image_map=None):
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
            task_override=task_override, image_map=image_map,
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
# F1: GradCAM on connector output
# ---------------------------------------------------------------------------

def compute_gradcam_connector(policy, sample, dataset, image_key, device,
                               noise_seed=42, task_override=None,
                               image_map=None):
    """
    GradCAM on the VLM connector output (post-pixel-shuffle, 64 tokens).

    Returns:
        numpy array ``(8, 8)`` in ``[0, 1]``, or *None* on failure.
    """
    try:
        connector = policy.model.vlm_with_expert.get_vlm_model().connector
    except AttributeError:
        print("    WARNING: Could not access VLM connector for GradCAM")
        return None

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, output):
        out = output[0] if isinstance(output, tuple) else output
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fwd_handle = connector.register_forward_hook(fwd_hook)
    bwd_handle = connector.register_full_backward_hook(bwd_hook)

    try:
        batch, _ = build_policy_batch_from_sample(
            sample, policy, device, batch_size=1,
            image_key_for_grad=image_key, dataset=dataset,
            task_override=task_override, image_map=image_map,
        )
        policy.reset()
        action_scalar = _run_forward_with_grad(policy, batch, device, noise_seed)
        action_scalar.backward()

        return _gradcam_from_captured(
            activations.get("value"), gradients.get("value"),
            spatial_shape=(8, 8),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("    WARNING: CUDA OOM during connector GradCAM")
        return None
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            print(f"    WARNING: MPS backward error: {e}")
        else:
            print(f"    WARNING: Connector GradCAM failed: {e}")
        return None
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def compute_gradcam_connector_maps(policy, dataset, episode_idx, num_frames,
                                    image_key, device, noise_seed=42,
                                    task_override=None, image_map=None):
    """Compute connector GradCAM for all frames in an episode."""
    frame_pairs = get_episode_frames(dataset, episode_idx, num_frames, image_key)
    maps = []
    for i, (frame_idx, img_tensor) in enumerate(tqdm(frame_pairs, desc="Connector GradCAM", unit="frame")):
        sample = dataset[frame_idx]
        policy.zero_grad()
        cam = compute_gradcam_connector(
            policy, sample, dataset, image_key, device,
            noise_seed=noise_seed, task_override=task_override,
            image_map=image_map,
        )
        if cam is not None:
            maps.append(cam)
            print(f"    Frame {i}: Connector GradCAM computed ({cam.shape})")
        else:
            maps.append(np.ones((8, 8)) * 0.5)
            print(f"    Frame {i}: Connector GradCAM failed, using uniform")
    # Return None if all failed
    if all((m == 0.5).all() for m in maps):
        return None
    return maps


# ---------------------------------------------------------------------------
# F2: GradCAM on VLM intermediate layers
# ---------------------------------------------------------------------------

def compute_gradcam_vlm_layers(policy, sample, dataset, image_key, device,
                                layer_indices, noise_seed=42,
                                task_override=None, image_map=None):
    """
    GradCAM on VLM text model intermediate layers.

    Hooks multiple layers simultaneously for a single forward+backward pass.
    Slices output to extract vision-token, language-token, and state-token
    attribution separately.

    Args:
        layer_indices: list of 0-indexed layer indices (e.g. [3, 7, 11, 15])

    Returns:
        dict mapping layer_index to {
            "vision_cam": numpy (8, 8) in [0,1],
            "lang_cam": numpy (n_lang,) in [0,1] or None,
            "state_cam": float in [0,1] or None,
        }, or *None* on failure.
    """
    try:
        text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    except AttributeError:
        print("    WARNING: Could not access VLM text model for layer GradCAM")
        return None

    # Build hook specs — hook the MLP submodule since SmolVLA's
    # forward_attn_layer() bypasses the layer's own forward().
    hook_specs = []
    for li in layer_indices:
        try:
            layer = text_model.layers[li]
            # Hook the MLP (it fires during forward_attn_layer)
            mlp = layer.mlp
        except (IndexError, AttributeError):
            num_available = len(text_model.layers)
            print(f"    WARNING: VLM layer {li} (0-indexed) not found — "
                  f"model has {num_available} layers (0-{num_available - 1}), skipping")
            continue
        hook_specs.append({"key": f"vlm_layer_{li}", "module": mlp})

    if not hook_specs:
        return None

    try:
        captured = _run_forward_with_multi_hooks(
            policy, sample, dataset, image_key, device,
            noise_seed=noise_seed, task_override=task_override,
            hook_specs=hook_specs, image_map=image_map,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("    WARNING: CUDA OOM during VLM layer GradCAM")
        return None
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            print(f"    WARNING: MPS backward error: {e}")
        else:
            print(f"    WARNING: VLM layer GradCAM failed: {e}")
        return None

    # Determine token boundaries
    batch = captured["batch"]
    n_lang = batch["observation.language.tokens"].shape[1]
    n_vision = 64  # after pixel shuffle

    result = {}
    for li in layer_indices:
        key = f"vlm_layer_{li}"
        if key not in captured:
            continue
        store = captured[key]
        A = store["activations"]
        grad_tap = store.get("_grad_tap")
        dA = grad_tap.grad if grad_tap is not None else None
        if A is None or dA is None:
            continue

        # Full-sequence GradCAM
        alpha = dA.mean(dim=1, keepdim=True)
        cam = (alpha * A).sum(dim=-1)
        cam = torch.relu(cam).squeeze(0)  # (seq_len,)
        cam = cam / (cam.max() + 1e-8)

        # Slice by modality
        vision_cam = cam[:n_vision].reshape(8, 8)
        vision_cam = vision_cam / (vision_cam.max() + 1e-8)

        lang_cam = None
        if n_lang > 0 and cam.shape[0] > n_vision + n_lang:
            lang_cam = cam[n_vision:n_vision + n_lang]
            lang_cam = lang_cam / (lang_cam.max() + 1e-8)
            lang_cam = lang_cam.detach().float().cpu().numpy()

        state_cam = None
        state_idx = n_vision + n_lang
        if cam.shape[0] > state_idx:
            state_cam = cam[state_idx].item()

        result[li] = {
            "vision_cam": vision_cam.detach().float().cpu().numpy(),
            "lang_cam": lang_cam,
            "state_cam": state_cam,
        }

    return result if result else None


def compute_gradcam_vlm_layers_maps(policy, dataset, episode_idx, num_frames,
                                     image_key, device, layer_indices,
                                     noise_seed=42, task_override=None,
                                     image_map=None):
    """Compute VLM layer GradCAM for all frames in an episode."""
    frame_pairs = get_episode_frames(dataset, episode_idx, num_frames, image_key)
    # result[layer_idx] = list of per-frame dicts
    all_results = {li: [] for li in layer_indices}
    for i, (frame_idx, img_tensor) in enumerate(tqdm(frame_pairs, desc="VLM layer GradCAM", unit="frame")):
        sample = dataset[frame_idx]
        policy.zero_grad()
        result = compute_gradcam_vlm_layers(
            policy, sample, dataset, image_key, device,
            layer_indices, noise_seed=noise_seed, task_override=task_override,
            image_map=image_map,
        )
        if result is not None:
            for li in layer_indices:
                if li in result:
                    all_results[li].append(result[li])
                else:
                    all_results[li].append({"vision_cam": np.ones((8, 8)) * 0.5,
                                            "lang_cam": None, "state_cam": None})
            print(f"    Frame {i}: VLM layer GradCAM computed")
        else:
            for li in layer_indices:
                all_results[li].append({"vision_cam": np.ones((8, 8)) * 0.5,
                                        "lang_cam": None, "state_cam": None})
            print(f"    Frame {i}: VLM layer GradCAM failed, using uniform")
    return all_results


# ---------------------------------------------------------------------------
# F4: Vision vs. State attribution ratio
# ---------------------------------------------------------------------------

def compute_vision_vs_state_ratio(policy, sample, dataset, image_key, device,
                                   noise_seed=42, task_override=None,
                                   image_map=None):
    """
    Compare gradient norms w.r.t. vision input vs. state input.

    Returns:
        dict with ``"vision_norm"``, ``"state_norm"``, ``"vision_share"``
        (float 0–1), or *None* on failure.
    """
    batch, grad_pkey = build_policy_batch_from_sample(
        sample, policy, device, batch_size=1,
        image_key_for_grad=image_key, dataset=dataset,
        task_override=task_override,
        state_requires_grad=True, image_map=image_map,
    )
    if grad_pkey is None:
        print("    WARNING: Could not identify gradient image tensor")
        return None

    state_key = "observation.state"
    if state_key not in batch or not batch[state_key].requires_grad:
        print("    WARNING: State tensor not available or not differentiable")
        return None

    try:
        policy.reset()
        action_scalar = _run_forward_with_grad(policy, batch, device, noise_seed)
        action_scalar.backward()

        img_grad = batch[grad_pkey].grad
        state_grad = batch[state_key].grad

        if img_grad is None or state_grad is None:
            print("    WARNING: Gradients did not flow to both inputs")
            return None

        vision_norm = img_grad.abs().sum().item()
        state_norm = state_grad.abs().sum().item()
        total = vision_norm + state_norm + 1e-12
        return {
            "vision_norm": vision_norm,
            "state_norm": state_norm,
            "vision_share": vision_norm / total,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("    WARNING: CUDA OOM during vision vs state attribution")
        return None
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            print(f"    WARNING: MPS backward error: {e}")
        else:
            print(f"    WARNING: Vision vs state attribution failed: {e}")
        return None


def compute_vision_vs_state_maps(policy, dataset, episode_idx, num_frames,
                                  image_key, device, noise_seed=42,
                                  task_override=None, image_map=None):
    """Compute vision vs state ratio for all frames."""
    frame_pairs = get_episode_frames(dataset, episode_idx, num_frames, image_key)
    results = []
    for i, (frame_idx, img_tensor) in enumerate(tqdm(frame_pairs, desc="Vision vs state", unit="frame")):
        sample = dataset[frame_idx]
        policy.zero_grad()
        r = compute_vision_vs_state_ratio(
            policy, sample, dataset, image_key, device,
            noise_seed=noise_seed, task_override=task_override,
            image_map=image_map,
        )
        results.append(r)
        if r is not None:
            print(f"    Frame {i}: vision={r['vision_norm']:.2f} "
                  f"({r['vision_share']:.0%}), "
                  f"state={r['state_norm']:.2f} "
                  f"({1-r['vision_share']:.0%})")
        else:
            print(f"    Frame {i}: Vision vs state failed")
    return results


# ---------------------------------------------------------------------------
# F6: Per-action-dimension GradCAM
# ---------------------------------------------------------------------------

def compute_per_action_dim_gradcam(policy, sample, dataset, image_key, device,
                                    noise_seed=42, task_override=None,
                                    action_dim_names=None, image_map=None):
    """
    GradCAM on SigLIP last layer, one map per action dimension.

    Uses a single forward pass and N backward passes with
    ``retain_graph=True`` (except the last).

    Args:
        action_dim_names: optional list of dimension names (for logging).

    Returns:
        list of numpy arrays ``(grid_h, grid_w)`` in ``[0, 1]``, one per
        action dim, or *None* on failure.
    """
    vision_encoder = find_vision_encoder(policy)
    if vision_encoder is None:
        print("    WARNING: Could not find vision encoder for per-dim GradCAM")
        return None

    try:
        last_layer = vision_encoder.encoder.layers[-1]
    except (AttributeError, IndexError):
        print("    WARNING: Could not access last encoder layer")
        return None

    activations = {}

    def fwd_hook(module, inp, output):
        out = output[0] if isinstance(output, tuple) else output
        activations["value"] = out

    fwd_handle = last_layer.register_forward_hook(fwd_hook)

    try:
        batch, _ = build_policy_batch_from_sample(
            sample, policy, device, batch_size=1,
            image_key_for_grad=image_key, dataset=dataset,
            task_override=task_override, image_map=image_map,
        )
        policy.reset()
        actions = _run_forward_with_grad_raw(policy, batch, device, noise_seed)

        A = activations.get("value")
        if A is None:
            print("    WARNING: Forward hook did not fire")
            return None

        n_dims = actions.shape[-1]
        grid_h = grid_w = int(A.shape[1] ** 0.5)
        if grid_h * grid_w != A.shape[1]:
            img_size = getattr(getattr(vision_encoder, "config", None), "image_size", None) or 512
            patch_size = getattr(getattr(vision_encoder, "config", None), "patch_size", None) or 16
            grid_h = grid_w = img_size // patch_size

        cam_maps = []
        magnitudes = []
        for d in tqdm(range(n_dims), desc="  Action dims", unit="dim", leave=False):
            # Register backward hook to capture per-dim gradients
            grad_store = {}

            def _make_bwd(store):
                def bwd_hook(module, grad_in, grad_out):
                    store["value"] = grad_out[0]
                return bwd_hook

            bwd_handle = last_layer.register_full_backward_hook(_make_bwd(grad_store))

            retain = (d < n_dims - 1)
            actions[:, 0, d].backward(retain_graph=retain)

            bwd_handle.remove()

            dA = grad_store.get("value")
            if dA is None:
                cam_maps.append(np.ones((grid_h, grid_w)) * 0.5)
                magnitudes.append(0.0)
                continue

            alpha = dA.mean(dim=1, keepdim=True)
            cam = (alpha * A).sum(dim=-1)
            cam = torch.relu(cam).squeeze(0)
            cam = cam[:grid_h * grid_w].reshape(grid_h, grid_w)
            raw_max = cam.max().item()
            magnitudes.append(raw_max)
            cam = cam / (cam.max() + 1e-8)
            cam_maps.append(cam.detach().float().cpu().numpy())

            # Zero grads for next dimension
            policy.zero_grad()
            if A.grad is not None:
                A.grad = None

            dim_name = action_dim_names[d] if action_dim_names and d < len(action_dim_names) else f"dim_{d}"
            print(f"      Action dim {d} ({dim_name}): GradCAM computed")

        return {"maps": cam_maps, "magnitudes": magnitudes}

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("    WARNING: CUDA OOM during per-action-dim GradCAM")
        return None
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            print(f"    WARNING: MPS backward error: {e}")
        else:
            print(f"    WARNING: Per-action-dim GradCAM failed: {e}")
        return None
    finally:
        fwd_handle.remove()


def compute_per_action_dim_maps(policy, dataset, episode_idx, num_frames,
                                 image_key, device, noise_seed=42,
                                 task_override=None, action_dim_names=None,
                                 image_map=None):
    """Compute per-action-dim GradCAM for all frames.

    Returns:
        ``(all_maps, all_magnitudes)`` — all_maps is a list of (list of cam
        maps per dim) per frame; all_magnitudes is a list of (list of float
        magnitudes per dim) per frame.
    """
    frame_pairs = get_episode_frames(dataset, episode_idx, num_frames, image_key)
    all_maps = []
    all_magnitudes = []
    for i, (frame_idx, img_tensor) in enumerate(tqdm(frame_pairs, desc="Per-action-dim", unit="frame")):
        sample = dataset[frame_idx]
        policy.zero_grad()
        print(f"    Frame {i}: Per-action-dim GradCAM...")
        result = compute_per_action_dim_gradcam(
            policy, sample, dataset, image_key, device,
            noise_seed=noise_seed, task_override=task_override,
            action_dim_names=action_dim_names, image_map=image_map,
        )
        if result is not None:
            all_maps.append(result["maps"])
            all_magnitudes.append(result["magnitudes"])
        else:
            all_maps.append(None)
            all_magnitudes.append(None)
            print(f"    Frame {i}: Per-action-dim GradCAM failed")
    return all_maps, all_magnitudes


# ---------------------------------------------------------------------------
# F5: Language-conditional comparison
# ---------------------------------------------------------------------------

def compute_language_conditional_diff(policy, sample, dataset, image_key,
                                       device, noise_seed=42,
                                       task_override=None, alt_task=None,
                                       image_map=None):
    """
    Run GradCAM twice — once with the original task, once with an
    alternative — and compute the difference.

    Args:
        alt_task: explicit alternative task string, or *None* for auto.

    Returns:
        dict with ``"original_cam"``, ``"alt_cam"``, ``"diff_cam"`` (all
        numpy (grid_h, grid_w)), ``"original_task"``, ``"alt_task"``,
        ``"tier"``, or *None* on failure.
    """
    original_task = task_override or _resolve_task_string(sample, dataset)

    if alt_task is None or alt_task == "auto":
        alt_task_str, tier = get_alternative_task_string(original_task, dataset)
    else:
        alt_task_str = alt_task
        tier = 0  # user-specified

    print(f"    Original task: \"{original_task}\"")
    print(f"    Alt task (tier {tier}): \"{alt_task_str}\"")

    # GradCAM with original task
    policy.zero_grad()
    original_cam = compute_gradcam_map(
        policy, sample, dataset, image_key, device,
        noise_seed=noise_seed, task_override=original_task,
        image_map=image_map,
    )
    if original_cam is None:
        return None

    # GradCAM with alternative task
    policy.zero_grad()
    alt_cam = compute_gradcam_map(
        policy, sample, dataset, image_key, device,
        noise_seed=noise_seed, task_override=alt_task_str,
        image_map=image_map,
    )
    if alt_cam is None:
        return None

    # Symmetric difference: positive = original stronger, negative = alt stronger
    diff = original_cam - alt_cam
    abs_max = np.abs(diff).max() + 1e-8
    diff = diff / abs_max  # [-1, 1]

    return {
        "original_cam": original_cam,
        "alt_cam": alt_cam,
        "diff_cam": diff,
        "original_task": original_task,
        "alt_task": alt_task_str,
        "tier": tier,
    }


def compute_language_conditional_maps(policy, dataset, episode_idx, num_frames,
                                       image_key, device, noise_seed=42,
                                       task_override=None, alt_task=None,
                                       image_map=None):
    """Compute language-conditional diff for all frames."""
    frame_pairs = get_episode_frames(dataset, episode_idx, num_frames, image_key)
    results = []
    for i, (frame_idx, img_tensor) in enumerate(tqdm(frame_pairs, desc="Language diff", unit="frame")):
        sample = dataset[frame_idx]
        print(f"    Frame {i}: Language-conditional comparison...")
        r = compute_language_conditional_diff(
            policy, sample, dataset, image_key, device,
            noise_seed=noise_seed, task_override=task_override,
            alt_task=alt_task, image_map=image_map,
        )
        results.append(r)
        if r is not None:
            print(f"    Frame {i}: Language diff computed")
        else:
            print(f"    Frame {i}: Language diff failed")
    return results


# ---------------------------------------------------------------------------
# Top-level: compute gradient maps for all frames
# ---------------------------------------------------------------------------

def compute_gradient_maps(policy, dataset, episode_idx, num_frames, image_key,
                          device, method="both", noise_seed=42,
                          task_override=None, smooth_n=1, smooth_sigma=0.15,
                          image_map=None):
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

    for i, (frame_idx, img_tensor) in enumerate(tqdm(frame_pairs, desc="Gradient maps", unit="frame")):
        sample = dataset[frame_idx]

        if do_saliency:
            policy.zero_grad()
            label = f"SmoothGrad (N={smooth_n})" if smooth_n > 1 else "Saliency"
            smap = compute_saliency_map(
                policy, sample, dataset, image_key, device,
                noise_seed=noise_seed, task_override=task_override,
                smooth_n=smooth_n, smooth_sigma=smooth_sigma,
                image_map=image_map,
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
                image_map=image_map,
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

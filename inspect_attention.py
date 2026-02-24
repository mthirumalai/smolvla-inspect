#!/usr/bin/env python3
"""
SmolVLA Attention Visualizer — End-to-End
==========================================

What this does:
  Loads a pretrained SmolVLA model and a real-world LeRobot dataset,
  runs inference on sample frames, and extracts attention heatmaps from
  the SigLIP vision encoder to show exactly which pixels drive the
  model's action predictions.

Architecture recap:
  SmolVLA = SigLIP (vision encoder) + SmolLM2 (language decoder) + Action Expert
  
  SigLIP is a Vision Transformer. It splits each image into 14x14 pixel patches,
  then runs self-attention across all patches. The attention weights tell us
  which patches "look at" which other patches.
  
  We extract the attention from the LAST layer of SigLIP, average across all
  heads, and compute the mean attention each patch receives from all others
  to produce a spatial heatmap over the original image. (SigLIP has no CLS
  token — it is a pure patch-based encoder with no special classification token.)

Usage:
  pip install lerobot[smolvla] matplotlib numpy Pillow
  python smolvla_attention_viz.py

  # Or with options:
  python smolvla_attention_viz.py \
      --model lerobot/smolvla_base \
      --dataset lerobot/svla_so101_pickplace \
      --episode 0 \
      --num-frames 8 \
      --output-dir ./attention_maps \
      --device cpu

Output:
  Saves PNG files showing original frames with attention heatmap overlays,
  plus a grid summary image. Each heatmap shows which image regions the
  vision encoder attends to most strongly.

Interpreting results:
  - GOOD policy: Attention concentrated on gripper, target object, and goal
  - OVERFITTING: Attention spread across background (shelves, cables, table grain)
  - This is the "smoking gun" for the background distribution shift problem
"""

import argparse
import logging
import math
import os
import sys
import warnings
from pathlib import Path

# Preload Homebrew FFmpeg 6 libavdevice so PyAV/av doesn't load its bundled copy;
# avoids "Class AVFFrameReceiver is implemented in both..." duplicate symbol warning.
_ffmpeg6_lib = "/opt/homebrew/opt/ffmpeg@6/lib"
if os.path.isdir(_ffmpeg6_lib):
    _libavdevice = os.path.join(_ffmpeg6_lib, "libavdevice.60.dylib")
    if os.path.isfile(_libavdevice):
        try:
            import ctypes
            ctypes.CDLL(_libavdevice)
        except OSError:
            pass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

try:
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
except ImportError:
    def resize_with_pad(img, width, height, pad_value=-1):
        """Aspect-ratio-preserving resize with top/left padding."""
        if img.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
        cur_height, cur_width = img.shape[2:]
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False,
        )
        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img

_CYAN_CMAP = LinearSegmentedColormap.from_list("cyan", ["black", "cyan", "white"])


# ---------------------------------------------------------------------------
# 1. Attention hook — intercepts SigLIP's self-attention weights
# ---------------------------------------------------------------------------

class SigLIPAttentionCapture:
    """
    Registers forward hooks on the SigLIP vision encoder's attention layers
    to capture attention weights during inference.
    
    SigLIP uses torch.nn.MultiheadAttention, which can return attn weights
    when called with need_weights=True. Since the LeRobot/transformers code
    may not pass that flag, we hook into the module and monkey-patch the
    forward call to force weight capture.
    """
    
    def __init__(self):
        self.attention_maps = []  # List of (layer_idx, attn_weights) tuples
        self.hooks = []
    
    def _make_hook(self, layer_idx):
        """Create a hook closure for a specific layer."""
        def hook_fn(module, input_args, output):
            """
            torch.nn.MultiheadAttention.forward returns:
              (attn_output, attn_output_weights)  when need_weights=True
              (attn_output, None)                  when need_weights=False
            
            We intercept the output. If weights are None, we re-run the
            attention computation to get them.
            """
            if isinstance(output, tuple) and len(output) == 2:
                attn_output, attn_weights = output
                if attn_weights is not None:
                    self.attention_maps.append((layer_idx, attn_weights.detach().cpu()))
                    return output
            
            # Weights not available — eager attention is forced so this
            # should not happen. Log a warning instead of attempting a
            # manual computation that would ignore multi-head reshaping.
            print(f"    WARNING: Could not capture attention weights at layer {layer_idx}")
            
            return output
        
        return hook_fn
    
    def register_hooks(self, vision_encoder):
        """
        Walk the vision encoder module tree and hook into all
        MultiheadAttention or equivalent attention modules.
        """
        self.clear()
        
        # Strategy 1: Look for nn.MultiheadAttention modules
        found_mha = False
        for name, module in vision_encoder.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                layer_idx = len(self.hooks)
                hook = module.register_forward_hook(self._make_hook(layer_idx))
                self.hooks.append(hook)
                found_mha = True
        
        if found_mha:
            print(f"  Hooked into {len(self.hooks)} MultiheadAttention layers")
            return
        
        # Strategy 2: Look for attention modules by name pattern
        # (transformers-style SiglipAttention or similar)
        for name, module in vision_encoder.named_modules():
            module_type = type(module).__name__.lower()
            if "attention" in module_type and "embed" not in module_type:
                layer_idx = len(self.hooks)
                hook = module.register_forward_hook(self._make_hook(layer_idx))
                self.hooks.append(hook)
        
        if self.hooks:
            print(f"  Hooked into {len(self.hooks)} attention modules (by name)")
        else:
            print("  WARNING: No attention modules found. Will use gradient-based fallback.")
    
    def clear(self):
        """Remove all hooks and clear captured maps."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.attention_maps = []
    
    def reset_maps(self):
        """Clear captured maps but keep hooks registered."""
        self.attention_maps = []
    
    def get_last_layer_attention(self):
        """
        Return attention weights from the last encoder layer.
        Shape: (batch, num_patches, num_patches) or (batch, heads, patches, patches)
        """
        if not self.attention_maps:
            return None
        # Sort by layer index, take last
        sorted_maps = sorted(self.attention_maps, key=lambda x: x[0])
        _, attn = sorted_maps[-1]
        return attn
    
    def get_all_layer_attentions(self):
        """Return attention weights from all layers, sorted by layer index."""
        if not self.attention_maps:
            return []
        return sorted(self.attention_maps, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# 1b. Cross-attention hook — action expert attending to vision tokens
# ---------------------------------------------------------------------------

class ActionVisionAttentionCapture:
    """
    Captures how the action expert's queries attend to vision tokens in
    the VLM prefix.

    SmolVLA has no explicit cross-attention *layers*.  Instead the VLM
    builds a KV cache from the prefix (vision + language + state tokens)
    and the action expert queries that cache inside
    ``forward_cross_attn_layer()``.  All attention goes through a single
    ``eager_attention_forward()`` method.

    We monkey-patch that method to intercept the softmax probabilities
    whenever expert queries (suffix) attend to prefix keys
    (detected by Q seq-len != K seq-len).

    After capture the caller selects just the columns corresponding to
    vision tokens and reshapes into a spatial heatmap.
    """

    def __init__(self):
        self.cross_attn_weights = []   # (layer_call_idx, probs)  probs: (B, heads, q_len, k_len)
        self._original_fn = None
        self._patched_obj = None
        self._call_idx = 0

    # ------------------------------------------------------------------
    # patch / unpatch
    # ------------------------------------------------------------------
    def register(self, vlm_with_expert):
        """Wrap ``eager_attention_forward`` on *vlm_with_expert*."""
        import types

        self.clear()
        self._original_fn = vlm_with_expert.eager_attention_forward
        self._patched_obj = vlm_with_expert
        capture = self

        def _wrapped(self_model, attention_mask, batch_size, head_dim,
                     query_states, key_states, value_states):
            # Ensure boolean mask (some code paths produce Long 0/1 masks)
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()

            # ---------- original computation ----------
            output = capture._original_fn(
                attention_mask, batch_size, head_dim,
                query_states, key_states, value_states,
            )

            # ---------- detect cross-attention ----------
            q_len = query_states.shape[1]
            k_len = key_states.shape[1]
            if q_len != k_len:
                # Re-derive attention probs (mirrors eager_attention_forward)
                num_att_heads = self_model.num_attention_heads
                num_kv_heads = self_model.num_key_value_heads
                num_kv_groups = num_att_heads // num_kv_heads
                seq_len_k = key_states.shape[1]

                ks = key_states[:, :, :, None, :].expand(
                    batch_size, seq_len_k, num_kv_heads, num_kv_groups, head_dim
                ).reshape(batch_size, seq_len_k, num_kv_heads * num_kv_groups, head_dim)

                q = query_states.to(dtype=torch.float32).transpose(1, 2)
                k = ks.to(dtype=torch.float32).transpose(1, 2)

                scores = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
                big_neg = torch.finfo(scores.dtype).min
                scores = torch.where(
                    attention_mask[:, None, :, :], scores, big_neg
                )
                probs = F.softmax(scores, dim=-1)
                capture.cross_attn_weights.append(
                    (capture._call_idx, probs.detach().cpu())
                )
                capture._call_idx += 1

            return output

        vlm_with_expert.eager_attention_forward = types.MethodType(
            _wrapped, vlm_with_expert
        )

    def clear(self):
        """Remove the monkey-patch and discard captured data."""
        if self._original_fn is not None and self._patched_obj is not None:
            self._patched_obj.eager_attention_forward = self._original_fn
        self._original_fn = None
        self._patched_obj = None
        self.cross_attn_weights = []
        self._call_idx = 0

    def reset_maps(self):
        self.cross_attn_weights = []
        self._call_idx = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def get_vision_token_range(self, policy):
        """
        Return ``(start, end)`` — the column indices in the prefix KV
        cache that correspond to vision tokens (one image, no special
        tokens by default).
        """
        add_special = getattr(policy.config, "add_image_special_tokens", False)
        # Run the connector on a dummy image to get the token count
        vlm_with_expert = policy.model.vlm_with_expert
        vision_model = vlm_with_expert.get_vlm_model().vision_model
        img_size = getattr(
            getattr(vision_model, "config", None), "image_size", 384
        )
        dummy = torch.zeros(1, 3, img_size, img_size,
                            device=next(vision_model.parameters()).device,
                            dtype=next(vision_model.parameters()).dtype)
        with torch.no_grad():
            hidden = vision_model(pixel_values=dummy).last_hidden_state
            after_connector = vlm_with_expert.get_vlm_model().connector(hidden)
        n_vision = after_connector.shape[1]

        start = 1 if add_special else 0
        end = start + n_vision
        return start, end, n_vision

    def get_mean_cross_attention(self, vision_start, vision_end):
        """
        Average captured cross-attention over all layers and heads,
        then select only the vision-token columns.

        Returns:
            Tensor of shape ``(n_vision_tokens,)`` or *None*.
        """
        if not self.cross_attn_weights:
            return None

        accum = None
        count = 0
        for _idx, probs in self.cross_attn_weights:
            # probs: (B, heads, q_len, k_len)
            # Average over batch, heads, and query tokens
            avg = probs[0].mean(dim=0).mean(dim=0)  # (k_len,)
            vision_avg = avg[vision_start:vision_end]
            if accum is None:
                accum = vision_avg
            else:
                accum = accum + vision_avg
            count += 1

        if count == 0:
            return None

        return accum / count


# ---------------------------------------------------------------------------
# 1c. Decoder attention capture — VLM+Expert self-attn & Expert cross-attn
# ---------------------------------------------------------------------------

class DecoderAttentionCapture:
    """
    Captures attention weights from ``SmolVLMWithExpert.eager_attention_forward``.

    SmolVLA bypasses the standard ``LlamaAttention.forward()`` path — instead,
    ``forward_attn_layer`` / ``forward_cross_attn_layer`` manually project
    Q/K/V and call ``eager_attention_forward`` directly. So we monkey-patch
    that method to intercept the attention probabilities.

    Captured maps are categorised by the Q/K sequence lengths:
      - **Self-attention** (``q_len == k_len``, ``q_len > 1``): prefill phase
        where VLM + Expert tokens are concatenated.
      - **Cross-attention** (``q_len != k_len``): Expert queries attending to
        the VLM prefix KV cache.
    """

    def __init__(self):
        self.self_attn_maps = []    # (seq_idx, probs) for prefill self-attn
        self.cross_attn_maps = []   # (seq_idx, probs) for expert cross-attn
        self._original_fn = None
        self._patched_obj = None
        self._self_idx = 0
        self._cross_idx = 0

    def register(self, vlm_with_expert):
        """Wrap ``eager_attention_forward`` on *vlm_with_expert*."""
        import types

        self.clear()
        self._original_fn = vlm_with_expert.eager_attention_forward
        self._patched_obj = vlm_with_expert
        capture = self

        def _wrapped(self_model, attention_mask, batch_size, head_dim,
                     query_states, key_states, value_states):
            # Ensure boolean mask
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()

            # ---------- original computation ----------
            output = capture._original_fn(
                attention_mask, batch_size, head_dim,
                query_states, key_states, value_states,
            )

            # ---------- capture attention probs ----------
            q_len = query_states.shape[1]
            k_len = key_states.shape[1]

            # Only capture meaningful attention (skip single-token
            # autoregressive steps which have q_len == 1)
            if q_len > 1:
                num_att_heads = self_model.num_attention_heads
                num_kv_heads = self_model.num_key_value_heads
                num_kv_groups = num_att_heads // num_kv_heads
                seq_len_k = key_states.shape[1]

                ks = key_states[:, :, :, None, :].expand(
                    batch_size, seq_len_k, num_kv_heads, num_kv_groups, head_dim
                ).reshape(batch_size, seq_len_k, num_kv_heads * num_kv_groups, head_dim)

                q = query_states.to(dtype=torch.float32).transpose(1, 2)
                k = ks.to(dtype=torch.float32).transpose(1, 2)

                scores = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
                big_neg = torch.finfo(scores.dtype).min
                scores = torch.where(
                    attention_mask[:, None, :, :], scores, big_neg
                )
                probs = F.softmax(scores, dim=-1)

                if q_len == k_len:
                    capture.self_attn_maps.append(
                        (capture._self_idx, probs.detach().cpu())
                    )
                    capture._self_idx += 1
                else:
                    capture.cross_attn_maps.append(
                        (capture._cross_idx, probs.detach().cpu())
                    )
                    capture._cross_idx += 1

            return output

        vlm_with_expert.eager_attention_forward = types.MethodType(
            _wrapped, vlm_with_expert
        )

    def clear(self):
        """Remove the monkey-patch and discard captured data."""
        if self._original_fn is not None and self._patched_obj is not None:
            self._patched_obj.eager_attention_forward = self._original_fn
        self._original_fn = None
        self._patched_obj = None
        self.self_attn_maps = []
        self.cross_attn_maps = []
        self._self_idx = 0
        self._cross_idx = 0

    def reset_maps(self):
        """Clear captured maps but keep the monkey-patch active."""
        self.self_attn_maps = []
        self.cross_attn_maps = []
        self._self_idx = 0
        self._cross_idx = 0

    def get_self_attn_layers(self):
        """Return prefill self-attention maps sorted by sequential index."""
        if not self.self_attn_maps:
            return []
        return sorted(self.self_attn_maps, key=lambda x: x[0])

    def get_cross_attn_layers(self):
        """Return expert cross-attention maps sorted by sequential index."""
        if not self.cross_attn_maps:
            return []
        return sorted(self.cross_attn_maps, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# 2. Gradient-based attention fallback (GradCAM-style)
# ---------------------------------------------------------------------------

class GradCAMFallback:
    """
    If we can't extract attention weights directly (e.g., because the
    attention implementation doesn't expose them), we use GradCAM on
    the last convolutional/linear layer of the vision encoder.
    
    GradCAM: Gradient-weighted Class Activation Mapping
    - Forward pass through the model
    - Backward pass from the action output w.r.t. vision features
    - Weight the feature maps by their gradient importance
    - Produces a heatmap showing which spatial regions influenced the output
    """
    
    def __init__(self):
        self.features = None
        self.gradients = None
        self.hook_f = None
        self.hook_b = None
    
    def register(self, target_layer):
        """Register forward and backward hooks on a target layer."""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.features = output[0].detach()
            else:
                self.features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hook_f = target_layer.register_forward_hook(forward_hook)
        self.hook_b = target_layer.register_full_backward_hook(backward_hook)
    
    def compute_cam(self):
        """Compute GradCAM heatmap from stored features and gradients."""
        if self.features is None or self.gradients is None:
            return None
        
        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=-1, keepdim=True)  # (B, T, 1) or (B, C, 1, 1)
        
        # Weighted combination of feature maps
        cam = (weights * self.features).sum(dim=-1)  # (B, T) or (B, C)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu()
    
    def clear(self):
        if self.hook_f:
            self.hook_f.remove()
        if self.hook_b:
            self.hook_b.remove()


# ---------------------------------------------------------------------------
# 3. Attention-to-heatmap conversion
# ---------------------------------------------------------------------------

def compute_padding_patches(input_hw, target_size, patch_size):
    """Number of pure-padding patch rows (top) and columns (left)
    produced by resize_with_pad for the given input dimensions."""
    if input_hw is None:
        return (0, 0)
    in_h, in_w = input_hw
    ratio = max(in_w / target_size, in_h / target_size)
    resized_h = int(in_h / ratio)
    resized_w = int(in_w / ratio)
    pad_h = max(0, target_size - resized_h)
    pad_w = max(0, target_size - resized_w)
    return (pad_h // patch_size, pad_w // patch_size)


def attention_to_heatmap(attn_weights, grid_size, image_size, content_crop=None):
    """
    Convert attention weights from patch-space to pixel-space heatmap.
    
    Args:
        attn_weights: Tensor of shape (num_patches,) or (H_patches, W_patches)
                      representing per-patch attention scores
        grid_size: (H_patches, W_patches) — the patch grid dimensions
        image_size: (H_pixels, W_pixels) — the original image dimensions
    
    Returns:
        heatmap: numpy array of shape (H_pixels, W_pixels) normalized to [0, 1]
    """
    h_patches, w_patches = grid_size
    h_img, w_img = image_size
    
    # Reshape to 2D grid if flat
    if attn_weights.dim() == 1:
        expected = h_patches * w_patches
        if attn_weights.shape[0] != expected:
            # Try to infer square grid
            side = int(math.sqrt(attn_weights.shape[0]))
            if side * side == attn_weights.shape[0]:
                h_patches, w_patches = side, side
            else:
                # Truncate or pad
                attn_weights = attn_weights[:expected]
        
        attn_2d = attn_weights.reshape(h_patches, w_patches)
    else:
        attn_2d = attn_weights

    # Crop out pure-padding patches (top rows, left columns) before upsampling
    if content_crop is not None:
        crop_h, crop_w = content_crop
        if crop_h > 0 or crop_w > 0:
            attn_2d = attn_2d[crop_h:, crop_w:]

    # Upsample to image resolution using bilinear interpolation
    attn_2d = attn_2d.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    heatmap = F.interpolate(attn_2d, size=(h_img, w_img), mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze().numpy()
    
    # Normalize to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    
    return heatmap


def compute_patch_attention_scores(attn_weights, method="mean"):
    """
    Reduce full attention matrix to per-patch importance scores.

    SigLIP is a pure patch encoder with no CLS token, so we use
    column-wise mean attention (how much attention each patch receives
    from all other patches) as the default aggregation.

    Args:
        attn_weights: Tensor, typically (num_patches, num_patches) or
                      (heads, num_patches, num_patches)
        method: How to aggregate:
            - "mean": Average attention each patch receives from all others

    Returns:
        scores: Tensor of shape (num_patches,)
    """
    # If multi-head, average across heads first
    if attn_weights.dim() == 3:
        attn_weights = attn_weights.mean(dim=0)  # (patches, patches)

    if attn_weights.dim() == 1:
        return attn_weights  # Already reduced

    # Column-wise mean = how much attention each patch receives
    scores = attn_weights.mean(dim=0)

    return scores


def compute_attention_rollout(all_layer_attentions):
    """
    Combine attention across all encoder layers using attention rollout.

    The method accounts for residual connections by mixing each layer's
    attention with an identity matrix (the residual stream keeps a copy of
    each token unchanged):

        rollout = I
        for A in layers:
            A_hat = 0.5 * I + 0.5 * A   # residual connection
            rollout = rollout @ A_hat
        normalize each row to sum to 1

    Args:
        all_layer_attentions: list of (layer_idx, attn_weights) tuples.
            Each attn_weights tensor has shape (batch, heads, patches, patches)
            or (heads, patches, patches) or (patches, patches).

    Returns:
        rollout: Tensor of shape (patches, patches) — the accumulated
                 attention from input patches to output patches.
    """
    matrices = []
    for _layer_idx, attn in sorted(all_layer_attentions, key=lambda x: x[0]):
        # Reduce to (patches, patches)
        while attn.dim() > 3:
            attn = attn[0]  # drop batch
        if attn.dim() == 3:
            attn = attn.mean(dim=0)  # average heads
        matrices.append(attn.float())

    if not matrices:
        return None

    n = matrices[0].shape[0]
    eye = torch.eye(n)
    rollout = eye.clone()

    for A in matrices:
        A_hat = 0.5 * eye + 0.5 * A
        rollout = rollout @ A_hat

    # Row-normalise so each row sums to 1
    rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-8)

    return rollout


def compute_positional_baseline(vision_encoder, attn_capture, device, method,
                                input_hw=None):
    """
    Compute the attention pattern produced by a content-free (mean-gray)
    image.  This captures the fixed positional component of attention so
    it can be subtracted from real frames to reveal the content-dependent
    signal.

    Args:
        vision_encoder: The SigLIP vision encoder module.
        attn_capture: A ``SigLIPAttentionCapture`` instance with hooks
            already registered on *vision_encoder*.
        device: Torch device.
        method: ``"last-layer"`` or ``"rollout"`` — same aggregation used
            for real frames so the baseline is comparable.
        input_hw: Optional ``(H, W)`` of the original input images.  When
            provided the baseline gray image is built at this aspect ratio
            then preprocessed with ``resize_with_pad`` so padding patches
            match those in real frames.

    Returns:
        Tensor of shape ``(num_patches,)`` — per-patch baseline scores.
    """
    # Build a mean-gray image at the encoder's expected resolution
    img_size = getattr(
        getattr(vision_encoder, "config", None), "image_size", None,
    ) or getattr(vision_encoder, "image_size", 384)

    if input_hw is not None:
        # Build gray at original aspect ratio, then resize_with_pad
        # so padding patches match the real frames exactly.
        in_h, in_w = input_hw
        gray_content = torch.full((1, 3, in_h, in_w), 0.5, device=device)
        gray = resize_with_pad(gray_content, img_size, img_size, pad_value=0)
        gray = gray * 2.0 - 1.0  # normalize to [-1, 1] matching SigLIP
    else:
        gray = torch.full((1, 3, img_size, img_size), 0.5, device=device)
    try:
        enc_dtype = next(vision_encoder.parameters()).dtype
        gray = gray.to(enc_dtype)
    except StopIteration:
        pass

    patch_size = getattr(vision_encoder, "patch_size", None) or getattr(
        getattr(vision_encoder, "config", None), "patch_size", 14,
    )
    n_patches_h = img_size // patch_size
    n_patches_w = img_size // patch_size
    patch_mask = torch.ones(1, n_patches_h, n_patches_w, dtype=torch.bool, device=device)

    # Forward pass through the vision encoder
    attn_capture.reset_maps()
    with torch.no_grad():
        try:
            if hasattr(vision_encoder, "embeddings") and hasattr(vision_encoder, "encoder"):
                embeddings = vision_encoder.embeddings(gray, patch_mask)
                vision_encoder.encoder(embeddings)
            elif hasattr(vision_encoder, "forward"):
                vision_encoder(gray)
            else:
                vision_encoder(pixel_values=gray, patch_attention_mask=patch_mask)
        except Exception as e:
            print(f"  WARNING: Baseline forward pass failed ({e}), skipping correction")
            return None

    # Reduce captured attention using the same method as real frames
    if method == "rollout":
        all_layers = attn_capture.get_all_layer_attentions()
        rollout_mat = compute_attention_rollout(all_layers)
        if rollout_mat is None:
            return None
        baseline_scores = rollout_mat.mean(dim=0)
    else:
        # "last-layer" or "all-layers" both use last-layer for the summary
        attn = attn_capture.get_last_layer_attention()
        if attn is None:
            return None
        while attn.dim() > 3:
            attn = attn[0]
        if attn.dim() == 3:
            attn = attn.mean(dim=0)
        baseline_scores = compute_patch_attention_scores(attn, method="mean")

    attn_capture.reset_maps()
    return baseline_scores


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def overlay_heatmap(image_np, heatmap, alpha=0.5, colormap="jet"):
    """
    Overlay a heatmap on an image.
    
    Args:
        image_np: numpy array (H, W, 3) in [0, 255] uint8 or [0, 1] float
        heatmap: numpy array (H, W) in [0, 1]
        alpha: blend factor (0 = only image, 1 = only heatmap)
        colormap: matplotlib colormap name
    
    Returns:
        blended: numpy array (H, W, 3) uint8
    """
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) float in [0, 1]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    blended = (
        (1 - alpha) * image_np.astype(np.float32) + 
        alpha * heatmap_colored.astype(np.float32)
    ).astype(np.uint8)
    
    return blended


def _resize_heatmap(heatmap, h, w):
    """Resize a heatmap to (h, w) if necessary."""
    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        return np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
        ) / 255.0
    return heatmap


def _frame_to_np(frame):
    """Convert a frame tensor or array to uint8 numpy (H, W, 3)."""
    if isinstance(frame, torch.Tensor):
        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
    else:
        frame_np = np.array(frame)
    return frame_np


def create_visualization_grid(frames, heatmaps, actions=None,
                               cross_attn_heatmaps=None,
                               episode_idx=0, output_path="attention_grid.png"):
    """
    Create a grid visualization showing original frames, heatmaps, and overlays.

    Layout per frame:
      Row 1: Original image
      Row 2: Vision encoder self-attention heatmap (colorized)
      Row 3: Overlay (image + self-attention heatmap blended)

    When *cross_attn_heatmaps* is provided two extra rows are added:
      Row 4: Action→Vision cross-attention heatmap
      Row 5: Dual-color overlay (self-attn blue, cross-attn red)
    """
    n_frames = len(frames)
    has_cross = cross_attn_heatmaps is not None and len(cross_attn_heatmaps) == n_frames
    n_rows = 5 if has_cross else 3

    fig = plt.figure(figsize=(4 * n_frames, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_frames, hspace=0.3, wspace=0.05)

    for i, (frame, heatmap) in enumerate(zip(frames, heatmaps)):
        frame_np = _frame_to_np(frame)
        h, w = frame_np.shape[:2]

        heatmap_resized = _resize_heatmap(heatmap, h, w)
        overlay = overlay_heatmap(frame_np, heatmap_resized, alpha=0.45)

        # Row 1: Original
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(frame_np)
        ax1.set_title(f"Frame {i}", fontsize=10)
        ax1.axis("off")
        if i == 0:
            ax1.set_ylabel("Original", fontsize=11, rotation=0, labelpad=60, va="center")

        # Row 2: Self-attention heatmap
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
        ax2.axis("off")
        if i == 0:
            ax2.set_ylabel("SigLIP\nself-attn", fontsize=11, rotation=0, labelpad=60, va="center")

        # Row 3: Self-attention overlay
        ax3 = fig.add_subplot(gs[2, i])
        ax3.imshow(overlay)
        ax3.axis("off")
        if i == 0:
            ax3.set_ylabel("Self-attn\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")

        if has_cross:
            cross_hm = _resize_heatmap(cross_attn_heatmaps[i], h, w)

            # Row 4: Cross-attention heatmap
            ax4 = fig.add_subplot(gs[3, i])
            ax4.imshow(cross_hm, cmap="Greens", vmin=0, vmax=1)
            ax4.axis("off")
            if i == 0:
                ax4.set_ylabel("Action\ncross-attn", fontsize=11, rotation=0, labelpad=60, va="center")

            # Row 5: Co-attention overlay (self-attn × cross-attn)
            co_attn = heatmap_resized * cross_hm          # element-wise product
            co_attn = co_attn / (co_attn.max() + 1e-8)   # renormalize to [0, 1]
            co_overlay = frame_np.copy()
            co_overlay = (0.5 * co_overlay.astype(np.float32)
                          + 0.5 * _CYAN_CMAP(co_attn)[:, :, :3] * 255)
            co_overlay = np.clip(co_overlay, 0, 255).astype(np.uint8)

            ax5 = fig.add_subplot(gs[4, i])
            ax5.imshow(co_overlay)
            ax5.axis("off")
            if i == 0:
                ax5.set_ylabel("Co-attention\noverlay", fontsize=11, rotation=0, labelpad=60, va="center")

    if has_cross:
        legend = ("Row 1: Original  |  Row 2: SigLIP self-attn heatmap  |  Row 3: Self-attn overlay  |  "
                  "Row 4: Action cross-attn heatmap  |  Row 5: Co-attention (self × cross)")
        dual_legend = "Co-attention: self-attn × cross-attn — bright regions are both visually salient and action-relevant"
    else:
        legend = "Row 1: Original  |  Row 2: SigLIP self-attn heatmap  |  Row 3: Overlay (heatmap on frame)"
        dual_legend = None

    if dual_legend:
        title = (
            f"SmolVLA Attention — Episode {episode_idx}\n\n"
            f"{legend}\n\n"
            f"{dual_legend}"
        )
    else:
        title = (
            f"SmolVLA Attention — Episode {episode_idx}\n\n"
            f"{legend}"
        )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved grid: {output_path}")


def save_individual_frames(frames, heatmaps, output_dir, episode_idx=0):
    """Save each frame's overlay as a separate high-res PNG."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (frame, heatmap) in enumerate(zip(frames, heatmaps)):
        if isinstance(frame, torch.Tensor):
            frame_np = frame.permute(1, 2, 0).numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        else:
            frame_np = np.array(frame)
        
        h, w = frame_np.shape[:2]
        if heatmap.shape[0] != h or heatmap.shape[1] != w:
            heatmap_resized = np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
            ) / 255.0
        else:
            heatmap_resized = heatmap
        
        overlay = overlay_heatmap(frame_np, heatmap_resized, alpha=0.45)
        
        # Save overlay
        path = os.path.join(output_dir, f"ep{episode_idx:03d}_frame{i:04d}_overlay.png")
        Image.fromarray(overlay).save(path)
        
        # Save raw heatmap
        path_hm = os.path.join(output_dir, f"ep{episode_idx:03d}_frame{i:04d}_heatmap.png")
        fig_hm, ax_hm = plt.subplots(figsize=(6, 6))
        ax_hm.imshow(heatmap_resized, cmap="jet")
        ax_hm.axis("off")
        fig_hm.savefig(path_hm, dpi=100, bbox_inches="tight")
        plt.close(fig_hm)
    
    print(f"  Saved {len(frames)} individual frames to {output_dir}/")


def create_per_head_grid(frame, attn_weights, grid_size, image_size,
                         output_path="per_head_attention.png", content_crop=None):
    """
    Visualise each attention head's pattern individually for a single
    frame.  Useful for identifying specialised heads (e.g. one tracking
    the gripper, another tracking the object).

    Args:
        frame: image tensor (C, H, W) or numpy (H, W, 3)
        attn_weights: Tensor of shape ``(heads, patches, patches)`` —
            one attention matrix per head (not yet averaged).
        grid_size: ``(H_patches, W_patches)``
        image_size: ``(H_pixels, W_pixels)``
        output_path: where to save the PNG
    """
    if isinstance(frame, torch.Tensor):
        frame_np = frame.permute(1, 2, 0).numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
    else:
        frame_np = np.array(frame)

    n_heads = attn_weights.shape[0]
    cols = min(n_heads, 8)
    rows = math.ceil(n_heads / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for h in range(n_heads):
        r, c = divmod(h, cols)
        head_attn = attn_weights[h]                      # (patches, patches)
        scores = head_attn.mean(dim=0)                    # per-patch importance
        hmap = attention_to_heatmap(scores, grid_size, image_size, content_crop=content_crop)

        blended = overlay_heatmap(frame_np.copy(), hmap, alpha=0.5)
        axes[r, c].imshow(blended)
        axes[r, c].set_title(f"Head {h}", fontsize=9)
        axes[r, c].axis("off")

    # Turn off unused subplots
    for idx in range(n_heads, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("SigLIP vision encoder per-head self-attention", fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved per-head grid: {output_path}")


# ---------------------------------------------------------------------------
# 5. Model and dataset loading
# ---------------------------------------------------------------------------

def find_vision_encoder(policy):
    """
    Navigate the SmolVLA model hierarchy to find the SigLIP vision encoder.
    
    SmolVLA structure (typical):
      policy.model
        .vlm_model (or .model)
          .vision_model (or .vision_encoder)
            .encoder
              .layers[0..N]
                .self_attn  ← this is what we hook into
    """
    model = policy
    
    # Traverse known paths
    search_paths = [
        # SmolVLA (LeRobot): policy.model.vlm_with_expert.vlm.model.vision_model
        "model.vlm_with_expert.vlm.model.vision_model",
        # SmolVLA / other VLM wrapping
        "model.vlm.vision_model",
        "model.vlm.model.vision_model",
        "model.vlm_model.vision_model",
        "model.vlm_model.model.vision_model",
        # Direct transformers-style
        "model.vision_model",
        "model.model.vision_model",
        # Idefics3-style (SmolVLM uses same impl)
        "model.vlm.model.vision_model.encoder",
        "model.vlm_model.model.vision_model.encoder",
    ]
    
    for path in search_paths:
        obj = model
        parts = path.split(".")
        try:
            for part in parts:
                obj = getattr(obj, part)
            print(f"  Found vision encoder at: policy.{path}")
            return obj
        except AttributeError:
            continue
    
    # Fallback: search by module type name (SigLIP, SmolVLM vision, etc.)
    print("  Searching for vision encoder by module type...")
    for name, module in model.named_modules():
        type_name = type(module).__name__.lower()
        if "visionencoder" in type_name or "siglip" in type_name:
            print(f"  Found vision encoder: {name} ({type(module).__name__})")
            return module
        if "visionmodel" in type_name and "text" not in type_name:
            print(f"  Found vision model: {name} ({type(module).__name__})")
            return module
    
    return None


def find_image_keys(dataset):
    """Find observation image keys in the dataset."""
    sample = dataset[0]
    image_keys = [k for k in sample.keys() if "image" in k.lower()]
    if not image_keys:
        image_keys = [k for k in sample.keys() if "pixel" in k.lower()]
    if not image_keys:
        image_keys = [k for k in sample.keys() if isinstance(sample[k], torch.Tensor) and sample[k].dim() == 3 and sample[k].shape[0] == 3]
    return image_keys


_image_key_warning_shown = False


def _match_image_keys(policy_img_keys, dataset_img_keys):
    """
    Match dataset image keys to policy image keys.

    Strategy (in priority order):
      1. Exact match — dataset key exists in policy keys
      2. Suffix match — last segment matches (e.g. both end in 'wrist')
      3. Positional fallback — pair by sorted order (with warning)

    Returns:
        List of ``(dataset_key, policy_key)`` pairs.
    """
    global _image_key_warning_shown
    mapping = []
    unmatched_pkeys = list(policy_img_keys)
    unmatched_dkeys = list(dataset_img_keys)

    # Pass 1: exact match
    for dkey in list(unmatched_dkeys):
        if dkey in unmatched_pkeys:
            mapping.append((dkey, dkey))
            unmatched_pkeys.remove(dkey)
            unmatched_dkeys.remove(dkey)

    # Pass 2: suffix match (last dotted segment, e.g. "camera1" or "wrist")
    for dkey in list(unmatched_dkeys):
        d_suffix = dkey.rsplit(".", 1)[-1]
        for pkey in list(unmatched_pkeys):
            p_suffix = pkey.rsplit(".", 1)[-1]
            if d_suffix == p_suffix:
                mapping.append((dkey, pkey))
                unmatched_pkeys.remove(pkey)
                unmatched_dkeys.remove(dkey)
                break

    # Pass 3: positional fallback
    if unmatched_dkeys and unmatched_pkeys:
        positional = list(zip(sorted(unmatched_dkeys), sorted(unmatched_pkeys)))
        for dkey, pkey in positional:
            mapping.append((dkey, pkey))
            unmatched_pkeys.remove(pkey)
            unmatched_dkeys.remove(dkey)
        if not _image_key_warning_shown:
            pairs = ", ".join(f"'{d}' -> '{p}'" for d, p in positional)
            print(f"    WARNING: No name match for images — mapping by position: {pairs}. "
                  f"Use --image-key if this is wrong.")
            _image_key_warning_shown = True

    return mapping


def _resolve_task_string(sample, dataset=None):
    """
    Get the task/language instruction for a sample.

    Priority:
      1. ``sample["task"]`` — always present in LeRobot datasets
      2. ``dataset.meta.tasks`` — first task in the dataset metadata
      3. Generic fallback
    """
    task = sample.get("task")
    if task is not None:
        if isinstance(task, list):
            task = task[0]
        return task

    # Try dataset metadata
    if dataset is not None:
        try:
            tasks_df = dataset.meta.tasks
            if len(tasks_df) > 0:
                return tasks_df.iloc[0].name
        except (AttributeError, IndexError):
            pass

    return "manipulate object"


def build_policy_batch_from_sample(sample, policy, device, batch_size=1,
                                   image_key_for_grad=None, dataset=None):
    """
    Build a batch dict that matches the policy's expected keys (e.g. observation.images.camera1),
    by mapping from the dataset sample keys (e.g. observation.images.up, observation.images.side).
    Policy expects config.image_features keys; dataset may use different names (up/side vs camera1/2/3).

    Also tokenizes the task string into ``observation.language.tokens`` and
    ``observation.language.attention_mask`` which ``select_action()`` requires.
    """
    policy_img_keys = list(getattr(policy.config, "image_features", {}))
    if not policy_img_keys:
        # Policy has no image_features config; use sample keys as-is
        batch = {}
        for key in sample:
            val = sample[key]
            if isinstance(val, torch.Tensor):
                batch[key] = val.unsqueeze(0).to(device) if batch_size == 1 else val.to(device)
            elif isinstance(val, str):
                batch[key] = [val]
            else:
                batch[key] = val
        return batch, None

    dataset_img_keys = sorted([k for k in sample.keys() if "image" in k.lower() and isinstance(sample.get(k), torch.Tensor)])
    if not dataset_img_keys:
        dataset_img_keys = sorted([k for k in sample.keys() if isinstance(sample.get(k), torch.Tensor) and sample[k].dim() >= 3 and sample[k].shape[0] == 3])

    batch = {}
    for key in sample:
        if key in dataset_img_keys:
            continue  # Fill with policy keys below
        val = sample[key]
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device) if batch_size == 1 else val.to(device)
        elif isinstance(val, str):
            batch[key] = [val]
        else:
            batch[key] = val

    # Map dataset image keys -> policy image keys (exact > suffix > positional)
    img_mapping = _match_image_keys(policy_img_keys, dataset_img_keys)
    grad_pkey = None
    for dkey, pkey in img_mapping:
        img = sample[dkey]
        if batch_size == 1:
            img = img.unsqueeze(0).to(device)
        else:
            img = img.to(device)
        if image_key_for_grad is not None and dkey == image_key_for_grad:
            img = img.clone().detach().requires_grad_(True)
            batch[pkey] = img
            grad_pkey = pkey
        else:
            batch[pkey] = img.clone().detach().requires_grad_(False)

    # --- Resolve task string ---
    task_text = _resolve_task_string(sample, dataset)
    if "task" not in batch:
        batch["task"] = [task_text]

    # --- Tokenize the task description into language tokens ---
    # select_action() expects 'observation.language.tokens' and
    # 'observation.language.attention_mask' which come from tokenizing
    # the task string with the model's built-in tokenizer.
    lang_key = "observation.language.tokens"
    lang_mask_key = "observation.language.attention_mask"
    if lang_key not in batch:
        if isinstance(task_text, list):
            task_text = task_text[0]
        try:
            tokenizer = policy.model.vlm_with_expert.processor.tokenizer
            tok_out = tokenizer(task_text, return_tensors="pt", padding=True)
            batch[lang_key] = tok_out["input_ids"].to(device)
            batch[lang_mask_key] = tok_out["attention_mask"].to(device)
        except Exception as e:
            print(f"    WARNING: Could not tokenize task string: {e}")

    return batch, grad_pkey


def get_episode_frames(dataset, episode_idx, num_frames, image_key):
    """
    Extract evenly-spaced frames from an episode.
    
    Returns list of (frame_index, image_tensor) tuples.
    """
    # Get episode boundaries
    try:
        # LeRobot v3 format
        ep_from = dataset.meta.episodes["dataset_from_index"][episode_idx]
        ep_to = dataset.meta.episodes["dataset_to_index"][episode_idx]
    except (AttributeError, KeyError):
        try:
            # LeRobot v2 format
            ep_from = dataset.episode_data_index["from"][episode_idx].item()
            ep_to = dataset.episode_data_index["to"][episode_idx].item()
        except (AttributeError, KeyError):
            # Fallback: assume ~200 frames per episode
            ep_from = episode_idx * 200
            ep_to = min(ep_from + 200, len(dataset))
    
    ep_length = ep_to - ep_from
    if ep_length <= 0:
        raise ValueError(f"Episode {episode_idx} is empty (from={ep_from}, to={ep_to})")
    
    # Sample evenly spaced frames
    if num_frames >= ep_length:
        indices = list(range(ep_from, ep_to))
    else:
        step = ep_length / num_frames
        indices = [int(ep_from + i * step) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        sample = dataset[idx]
        img = sample[image_key]
        frames.append((idx, img))
    
    print(f"  Episode {episode_idx}: {ep_length} frames, sampled {len(frames)}")
    return frames


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def extract_attention_maps(policy, dataset, episode_idx=0, num_frames=8,
                           image_key=None, device="cpu",
                           method="last-layer", cross_attention=False,
                           show_heads=False, output_dir="./outputs",
                           raw_attention=False):
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

    Returns:
        frames: list of image tensors (C, H, W)
        heatmaps: list of numpy heatmaps (H, W) in [0, 1]
        actions: list of predicted actions (or None)
        cross_attn_heatmaps: list of numpy heatmaps or *None*
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
    task_str = _resolve_task_string(first_sample, dataset)
    print(f"  Task string: \"{task_str}\"")

    # --- Compute positional baseline (once, after frames are loaded so we
    #     know the input aspect ratio for a properly padded baseline) ---
    first_img = frame_pairs[0][1]  # (C, H, W)
    input_hw = (first_img.shape[1], first_img.shape[2])

    baseline_scores = None
    if not raw_attention:
        baseline_scores = compute_positional_baseline(
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

    # --- Run inference and collect attention ---
    print(f"\n[4/4] Running forward passes and extracting attention (method={method})...")
    frames = []
    heatmaps = []
    cross_attn_heatmaps = [] if cross_capture else None
    actions = []

    policy.eval()

    # We need a full policy forward pass when cross-attention is requested
    # (the vision encoder hooks still fire during select_action too)
    use_full_forward = cross_attention and cross_capture is not None

    raw_heads_attn = None   # populated when show_heads is True (first frame only)
    n_patches_h = n_patches_w = 0  # set during vision-encoder-only forward

    for i, (frame_idx, img_tensor) in enumerate(frame_pairs):
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
                                         content_crop=content_crop)
            heatmaps.append(heatmap)

            print(f"    Frame {i}: {n_patches} patches → "
                  f"{grid_h}x{grid_w} grid → {img_h}x{img_w} heatmap ({method})")

            # Per-head grid for first frame
            if show_heads and i == 0 and raw_heads_attn is not None:
                head_path = os.path.join(output_dir, f"per_head_ep{episode_idx:03d}.png")
                create_per_head_grid(
                    img_tensor, raw_heads_attn,
                    (grid_h, grid_w), (img_h, img_w),
                    output_path=head_path,
                    content_crop=content_crop,
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
                cross_hm = attention_to_heatmap(cross_scores, (cs_h, cs_w), (img_h, img_w))
                cross_attn_heatmaps.append(cross_hm)
                print(f"    Frame {i}: Cross-attention captured ({n_vis} vision tokens)")
            else:
                img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
                cross_attn_heatmaps.append(np.ones((img_h, img_w)) * 0.5)
                print(f"    Frame {i}: No cross-attention captured, using uniform")

    # Cleanup
    attn_capture.clear()
    if cross_capture:
        cross_capture.clear()

    return frames, heatmaps, actions, cross_attn_heatmaps


# ---------------------------------------------------------------------------
# 7. Alternative: Pure gradient-based visualization (no hooks needed)
# ---------------------------------------------------------------------------

def gradient_attention_map(policy, dataset, frame_idx, image_key, device="cpu"):
    """
    Compute input-gradient saliency map as a fallback.
    
    This doesn't require hooking into attention — it directly computes
    which input pixels most affect the output actions by backpropagating
    through the entire model.
    
    Interpretation: Bright pixels = changing this pixel would change
    the predicted action the most.
    """
    sample = dataset[frame_idx]
    # Build batch using policy-expected image keys (camera1, camera2, ...)
    # so we don't get "All image features are missing" when dataset uses up/side.
    batch, grad_pkey = build_policy_batch_from_sample(
        sample, policy, device, batch_size=1, image_key_for_grad=image_key,
        dataset=dataset,
    )
    if grad_pkey is None:
        # No policy image key matched; try legacy: use raw sample keys
        img = sample[image_key].unsqueeze(0).to(device).float()
        img.requires_grad_(True)
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else ([v] if isinstance(v, str) else v) for k, v in sample.items()}
        batch[image_key] = img
        if "task" not in batch:
            batch["task"] = ["pick and place"]
        grad_tensor = img
    else:
        grad_tensor = batch[grad_pkey]

    try:
        policy.train()  # Need gradients
        action = policy.select_action(batch)

        # Backpropagate from action norm
        if isinstance(action, dict):
            action_tensor = list(action.values())[0]
        elif isinstance(action, torch.Tensor):
            action_tensor = action
        else:
            return None

        loss = action_tensor.sum()
        loss.backward()

        # Saliency = absolute gradient magnitude across channels
        if grad_tensor.grad is None:
            return None
        saliency = grad_tensor.grad.abs().squeeze(0)
        if saliency.dim() == 3:
            saliency = saliency.mean(dim=0)  # (H, W)
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency.detach().cpu().numpy()

    except Exception as e:
        print(f"  Gradient saliency failed: {e}")
        return None
    finally:
        policy.eval()


# ---------------------------------------------------------------------------
# 8. Model health diagnostics
# ---------------------------------------------------------------------------

def compute_weightwatcher_alpha(policy):
    """
    Run WeightWatcher spectral analysis on each trainable (and frozen
    reference) component of SmolVLA.

    Returns dict keyed by component name, values are lists of
    ``{layer, alpha}`` dicts.  Components that are too small for
    reliable SVD are reported with ``alpha=None``.
    """
    try:
        import weightwatcher as ww
    except ImportError:
        print("  ERROR: weightwatcher not installed. Run:  pip install weightwatcher")
        print("  Skipping spectral alpha analysis.")
        return None

    components = {}

    # --- helper: run ww on a submodel, return list of {layer, alpha} ---
    def _analyze(name, submodel):
        try:
            watcher = ww.WeightWatcher(model=submodel)
            details = watcher.analyze(min_evals=50)
            results = []
            for idx, row in details.iterrows():
                results.append({"layer": idx, "alpha": row.get("alpha", None)})
            components[name] = results
        except Exception as e:
            print(f"    WARNING: WeightWatcher failed on {name}: {e}")
            components[name] = []

    vlm_with_expert = policy.model.vlm_with_expert

    # Trainable components
    print("  Analyzing expert layers...")
    _analyze("Expert (trainable)", vlm_with_expert.lm_expert)

    print("  Analyzing connector...")
    _analyze("Connector (trainable)", vlm_with_expert.get_vlm_model().connector)

    # Projection heads
    proj_names = [n for n, _ in policy.model.named_children()
                  if "proj" in n.lower()]
    if proj_names:
        print(f"  Analyzing projection heads ({', '.join(proj_names)})...")
        for pn in proj_names:
            _analyze(f"Projection/{pn} (trainable)", getattr(policy.model, pn))

    # Frozen reference components
    print("  Analyzing vision encoder (frozen reference)...")
    _analyze("Vision Encoder (frozen)", vlm_with_expert.get_vlm_model().vision_model)

    print("  Analyzing VLM text model (frozen reference)...")
    _analyze("VLM Text Model (frozen)", vlm_with_expert.get_vlm_model().text_model)

    return components


def compute_attention_entropy(attn_maps, num_patches):
    """
    Per-head, per-layer attention entropy as a fraction of maximum
    entropy (``log(num_patches)``).

    Args:
        attn_maps: list of ``(layer_idx, attn_weights)`` tuples.
            Each ``attn_weights`` has shape ``(batch, heads, patches, patches)``
            or ``(heads, patches, patches)``.
        num_patches: total number of patches (for max-entropy normalisation).

    Returns:
        list of ``{layer, head, entropy, entropy_ratio}`` dicts.
    """
    max_entropy = math.log(num_patches) if num_patches > 1 else 1.0
    eps = 1e-8
    results = []

    for layer_idx, attn in sorted(attn_maps, key=lambda x: x[0]):
        # Reduce to (heads, patches, patches)
        while attn.dim() > 3:
            attn = attn[0]
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)

        n_heads = attn.shape[0]
        for h in range(n_heads):
            head_attn = attn[h].float()  # (patches, patches)
            # Entropy per row, then average across rows
            ent = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1).mean().item()
            results.append({
                "layer": layer_idx,
                "head": h,
                "entropy": ent,
                "entropy_ratio": ent / max_entropy,
            })

    return results


def compute_head_redundancy(attn_maps):
    """
    Pairwise cosine similarity between flattened head attention patterns
    within each layer.

    Args:
        attn_maps: list of ``(layer_idx, attn_weights)`` tuples.

    Returns:
        list of ``{layer, mean_redundancy, max_redundancy}`` dicts.
    """
    results = []

    for layer_idx, attn in sorted(attn_maps, key=lambda x: x[0]):
        while attn.dim() > 3:
            attn = attn[0]
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)

        n_heads = attn.shape[0]
        if n_heads < 2:
            results.append({
                "layer": layer_idx,
                "mean_redundancy": 0.0,
                "max_redundancy": 0.0,
            })
            continue

        # Flatten each head's attention to a vector
        flat = attn.float().reshape(n_heads, -1)  # (heads, patches*patches)
        # Normalise
        flat_norm = F.normalize(flat, dim=1)
        # Pairwise cosine similarity
        sim = torch.mm(flat_norm, flat_norm.t())  # (heads, heads)

        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(n_heads, n_heads, dtype=torch.bool), diagonal=1)
        pairwise = sim[mask]
        results.append({
            "layer": layer_idx,
            "mean_redundancy": pairwise.mean().item(),
            "max_redundancy": pairwise.max().item(),
        })

    return results


def classify_health(alpha=None, entropy=None, redundancy=None, thresholds=None):
    """
    Classify a single metric value into a status string and ANSI color.

    Alpha thresholds (hardcoded, RMT-derived):
        <2 overcorrelated, 2-4 healthy, 4-6 undertrained, >6 severely undertrained

    Entropy/redundancy thresholds come from *thresholds* dict.

    Returns:
        ``(ansi_status_str, ansi_color_code, plain_label)``
    """
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if thresholds is None:
        thresholds = {}

    if alpha is not None:
        if alpha < 2:
            return f"{RED}overcorrelated{RESET}", RED, "overcorrelated"
        elif alpha <= 4:
            return f"{GREEN}healthy{RESET}", GREEN, "healthy"
        elif alpha <= 6:
            return f"{YELLOW}undertrained{RESET}", YELLOW, "undertrained"
        else:
            return f"{RED}severely undertrained{RESET}", RED, "severely undertrained"

    if entropy is not None:
        crit = thresholds.get("entropy_critical", 0.95)
        warn = thresholds.get("entropy_warn", 0.8)
        low = thresholds.get("entropy_low", 0.1)
        if entropy >= crit:
            return f"{RED}uniform/dead{RESET}", RED, "uniform/dead"
        elif entropy >= warn:
            return f"{YELLOW}unfocused{RESET}", YELLOW, "unfocused"
        elif entropy <= low:
            return f"{RED}collapsed{RESET}", RED, "collapsed"
        else:
            return f"{GREEN}healthy{RESET}", GREEN, "healthy"

    if redundancy is not None:
        r_crit = thresholds.get("redundancy_critical", 0.9)
        r_warn = thresholds.get("redundancy_warn", 0.7)
        if redundancy >= r_crit:
            return f"{RED}collapsed{RESET}", RED, "collapsed"
        elif redundancy >= r_warn:
            return f"{YELLOW}high redundancy{RESET}", YELLOW, "high redundancy"
        else:
            return f"{GREEN}diverse{RESET}", GREEN, "diverse"

    return "unknown", "", "unknown"


def print_health_report(ww_results, entropy_results, redundancy_results, thresholds):
    """
    Print a formatted terminal table with ANSI color-coded status per
    layer/component.
    """
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    print(f"\n{'=' * 78}")
    print(f"{BOLD}MODEL HEALTH REPORT{RESET}")
    print(f"{'=' * 78}")

    # --- Section 1: Weight Spectral Analysis ---
    if ww_results is not None:
        print(f"\n{BOLD}1. Weight Spectral Analysis (alpha){RESET}")
        print(f"   Fits a power-law to each weight matrix's singular values.")
        print(f"   Alpha (α) measures how well-trained a layer is — values of 2-4 indicate")
        print(f"   strong correlation structure learned during training. High alpha means the")
        print(f"   layer hasn't learned enough structure; low alpha means overcorrelation.")
        print(f"   {DIM}Healthy: 2-4  |  Undertrained: 4-6  |  Overcorrelated: <2  |  Severe: >6{RESET}")
        print(f"   {'Component':<35} {'Wt Matrices':>11}  {'Mean α':>8}  {'Min α':>8}  {'Max α':>8}  Status")
        print(f"   {'-' * 77}")
        for comp_name, layers in ww_results.items():
            alphas = [l["alpha"] for l in layers if l.get("alpha") is not None]
            if not alphas:
                print(f"   {comp_name:<35} {'—':>11}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {DIM}too small{RESET}")
                continue
            mean_a = sum(alphas) / len(alphas)
            min_a = min(alphas)
            max_a = max(alphas)
            status, _, _ = classify_health(alpha=mean_a)
            print(f"   {comp_name:<35} {len(layers):>11}  {mean_a:>8.2f}  {min_a:>8.2f}  {max_a:>8.2f}  {status}")
    else:
        print(f"\n{BOLD}1. Weight Spectral Analysis{RESET}")
        print(f"   {DIM}Skipped (weightwatcher not available){RESET}")

    # --- Section 2: Attention Entropy ---
    if entropy_results:
        print(f"\n{BOLD}2. Attention Entropy (fraction of max){RESET}")
        print(f"   Measures how spread out each attention head's focus is.")
        print(f"   Low entropy means the head attends to very few tokens (collapsed/dead).")
        print(f"   High entropy means the head spreads attention nearly uniformly (unfocused).")
        print(f"   Healthy heads are selective but not degenerate — attending to a meaningful subset.")
        print(f"   {DIM}Collapsed: <{thresholds.get('entropy_low', 0.1):.2f}  |  "
              f"Healthy: {thresholds.get('entropy_low', 0.1):.2f}-{thresholds.get('entropy_warn', 0.8):.2f}  |  "
              f"Unfocused: >{thresholds.get('entropy_warn', 0.8):.2f}  |  "
              f"Dead: >{thresholds.get('entropy_critical', 0.95):.2f}{RESET}")

        from collections import defaultdict
        for comp_name, comp_entries in entropy_results.items():
            print(f"\n   {BOLD}{comp_name}{RESET}")
            by_layer = defaultdict(list)
            for e in comp_entries:
                by_layer[e["layer"]].append(e["entropy_ratio"])

            print(f"   {'Layer':>6}  {'Mean Ent':>10}  {'Min Ent':>10}  {'Max Ent':>10}  Status")
            print(f"   {'-' * 56}")
            for layer in sorted(by_layer.keys()):
                vals = by_layer[layer]
                mean_e = sum(vals) / len(vals)
                min_e = min(vals)
                max_e = max(vals)
                status, _, _ = classify_health(entropy=mean_e, thresholds=thresholds)
                print(f"   {layer:>6}  {mean_e:>10.4f}  {min_e:>10.4f}  {max_e:>10.4f}  {status}")
    else:
        print(f"\n{BOLD}2. Attention Entropy{RESET}")
        print(f"   {DIM}No data{RESET}")

    # --- Section 3: Head Redundancy ---
    if redundancy_results:
        print(f"\n{BOLD}3. Head Redundancy (cosine similarity){RESET}")
        print(f"   Measures how similar the attention heads are to each other within each layer.")
        print(f"   Each layer has multiple heads that should learn different patterns (e.g., one")
        print(f"   head for spatial relations, another for color). High similarity means heads are")
        print(f"   redundant — wasted capacity. Collapsed means nearly identical heads.")
        print(f"   {DIM}Diverse: <{thresholds.get('redundancy_warn', 0.7):.2f}  |  "
              f"High: >{thresholds.get('redundancy_warn', 0.7):.2f}  |  "
              f"Collapsed: >{thresholds.get('redundancy_critical', 0.9):.2f}{RESET}")

        for comp_name, comp_entries in redundancy_results.items():
            print(f"\n   {BOLD}{comp_name}{RESET}")
            print(f"   {'Layer':>6}  {'Mean Sim':>10}  {'Max Sim':>10}  Status")
            print(f"   {'-' * 46}")
            for r in comp_entries:
                status, _, _ = classify_health(redundancy=r["mean_redundancy"], thresholds=thresholds)
                print(f"   {r['layer']:>6}  {r['mean_redundancy']:>10.4f}  {r['max_redundancy']:>10.4f}  {status}")
    else:
        print(f"\n{BOLD}3. Head Redundancy{RESET}")
        print(f"   {DIM}No data{RESET}")

    print(f"\n{'=' * 78}\n")


def _status_emoji(label):
    """Map a plain health label to a markdown-friendly status indicator."""
    if label in ("healthy", "diverse"):
        return "OK"
    elif label in ("undertrained", "unfocused", "high redundancy"):
        return "WARN"
    else:
        return "CRITICAL"


def generate_health_markdown(ww_results, entropy_results, redundancy_results, thresholds):
    """
    Build a Markdown report string from health diagnostics results.
    """
    from collections import defaultdict
    lines = []
    w = lines.append

    w("# Model Health Report\n")

    # --- Section 1: Weight Spectral Analysis ---
    w("## 1. Weight Spectral Analysis (alpha)\n")
    w("Fits a power-law to each weight matrix's singular values. "
       "Alpha measures how well-trained a layer is — values of 2-4 indicate "
       "strong correlation structure learned during training. High alpha means the "
       "layer hasn't learned enough structure; low alpha means overcorrelation.\n")
    if ww_results is not None:
        w("> Healthy: 2-4 | Undertrained: 4-6 | Overcorrelated: <2 | Severe: >6\n")
        w("| Component | Wt Matrices | Mean alpha | Min alpha | Max alpha | Status |")
        w("|-----------|------------:|-----------:|----------:|----------:|--------|")
        for comp_name, layers in ww_results.items():
            alphas = [l["alpha"] for l in layers if l.get("alpha") is not None]
            if not alphas:
                w(f"| {comp_name} | -- | N/A | N/A | N/A | too small |")
                continue
            mean_a = sum(alphas) / len(alphas)
            min_a = min(alphas)
            max_a = max(alphas)
            _, _, label = classify_health(alpha=mean_a)
            badge = _status_emoji(label)
            w(f"| {comp_name} | {len(layers)} | {mean_a:.2f} | {min_a:.2f} | {max_a:.2f} | {badge} {label} |")
    else:
        w("*Skipped (weightwatcher not available)*\n")

    # --- Section 2: Attention Entropy ---
    w("\n## 2. Attention Entropy (fraction of max)\n")
    w("Measures how spread out each attention head's focus is. "
       "Low entropy means the head attends to very few tokens (collapsed/dead). "
       "High entropy means the head spreads attention nearly uniformly (unfocused). "
       "Healthy heads are selective but not degenerate — attending to a meaningful subset.\n")
    if entropy_results:
        low = thresholds.get("entropy_low", 0.1)
        warn = thresholds.get("entropy_warn", 0.8)
        crit = thresholds.get("entropy_critical", 0.95)
        w(f"> Collapsed: <{low:.2f} | Healthy: {low:.2f}-{warn:.2f} | Unfocused: >{warn:.2f} | Dead: >{crit:.2f}\n")

        for comp_name, comp_entries in entropy_results.items():
            w(f"\n### {comp_name}\n")
            by_layer = defaultdict(list)
            for e in comp_entries:
                by_layer[e["layer"]].append(e["entropy_ratio"])

            w("| Layer | Mean Ent | Min Ent | Max Ent | Status |")
            w("|------:|---------:|--------:|--------:|--------|")
            for layer in sorted(by_layer.keys()):
                vals = by_layer[layer]
                mean_e = sum(vals) / len(vals)
                min_e = min(vals)
                max_e = max(vals)
                _, _, label = classify_health(entropy=mean_e, thresholds=thresholds)
                badge = _status_emoji(label)
                w(f"| {layer} | {mean_e:.4f} | {min_e:.4f} | {max_e:.4f} | {badge} {label} |")
    else:
        w("*No data*\n")

    # --- Section 3: Head Redundancy ---
    w("\n## 3. Head Redundancy (cosine similarity)\n")
    w("Measures how similar the attention heads are to each other within each layer. "
       "Each layer has multiple heads that should learn different patterns (e.g., one "
       "head for spatial relations, another for color). High similarity means heads are "
       "redundant — wasted capacity. Collapsed means nearly identical heads.\n")
    if redundancy_results:
        r_warn = thresholds.get("redundancy_warn", 0.7)
        r_crit = thresholds.get("redundancy_critical", 0.9)
        w(f"> Diverse: <{r_warn:.2f} | High: >{r_warn:.2f} | Collapsed: >{r_crit:.2f}\n")

        for comp_name, comp_entries in redundancy_results.items():
            w(f"\n### {comp_name}\n")
            w("| Layer | Mean Sim | Max Sim | Status |")
            w("|------:|---------:|--------:|--------|")
            for r in comp_entries:
                _, _, label = classify_health(redundancy=r["mean_redundancy"], thresholds=thresholds)
                badge = _status_emoji(label)
                w(f"| {r['layer']} | {r['mean_redundancy']:.4f} | {r['max_redundancy']:.4f} | {badge} {label} |")
    else:
        w("*No data*\n")

    return "\n".join(lines)


def plot_health_report(ww_results, entropy_results, redundancy_results, output_path):
    """
    3-panel vertical matplotlib figure saved to *output_path*.

    Panel 1: alpha per layer (bars) with reference lines at 2 and 6
    Panel 2: mean entropy ratio per layer
    Panel 3: mean head redundancy per layer
    """
    from collections import defaultdict

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    # --- Panel 1: Spectral Alpha ---
    ax1 = axes[0]
    if ww_results is not None:
        bar_labels = []
        bar_vals = []
        bar_colors = []
        separator_positions = []
        label_positions = []
        offset = 0

        for comp_name, layers in ww_results.items():
            alphas = [l["alpha"] for l in layers if l.get("alpha") is not None]
            if not alphas:
                continue
            comp_start = offset
            for i, a in enumerate(alphas):
                bar_labels.append(f"L{layers[i]['layer']}")
                bar_vals.append(a)
                if a < 2:
                    bar_colors.append("#e74c3c")       # red
                elif a <= 4:
                    bar_colors.append("#2ecc71")       # green
                elif a <= 6:
                    bar_colors.append("#f39c12")       # yellow
                else:
                    bar_colors.append("#e74c3c")       # red
                offset += 1
            comp_end = offset
            label_positions.append(((comp_start + comp_end - 1) / 2, comp_name))
            separator_positions.append(offset - 0.5)

        # Remove last separator
        if separator_positions:
            separator_positions.pop()

        if bar_vals:
            x = range(len(bar_vals))
            ax1.bar(x, bar_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax1.axhline(y=2, color="green", linestyle="--", linewidth=1, label="α=2 (lower healthy)")
            ax1.axhline(y=6, color="red", linestyle="--", linewidth=1, label="α=6 (upper healthy)")
            ax1.set_ylabel("Alpha (α)")
            ax1.legend(loc="upper right", fontsize=8)

            for sep_x in separator_positions:
                ax1.axvline(x=sep_x, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

            if len(bar_labels) > 30:
                step = max(1, len(bar_labels) // 20)
                ax1.set_xticks(range(0, len(bar_labels), step))
                ax1.set_xticklabels([bar_labels[i] for i in range(0, len(bar_labels), step)],
                                    rotation=45, ha="right", fontsize=6)
            else:
                ax1.set_xticks(x)
                ax1.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=6)

            # Component labels above bars
            for x_center, comp_label in label_positions:
                ax1.text(x_center, 1.02, comp_label, ha="center", va="bottom",
                         fontsize=7, fontweight="bold", transform=ax1.get_xaxis_transform())
        else:
            ax1.text(0.5, 0.5, "No alpha data (layers too small)",
                     ha="center", va="center", transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, "WeightWatcher not available",
                 ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Weight Spectral Analysis — Alpha per Layer", fontweight="bold", pad=20)

    # --- Panel 2: Attention Entropy (grouped by component) ---
    ax2 = axes[1]
    if entropy_results:
        bar_labels = []
        bar_means = []
        bar_colors = []
        separator_positions = []  # x positions for vertical lines between components
        label_positions = []      # (x_center, comp_name) for component labels
        offset = 0

        for comp_name, comp_entries in entropy_results.items():
            by_layer = defaultdict(list)
            for e in comp_entries:
                by_layer[e["layer"]].append(e["entropy_ratio"])
            layers_sorted = sorted(by_layer.keys())
            comp_start = offset

            for layer in layers_sorted:
                vals = by_layer[layer]
                mean_e = sum(vals) / len(vals)
                bar_labels.append(f"L{layer}")
                bar_means.append(mean_e)
                if mean_e >= 0.95:
                    bar_colors.append("#e74c3c")
                elif mean_e >= 0.8:
                    bar_colors.append("#f39c12")
                elif mean_e <= 0.1:
                    bar_colors.append("#e74c3c")
                else:
                    bar_colors.append("#2ecc71")
                offset += 1

            comp_end = offset
            label_positions.append(((comp_start + comp_end - 1) / 2, comp_name))
            if comp_end < sum(len(v) for v in [defaultdict(list)] * 0) or offset > 0:
                separator_positions.append(offset - 0.5)

        # Remove last separator (no line after the last component)
        if separator_positions:
            separator_positions.pop()

        ax2.bar(range(len(bar_means)), bar_means, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax2.axhline(y=0.8, color="#f39c12", linestyle="--", linewidth=1, label="warn (0.8)")
        ax2.axhline(y=0.95, color="#e74c3c", linestyle="--", linewidth=1, label="critical (0.95)")
        ax2.axhline(y=0.1, color="#e74c3c", linestyle=":", linewidth=1, label="collapsed (0.1)")

        # Vertical separators between components
        for sep_x in separator_positions:
            ax2.axvline(x=sep_x, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

        # Component labels at top
        for x_center, comp_label in label_positions:
            ax2.text(x_center, 1.02, comp_label, ha="center", va="bottom",
                     fontsize=7, fontweight="bold", transform=ax2.get_xaxis_transform())

        if len(bar_labels) > 40:
            step = max(1, len(bar_labels) // 30)
            ax2.set_xticks(range(0, len(bar_labels), step))
            ax2.set_xticklabels([bar_labels[i] for i in range(0, len(bar_labels), step)],
                                fontsize=6, rotation=45, ha="right")
        else:
            ax2.set_xticks(range(len(bar_labels)))
            ax2.set_xticklabels(bar_labels, fontsize=6, rotation=45, ha="right")
        ax2.set_ylabel("Entropy / max entropy")
        ax2.set_ylim(0, 1.12)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("Attention Entropy per Layer (mean across heads)", fontweight="bold", pad=20)

    # --- Panel 3: Head Redundancy (grouped by component) ---
    ax3 = axes[2]
    if redundancy_results:
        bar_labels = []
        bar_means = []
        bar_colors = []
        separator_positions = []
        label_positions = []
        offset = 0

        for comp_name, comp_entries in redundancy_results.items():
            comp_start = offset
            for r in comp_entries:
                bar_labels.append(f"L{r['layer']}")
                bar_means.append(r["mean_redundancy"])
                m = r["mean_redundancy"]
                if m >= 0.9:
                    bar_colors.append("#e74c3c")
                elif m >= 0.7:
                    bar_colors.append("#f39c12")
                else:
                    bar_colors.append("#2ecc71")
                offset += 1

            comp_end = offset
            label_positions.append(((comp_start + comp_end - 1) / 2, comp_name))
            separator_positions.append(offset - 0.5)

        # Remove last separator
        if separator_positions:
            separator_positions.pop()

        ax3.bar(range(len(bar_means)), bar_means, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax3.axhline(y=0.7, color="#f39c12", linestyle="--", linewidth=1, label="warn (0.7)")
        ax3.axhline(y=0.9, color="#e74c3c", linestyle="--", linewidth=1, label="critical (0.9)")

        for sep_x in separator_positions:
            ax3.axvline(x=sep_x, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

        for x_center, comp_label in label_positions:
            ax3.text(x_center, 1.02, comp_label, ha="center", va="bottom",
                     fontsize=7, fontweight="bold", transform=ax3.get_xaxis_transform())

        if len(bar_labels) > 40:
            step = max(1, len(bar_labels) // 30)
            ax3.set_xticks(range(0, len(bar_labels), step))
            ax3.set_xticklabels([bar_labels[i] for i in range(0, len(bar_labels), step)],
                                fontsize=6, rotation=45, ha="right")
        else:
            ax3.set_xticks(range(len(bar_labels)))
            ax3.set_xticklabels(bar_labels, fontsize=6, rotation=45, ha="right")
        ax3.set_ylabel("Mean cosine similarity")
        ax3.set_ylim(0, 1.12)
        ax3.legend(loc="upper right", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No redundancy data", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Head Redundancy per Layer", fontweight="bold", pad=20)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved health report plot: {output_path}")


def run_model_health_report(policy, dataset, args):
    """
    Orchestrator for ``--model-health`` mode.

    Step 1: Spectral alpha via WeightWatcher (no data needed).
    Step 2: Sample frames, run forward passes, compute entropy + redundancy.
    Step 3: Print terminal report and save plot.
    """
    device = torch.device(args.device)

    thresholds = {
        "entropy_warn": args.entropy_warn,
        "entropy_critical": args.entropy_critical,
        "entropy_low": args.entropy_low,
        "redundancy_warn": args.redundancy_warn,
        "redundancy_critical": args.redundancy_critical,
    }

    # ------------------------------------------------------------------
    # Step 1: WeightWatcher spectral analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("MODEL HEALTH DIAGNOSTICS")
    print(f"{'=' * 70}")
    print("\n[1/3] Running WeightWatcher spectral analysis...")
    ww_results = compute_weightwatcher_alpha(policy)

    # ------------------------------------------------------------------
    # Step 2: Sample frames, capture per-head attention, compute metrics
    # ------------------------------------------------------------------
    print(f"\n[2/3] Capturing attention maps over {args.health_frames} frames...")

    # Find vision encoder and set up hooks
    vision_encoder = find_vision_encoder(policy)
    if vision_encoder is None:
        print("  ERROR: Could not find vision encoder. Skipping attention metrics.")
        entropy_results = {}
        redundancy_results = {}
    else:
        # Force eager attention on SigLIP
        for mod in vision_encoder.modules():
            if getattr(mod, "config", None) is not None and hasattr(mod.config, "_attn_implementation"):
                mod.config._attn_implementation = "eager"

        # Force per-head weights (disable averaging) on SigLIP MHA
        mha_modules = []
        for mod in vision_encoder.modules():
            if isinstance(mod, torch.nn.MultiheadAttention):
                mha_modules.append((mod, getattr(mod, "average_attn_weights", True)))
                mod.average_attn_weights = False

        # Set up capture objects
        sigclip_capture = SigLIPAttentionCapture()
        sigclip_capture.register_hooks(vision_encoder)

        vlm_with_expert = policy.model.vlm_with_expert

        # SmolVLA bypasses self_attn.forward() — all attention flows through
        # eager_attention_forward(). DecoderAttentionCapture patches that
        # method to capture both self-attention (prefill) and cross-attention
        # (expert → VLM prefix).
        decoder_capture = DecoderAttentionCapture()
        print("  Setting up decoder attention capture (eager_attention_forward)...")
        decoder_capture.register(vlm_with_expert)

        # Resolve image key
        image_key = args.image_key
        if image_key is None:
            image_keys = find_image_keys(dataset)
            if image_keys:
                image_key = image_keys[0]
            else:
                print("  ERROR: No image keys found in dataset.")
                sigclip_capture.clear()
                decoder_capture.clear()
                for mod, orig in mha_modules:
                    mod.average_attn_weights = orig
                entropy_results = {}
                redundancy_results = {}
                image_key = None

        if image_key is not None:
            # Sample frames
            frame_pairs = get_episode_frames(
                dataset, args.episode, args.health_frames, image_key,
            )

            # Determine SigLIP patch grid for entropy normalisation
            patch_size = getattr(vision_encoder, "patch_size", None) or getattr(
                getattr(vision_encoder, "config", None), "patch_size", 14,
            )
            img_size = getattr(
                getattr(vision_encoder, "config", None), "image_size", 384,
            )
            n_patches_side = img_size // patch_size
            sigclip_num_patches = n_patches_side * n_patches_side

            # Component names for the three attention sources
            COMP_SIGLIP = "SigLIP Vision (12L, 12H)"
            COMP_DECODER_SA = "VLM+Expert Joint Self-Attn (16L, 15H)"
            COMP_EXPERT_XA = "Expert-to-VLM Cross-Attn (16L, 8H)"

            all_entropy = {COMP_SIGLIP: [], COMP_DECODER_SA: [], COMP_EXPERT_XA: []}
            all_redundancy = {COMP_SIGLIP: [], COMP_DECODER_SA: [], COMP_EXPERT_XA: []}

            policy.eval()

            for i, (frame_idx, img_tensor) in enumerate(frame_pairs):
                sigclip_capture.reset_maps()
                decoder_capture.reset_maps()
                sample = dataset[frame_idx]

                with torch.no_grad():
                    # Use full policy forward so we capture all layers
                    policy.reset()
                    batch, _ = build_policy_batch_from_sample(
                        sample, policy, device, batch_size=1,
                        image_key_for_grad=None, dataset=dataset,
                    )
                    try:
                        policy.select_action(batch)
                    except Exception as e:
                        print(f"    Frame {i} forward pass error: {e}")
                        continue

                # --- SigLIP ---
                counts = []
                siglip_attns = sigclip_capture.get_all_layer_attentions()
                if siglip_attns:
                    frame_ent = compute_attention_entropy(siglip_attns, sigclip_num_patches)
                    frame_red = compute_head_redundancy(siglip_attns)
                    all_entropy[COMP_SIGLIP].append(frame_ent)
                    all_redundancy[COMP_SIGLIP].append(frame_red)
                    counts.append(f"SigLIP={len(siglip_attns)}")
                else:
                    counts.append("SigLIP=0")

                # --- Decoder Self-Attention (prefill: VLM+Expert concatenated) ---
                sa_attns = decoder_capture.get_self_attn_layers()
                if sa_attns:
                    sa_num_tokens = sa_attns[0][1].shape[-1]
                    frame_ent = compute_attention_entropy(sa_attns, sa_num_tokens)
                    frame_red = compute_head_redundancy(sa_attns)
                    all_entropy[COMP_DECODER_SA].append(frame_ent)
                    all_redundancy[COMP_DECODER_SA].append(frame_red)
                    counts.append(f"Decoder SA={len(sa_attns)}")
                else:
                    counts.append("Decoder SA=0")

                # --- Expert Cross-Attention ---
                xa_attns_raw = decoder_capture.get_cross_attn_layers()
                if xa_attns_raw:
                    # Remap sequential indices → actual layer indices.
                    # During generation, each autoregressive step runs through
                    # all decoder layers, so call_idx % num_layers = layer.
                    num_dec_layers = vlm_with_expert.num_vlm_layers
                    xa_attns = [(idx % num_dec_layers, probs) for idx, probs in xa_attns_raw]
                    xa_num_tokens = xa_attns[0][1].shape[-1]
                    frame_ent = compute_attention_entropy(xa_attns, xa_num_tokens)
                    frame_red = compute_head_redundancy(xa_attns)
                    all_entropy[COMP_EXPERT_XA].append(frame_ent)
                    all_redundancy[COMP_EXPERT_XA].append(frame_red)
                    counts.append(f"Expert XA={len(xa_attns_raw)}")
                else:
                    counts.append("Expert XA=0")

                print(f"    Frame {i}: {', '.join(counts)}")

            # Cleanup
            sigclip_capture.clear()
            decoder_capture.clear()

            # Restore MHA averaging setting
            for mod, orig in mha_modules:
                mod.average_attn_weights = orig

            # Average metrics across frames, per component
            from collections import defaultdict

            entropy_results = {}
            for comp_name, frame_list in all_entropy.items():
                if not frame_list:
                    continue
                ent_accum = defaultdict(lambda: defaultdict(list))
                for frame_ent in frame_list:
                    for e in frame_ent:
                        ent_accum[(e["layer"], e["head"])]["entropy_ratio"].append(e["entropy_ratio"])
                        ent_accum[(e["layer"], e["head"])]["entropy"].append(e["entropy"])

                comp_results = []
                for (layer, head), vals in sorted(ent_accum.items()):
                    comp_results.append({
                        "layer": layer,
                        "head": head,
                        "entropy": sum(vals["entropy"]) / len(vals["entropy"]),
                        "entropy_ratio": sum(vals["entropy_ratio"]) / len(vals["entropy_ratio"]),
                    })
                if comp_results:
                    entropy_results[comp_name] = comp_results

            redundancy_results = {}
            for comp_name, frame_list in all_redundancy.items():
                if not frame_list:
                    continue
                red_accum = defaultdict(lambda: {"mean": [], "max": []})
                for frame_red in frame_list:
                    for r in frame_red:
                        red_accum[r["layer"]]["mean"].append(r["mean_redundancy"])
                        red_accum[r["layer"]]["max"].append(r["max_redundancy"])

                comp_results = []
                for layer in sorted(red_accum.keys()):
                    vals = red_accum[layer]
                    comp_results.append({
                        "layer": layer,
                        "mean_redundancy": sum(vals["mean"]) / len(vals["mean"]),
                        "max_redundancy": sum(vals["max"]) / len(vals["max"]),
                    })
                if comp_results:
                    redundancy_results[comp_name] = comp_results

    # ------------------------------------------------------------------
    # Step 3: Print report and save plot
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating report...")
    print_health_report(ww_results, entropy_results, redundancy_results, thresholds)

    os.makedirs(args.output_dir, exist_ok=True)

    md_path = os.path.join(args.output_dir, "model_health_report.md")
    md_text = generate_health_markdown(ww_results, entropy_results, redundancy_results, thresholds)
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"  Saved markdown report: {md_path}")

    plot_path = os.path.join(args.output_dir, "model_health_report.png")
    plot_health_report(ww_results, entropy_results, redundancy_results, plot_path)

    print(f"Done! Reports saved to {args.output_dir}/")


# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------

def load_defaults():
    """Load defaults from configs/defaults.yaml if it exists."""
    config_path = Path(__file__).parent / "configs" / "defaults.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    return {}


def main():
    defaults = load_defaults()

    parser = argparse.ArgumentParser(
        description="See what SmolVLA's vision encoder is looking at.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_attention.py
  python inspect_attention.py --episode 3 --num-frames 12 --device cuda
  python inspect_attention.py --model path/to/finetuned_checkpoint
        """
    )
    
    parser.add_argument("--model", type=str,
                        default=defaults.get("model", "lerobot/smolvla_base"))
    parser.add_argument("--dataset", type=str,
                        default=defaults.get("dataset", "lerobot/svla_so101_pickplace"))
    parser.add_argument("--episode", type=int,
                        default=defaults.get("episode", 0))
    parser.add_argument("--num-frames", type=int,
                        default=defaults.get("num_frames", 8))
    parser.add_argument("--image-key", type=str, default=None)
    parser.add_argument("--output-dir", type=str,
                        default=defaults.get("output_dir", "./outputs"))
    parser.add_argument("--device", type=str,
                        default=defaults.get("device", "auto"),
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--save-individual", action="store_true",
                        default=defaults.get("save_individual", False))
    parser.add_argument("--method", type=str,
                        default=defaults.get("method", "last-layer"),
                        choices=["last-layer", "rollout", "all-layers"],
                        help="Self-attention aggregation method")
    parser.add_argument("--cross-attention", action="store_true",
                        default=defaults.get("cross_attention", False),
                        help="Capture action-expert → vision cross-attention (slower)")
    parser.add_argument("--show-heads", action="store_true",
                        default=defaults.get("show_heads", False),
                        help="Save a per-head attention grid for the first frame")
    parser.add_argument("--raw-attention", action="store_true",
                        default=defaults.get("raw_attention", False),
                        help="Skip positional baseline subtraction (show raw attention)")

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

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SmolVLA Attention Visualizer")
    print("=" * 70)
    print(f"  Device: {args.device}")
    
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

    # --- Model health mode (early exit) ---
    if args.model_health:
        run_model_health_report(policy, dataset, args)
        return

    # --- Extract attention maps ---
    print(f"\n[Step 3] Extracting attention maps...")
    
    cross_attn_heatmaps = None
    try:
        frames, heatmaps, actions, cross_attn_heatmaps = extract_attention_maps(
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
            saliency = gradient_attention_map(policy, dataset, frame_idx, image_key, args.device)
            if saliency is not None:
                heatmaps.append(saliency)
            else:
                heatmaps.append(np.ones((img_tensor.shape[1], img_tensor.shape[2])) * 0.5)

        actions = []
    
    if not frames:
        print("\nERROR: No frames extracted. Check episode index and dataset.")
        sys.exit(1)
    
    # --- Generate visualizations ---
    print(f"\n[Step 4] Generating visualizations...")
    
    grid_path = os.path.join(args.output_dir, f"attention_grid_ep{args.episode:03d}.png")
    create_visualization_grid(
        frames=frames,
        heatmaps=heatmaps,
        actions=actions,
        cross_attn_heatmaps=cross_attn_heatmaps,
        episode_idx=args.episode,
        output_path=grid_path,
    )
    
    if args.save_individual:
        save_individual_frames(
            frames=frames,
            heatmaps=heatmaps,
            output_dir=os.path.join(args.output_dir, f"episode_{args.episode:03d}"),
            episode_idx=args.episode,
        )
    
    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("DONE!")
    print(f"{'=' * 70}")
    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  Grid visualization: {grid_path}")
    if args.save_individual:
        print(f"  Individual frames:  {args.output_dir}/episode_{args.episode:03d}/")
    
    print()


if __name__ == "__main__":
    main()

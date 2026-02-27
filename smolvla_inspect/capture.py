"""
Attention capture classes for SigLIP, cross-attention, decoder, and GradCAM.

No internal package dependencies.
"""

import torch
import torch.nn.functional as F


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

    def get_per_step_cross_attention(self, vision_start, vision_end,
                                       num_expert_layers=16):
        """
        Group captured cross-attention by denoising step and average
        across layers within each step.

        SmolVLA runs 10 denoising steps, each invoking all 16 expert
        layers.  ``step = call_idx // num_expert_layers``.

        Returns:
            list of Tensors ``(n_vision_tokens,)`` — one per denoising
            step, or *None* if no cross-attention was captured.
        """
        if not self.cross_attn_weights:
            return None

        # Group by step
        from collections import defaultdict
        steps = defaultdict(list)
        for call_idx, probs in self.cross_attn_weights:
            step = call_idx // num_expert_layers
            # probs: (B, heads, q_len, k_len)
            avg = probs[0].mean(dim=0).mean(dim=0)  # (k_len,)
            vision_avg = avg[vision_start:vision_end]
            steps[step].append(vision_avg)

        if not steps:
            return None

        result = []
        for step_idx in sorted(steps.keys()):
            tensors = steps[step_idx]
            stacked = torch.stack(tensors, dim=0).mean(dim=0)  # (n_vision_tokens,)
            result.append(stacked)

        return result

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

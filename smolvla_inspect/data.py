"""
Dataset and model helper functions.

No internal package dependencies.
"""

import torch


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


def parse_image_map(image_map_str):
    """
    Parse a ``--image-map`` CLI string into a dict.

    Accepts either short suffixes or full dotted keys::

        "front=camera2,side=camera3"
        "observation.images.front=observation.images.camera2"

    Returns:
        dict mapping dataset key (or suffix) → policy key (or suffix).
    """
    if not image_map_str:
        return {}
    result = {}
    for pair in image_map_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        left, right = pair.split("=", 1)
        result[left.strip()] = right.strip()
    return result


def _match_image_keys(policy_img_keys, dataset_img_keys, image_map=None):
    """
    Match dataset image keys to policy image keys.

    Strategy (in priority order):
      0. Explicit ``image_map`` overrides — user-provided mappings
      1. Exact match — dataset key exists in policy keys
      2. Suffix match — last segment matches (e.g. both end in 'wrist')
      3. Positional fallback — pair by sorted order (with warning)

    Args:
        image_map: optional dict mapping dataset key (or suffix) → policy
            key (or suffix).  Both full keys (``observation.images.front``)
            and short suffixes (``front``) are accepted.

    Returns:
        List of ``(dataset_key, policy_key)`` pairs.
    """
    global _image_key_warning_shown
    mapping = []
    unmatched_pkeys = list(policy_img_keys)
    unmatched_dkeys = list(dataset_img_keys)

    # Pass 0: explicit user overrides
    if image_map:
        for dkey in list(unmatched_dkeys):
            d_suffix = dkey.rsplit(".", 1)[-1]
            # Try full key first, then suffix
            target = image_map.get(dkey) or image_map.get(d_suffix)
            if target is None:
                continue
            # Resolve target: full key match or suffix match in policy keys
            matched_pkey = None
            if target in unmatched_pkeys:
                matched_pkey = target
            else:
                for pkey in unmatched_pkeys:
                    if pkey.rsplit(".", 1)[-1] == target:
                        matched_pkey = pkey
                        break
            if matched_pkey is not None:
                mapping.append((dkey, matched_pkey))
                unmatched_pkeys.remove(matched_pkey)
                unmatched_dkeys.remove(dkey)

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
                  f"Use --image-map if this is wrong.")
            _image_key_warning_shown = True

    return mapping


def _resolve_task_string(sample, dataset=None, task_override=None):
    """
    Get the task/language instruction for a sample.

    Priority:
      1. ``sample["task"]`` — always present in LeRobot datasets
      2. ``dataset.meta.tasks`` — first task in the dataset metadata
      3. Generic fallback
    """
    if task_override is not None:
        return task_override

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


def get_alternative_task_string(original_task, dataset=None):
    """
    Select a contrasting task string for language-conditional comparison.

    Tiers:
      1. Different task from ``dataset.meta.tasks`` (if >1 task)
      2. ``"do not " + original_task`` (semantic negation)
      3. ``"observe the scene"`` (always available)

    Returns:
        ``(alt_task, tier)`` — the alternative string and which tier was used.
    """
    # Tier 1: different task from dataset metadata
    if dataset is not None:
        try:
            tasks_df = dataset.meta.tasks
            if len(tasks_df) > 1:
                for idx in range(len(tasks_df)):
                    candidate = tasks_df.iloc[idx].name
                    if candidate != original_task:
                        return candidate, 1
        except (AttributeError, IndexError):
            pass

    # Tier 2: semantic negation
    if original_task and original_task.strip():
        return f"do not {original_task}", 2

    # Tier 3: fallback
    return "observe the scene", 3


def build_policy_batch_from_sample(sample, policy, device, batch_size=1,
                                   image_key_for_grad=None, dataset=None,
                                   task_override=None, state_requires_grad=False,
                                   image_map=None):
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
    img_mapping = _match_image_keys(policy_img_keys, dataset_img_keys, image_map=image_map)
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
    task_text = _resolve_task_string(sample, dataset, task_override=task_override)
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

    # --- Enable gradient on state tensor if requested (for F4: vision vs state) ---
    if state_requires_grad:
        state_key = "observation.state"
        if state_key in batch:
            batch[state_key] = batch[state_key].clone().detach().requires_grad_(True)

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

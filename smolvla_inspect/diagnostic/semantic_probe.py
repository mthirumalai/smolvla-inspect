"""Semantic and QK probes for spatial-vs-object diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from ..gradient import compute_gradcam_map, extract_siglip_last_layer_features
from .models import (
    CounterfactualResult,
    QKHeadSummary,
    QKProbeReport,
    SceneSegmentation,
    SemanticCounterfactualSummary,
    SemanticFrameSummary,
    SemanticProbeReport,
)

_DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-512"
_SIGLIP_TEXT_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


def semantic_candidate_labels(
    scene: SceneSegmentation,
    target_object: str | None = None,
) -> list[str]:
    """Collect unique non-gripper labels, keeping *target_object* first."""
    labels = []
    if target_object is not None:
        labels.append(target_object)
    for obj in scene.objects:
        label = obj.label
        if "gripper" in label.lower():
            continue
        if label not in labels:
            labels.append(label)
    return labels


def _infer_siglip_model_id(policy) -> str:
    """Best-effort guess of the matching SigLIP text tower."""
    try:
        from ..data import find_vision_encoder

        vision_encoder = find_vision_encoder(policy)
    except Exception:
        vision_encoder = None

    if vision_encoder is None:
        return _DEFAULT_SIGLIP_MODEL

    cfg = getattr(vision_encoder, "config", None)
    image_size = getattr(cfg, "image_size", 512)
    patch_size = getattr(cfg, "patch_size", 16)
    hidden_size = getattr(cfg, "hidden_size", 768)
    if hidden_size >= 1024:
        return f"google/siglip-large-patch{patch_size}-{image_size}"
    return f"google/siglip-base-patch{patch_size}-{image_size}"


def _text_device(device: str | torch.device) -> str:
    device_str = str(device)
    if device_str.startswith(("cuda", "cpu")):
        return device_str
    return "cpu"


def _get_siglip_text_encoder(model_id: str, device: str):
    key = (model_id, device)
    if key in _SIGLIP_TEXT_CACHE:
        return _SIGLIP_TEXT_CACHE[key]

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    _SIGLIP_TEXT_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _unique_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for label in labels:
        if not label or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result


def _resize_map(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr.astype(np.float32, copy=False)
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=target_shape, mode="bilinear", align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool, copy=False)
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=target_shape, mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy() > 0.5


def _normalize_map(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if not valid_mask.any():
        return out
    valid = values[valid_mask].astype(np.float32)
    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmax > vmin:
        out[valid_mask] = (valid - vmin) / (vmax - vmin)
    return out


def _topk_mask(values: np.ndarray, valid_mask: np.ndarray, pct: float = 0.05) -> np.ndarray:
    mask = np.zeros_like(values, dtype=bool)
    if not valid_mask.any():
        return mask
    flat = values[valid_mask]
    k = max(4, int(np.ceil(flat.size * pct)))
    k = min(k, flat.size)
    threshold = np.partition(flat, flat.size - k)[flat.size - k]
    mask[valid_mask] = values[valid_mask] >= threshold
    return mask


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    region = mask.astype(bool)
    if not region.any():
        return None
    return float(values[region].mean())


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float | None:
    region = mask.astype(bool)
    if not region.any():
        return None
    return float(values[region].max())


def _weighted_mean(values: np.ndarray, weights: np.ndarray, valid_mask: np.ndarray) -> float | None:
    weights = np.where(valid_mask, np.maximum(weights, 0.0), 0.0).astype(np.float32)
    total = float(weights.sum())
    if total <= 1e-8:
        return None
    return float((values * weights).sum() / total)


def _pearson_corr(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = valid_mask.astype(bool)
    if mask.sum() < 4:
        return 0.0
    av = a[mask].astype(np.float32)
    bv = b[mask].astype(np.float32)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _map_mass(values: np.ndarray, region_mask: np.ndarray, valid_mask: np.ndarray) -> float:
    if not valid_mask.any():
        return 0.0
    norm = _normalize_map(values, valid_mask)
    total = float(norm[valid_mask].sum())
    if total <= 1e-8:
        return 0.0
    return float(norm[region_mask & valid_mask].sum() / total)


def _clone_sample(sample: dict, image_key: str) -> dict:
    cloned = dict(sample)
    cloned[image_key] = sample[image_key].clone()
    return cloned


def _tensor_to_hwc(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _hwc_to_tensor(arr: np.ndarray, device: str = "cpu") -> torch.Tensor:
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def _shift_mask(mask: np.ndarray, shift_pixels: tuple[int, int]) -> np.ndarray:
    dx, dy = shift_pixels
    shifted = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return shifted
    ny = ys + dy
    nx = xs + dx
    valid = (ny >= 0) & (ny < mask.shape[0]) & (nx >= 0) & (nx < mask.shape[1])
    shifted[ny[valid], nx[valid]] = True
    return shifted


def _make_relocated_sample(sample: dict, image_key: str, segmentation: SceneSegmentation,
                           target_object: str, shift_pixels: tuple[int, int]) -> tuple[dict | None, dict]:
    img_tensor = sample[image_key]
    img_hwc = _tensor_to_hwc(img_tensor)
    h, w = img_hwc.shape[:2]
    obj_mask = segmentation.get_mask(target_object)
    if obj_mask is None:
        return None, {}
    obj_mask = _resize_mask(obj_mask, (h, w))
    dx, dy = shift_pixels
    modified_hwc = img_hwc.copy()
    ys, xs = np.where(obj_mask)
    if len(ys) == 0:
        return None, {}

    fill_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    modified_hwc[obj_mask] = fill_color
    for oy, ox in zip(ys, xs):
        ny = oy + dy
        nx = ox + dx
        if 0 <= ny < h and 0 <= nx < w:
            modified_hwc[ny, nx] = img_hwc[oy, ox]
    modified_hwc = np.clip(modified_hwc, 0.0, 1.0)

    moved_mask = _shift_mask(obj_mask, shift_pixels)
    old_anchor_mask = obj_mask & ~moved_mask
    moved_target_mask = moved_mask & ~obj_mask
    if not old_anchor_mask.any():
        old_anchor_mask = obj_mask
    if not moved_target_mask.any():
        moved_target_mask = moved_mask

    new_sample = _clone_sample(sample, image_key)
    new_sample[image_key] = _hwc_to_tensor(modified_hwc, device=str(img_tensor.device))
    return new_sample, {
        "original_mask": obj_mask,
        "old_anchor_mask": old_anchor_mask,
        "moved_target_mask": moved_target_mask,
    }


def compute_patch_text_similarity(
    policy,
    sample,
    dataset,
    image_key,
    device,
    labels: list[str],
    *,
    image_map=None,
    capture_qk: bool = False,
):
    """Extract final-layer patch features and cosine maps for *labels*."""
    labels = _unique_labels(labels)
    if not labels:
        return None

    features = extract_siglip_last_layer_features(
        policy, sample, dataset, image_key, device,
        image_map=image_map, capture_qk=capture_qk,
    )
    if features is None:
        return None

    model_id = _infer_siglip_model_id(policy)
    device_str = _text_device(device)
    try:
        tokenizer, text_model = _get_siglip_text_encoder(model_id, device_str)
    except Exception as exc:
        if model_id != _DEFAULT_SIGLIP_MODEL:
            try:
                model_id = _DEFAULT_SIGLIP_MODEL
                tokenizer, text_model = _get_siglip_text_encoder(model_id, device_str)
            except Exception:
                print(f"    WARNING: SigLIP text encoder unavailable for semantic probe ({exc})")
                return None
        else:
            print(f"    WARNING: SigLIP text encoder unavailable for semantic probe ({exc})")
            return None

    with torch.no_grad():
        inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device_str) for key, value in inputs.items()}
        text_embeddings = text_model.get_text_features(**inputs).float()
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

    patch_features = torch.from_numpy(features["patch_features"]).float()
    if patch_features.shape[-1] != text_embeddings.shape[-1]:
        print(
            "    WARNING: SigLIP patch/text dimensions do not match for semantic probe "
            f"({patch_features.shape[-1]} vs {text_embeddings.shape[-1]})"
        )
        return None
    similarity_maps: dict[str, np.ndarray] = {}
    for idx, label in enumerate(labels):
        sim = torch.tensordot(patch_features, text_embeddings[idx].cpu(), dims=([-1], [0]))
        similarity_maps[label] = sim.cpu().numpy()

    result = dict(features)
    result["labels"] = labels
    result["model_id"] = model_id
    result["similarity_maps"] = similarity_maps
    return result


def probe_semantic_frame(
    policy,
    sample,
    dataset,
    image_key,
    device,
    *,
    target_object: str | None,
    candidate_labels: list[str],
    attention_heatmap: np.ndarray | None = None,
    gradcam_map: np.ndarray | None = None,
    image_map=None,
    frame_id: str = "primary",
):
    """Build a semantic summary for one frame and return internal artifacts."""
    if target_object is None:
        return None, None

    artifact = compute_patch_text_similarity(
        policy, sample, dataset, image_key, device, candidate_labels,
        image_map=image_map, capture_qk=False,
    )
    if artifact is None:
        return None, None

    similarity_maps = artifact["similarity_maps"]
    valid_mask = artifact["valid_patch_mask"]
    grid_shape = artifact["grid_size"]
    target_map = similarity_maps.get(target_object)
    if target_map is None:
        return None, None

    if gradcam_map is None:
        try:
            gradcam_map = compute_gradcam_map(
                policy, sample, dataset, image_key, device, image_map=image_map,
            )
        except Exception:
            gradcam_map = None

    gradcam_grid = _resize_map(gradcam_map, grid_shape) if gradcam_map is not None else None
    attention_grid = _resize_map(attention_heatmap, grid_shape) if attention_heatmap is not None else None

    if gradcam_grid is not None:
        gradcam_top = _topk_mask(gradcam_grid, valid_mask)
    else:
        gradcam_top = np.zeros(grid_shape, dtype=bool)
    if attention_grid is not None:
        attention_top = _topk_mask(attention_grid, valid_mask)
    else:
        attention_top = np.zeros(grid_shape, dtype=bool)

    causal_mask = np.zeros(grid_shape, dtype=bool)
    if gradcam_top.any() and attention_top.any():
        causal_mask = gradcam_top & attention_top
        if not causal_mask.any():
            causal_mask = gradcam_top
    elif gradcam_top.any():
        causal_mask = gradcam_top
    elif attention_top.any():
        causal_mask = attention_top
    else:
        causal_mask = valid_mask.copy()

    if gradcam_grid is not None:
        causal_weights = _normalize_map(gradcam_grid, valid_mask)
    elif attention_grid is not None:
        causal_weights = _normalize_map(attention_grid, valid_mask)
    else:
        causal_weights = valid_mask.astype(np.float32)
    causal_weights = np.where(causal_mask, causal_weights, 0.0)

    target_peak = _masked_max(target_map, valid_mask)
    target_mean_causal = _masked_mean(target_map, causal_mask & valid_mask)
    causal_alignment = _weighted_mean(target_map, causal_weights, valid_mask)
    background_mask = valid_mask & ~causal_mask
    background_mean = _masked_mean(target_map, background_mask)
    background_gap = None
    if target_mean_causal is not None and background_mean is not None:
        background_gap = float(target_mean_causal - background_mean)

    non_target_scores = []
    for label, sim_map in similarity_maps.items():
        if label == target_object:
            continue
        score = _masked_mean(sim_map, causal_mask & valid_mask)
        if score is not None:
            non_target_scores.append((label, score))

    best_non_target_label = None
    target_margin = None
    if target_mean_causal is not None and non_target_scores:
        best_non_target_label, best_non_target_score = max(non_target_scores, key=lambda item: item[1])
        target_margin = float(target_mean_causal - best_non_target_score)

    summary_bits = []
    if target_margin is not None:
        if target_margin > 0.10:
            summary_bits.append(
                f"Causal patches align more strongly with '{target_object}' than other detected objects "
                f"(margin={target_margin:.3f})."
            )
        elif target_margin < 0.0:
            summary_bits.append(
                f"Causal patches are not semantically distinctive for '{target_object}' "
                f"(margin={target_margin:.3f})."
            )
    if background_gap is not None:
        if background_gap > 0.10:
            summary_bits.append(
                f"Target semantics concentrate on causal patches rather than the low-causal background "
                f"(gap={background_gap:.3f})."
            )
        elif background_gap < 0.0:
            summary_bits.append(
                f"Target semantics are no stronger on causal patches than on the low-causal background "
                f"(gap={background_gap:.3f})."
            )

    frame = SemanticFrameSummary(
        frame_id=frame_id,
        target_object=target_object,
        candidate_labels=candidate_labels,
        best_non_target_label=best_non_target_label,
        target_semantic_peak=target_peak,
        target_semantic_mean_on_causal_patches=target_mean_causal,
        target_margin_over_best_non_target=target_margin,
        causal_semantic_alignment=causal_alignment,
        background_semantic_gap=background_gap,
        summary=" ".join(summary_bits) if summary_bits else "Semantic evidence was weak or unavailable on the causal patches.",
    )
    internal = {
        "target_map": target_map,
        "valid_patch_mask": valid_mask,
        "causal_mask": causal_mask,
        "causal_weights": causal_weights,
        "similarity_maps": similarity_maps,
        "artifact": artifact,
    }
    return frame, internal


def summarize_semantic_probe(report: SemanticProbeReport | None) -> str:
    """Short summary string for prompts and logs."""
    if report is None or report.primary_frame is None:
        return "Semantic patch-to-text probe unavailable."
    frame = report.primary_frame
    bits = [f"Target object: {report.target_object or 'N/A'}."]
    if frame.target_margin_over_best_non_target is not None:
        bits.append(f"Target margin over best non-target: {frame.target_margin_over_best_non_target:.3f}.")
    if frame.background_semantic_gap is not None:
        bits.append(f"Background gap: {frame.background_semantic_gap:.3f}.")
    bits.append(frame.summary)
    return " ".join(bits)


def build_semantic_probe_report(
    target_object: str | None,
    primary_frame: SemanticFrameSummary | None,
    cf_results: list[CounterfactualResult],
) -> SemanticProbeReport | None:
    """Assemble the primary-frame and counterfactual semantic summaries."""
    if target_object is None and primary_frame is None:
        return None

    counterfactuals: dict[str, SemanticCounterfactualSummary] = {}
    for result in cf_results:
        metrics = result.metrics or {}
        if result.test_type == "object_relocation" and "semantic_follow_ratio" in metrics:
            follow = float(metrics.get("semantic_follow_ratio", 0.0))
            anchor = float(metrics.get("semantic_anchor_ratio", 0.0))
            summary = (
                f"After relocation, target semantics follow the moved object "
                f"(follow={follow:.2f}, anchor={anchor:.2f})."
                if follow >= anchor
                else f"After relocation, target semantics stay closer to the old anchor "
                f"(anchor={anchor:.2f}, follow={follow:.2f})."
            )
            counterfactuals[result.test_type] = SemanticCounterfactualSummary(
                test_type=result.test_type,
                summary=summary,
                metrics=metrics,
            )
        elif result.test_type == "occlusion_targeted" and "occlusion_target_semantic_drop" in metrics:
            drop = float(metrics.get("occlusion_target_semantic_drop", 0.0))
            summary = (
                f"Occluding the target reduces target-semantic evidence in the target region "
                f"(drop={drop:.3f})."
                if drop > 0
                else f"Occluding the target does not reduce target-semantic evidence "
                f"(drop={drop:.3f})."
            )
            counterfactuals[result.test_type] = SemanticCounterfactualSummary(
                test_type=result.test_type,
                summary=summary,
                metrics=metrics,
            )

    if primary_frame is not None:
        summary = primary_frame.summary
    elif counterfactuals:
        summary = "Semantic evidence is available only through counterfactual follow-through probes."
    else:
        summary = "Semantic patch-to-text probe unavailable."

    frames = [primary_frame] if primary_frame is not None else []
    return SemanticProbeReport(
        target_object=target_object,
        summary=summary,
        primary_frame=primary_frame,
        frames=frames,
        counterfactuals=counterfactuals,
    )


def build_qk_probe_report(
    policy,
    sample,
    dataset,
    image_key,
    device,
    *,
    target_object: str | None,
    scene: SceneSegmentation,
    target_semantic_map: np.ndarray | None,
    positional_baseline: np.ndarray | None = None,
    relocation_result: CounterfactualResult | None = None,
    image_map=None,
) -> QKProbeReport | None:
    """Summarize last-layer SigLIP QK behavior with semantic-vs-positional evidence."""
    if target_object is None or target_semantic_map is None:
        return None

    artifact = compute_patch_text_similarity(
        policy, sample, dataset, image_key, device, [target_object],
        image_map=image_map, capture_qk=True,
    )
    if artifact is None or "qk_key_maps" not in artifact:
        return None

    valid_mask = artifact["valid_patch_mask"]
    grid_shape = artifact["grid_size"]
    qk_maps = artifact["qk_key_maps"]
    target_map = _resize_map(target_semantic_map, grid_shape)
    positional_grid = (
        _resize_map(positional_baseline, grid_shape)
        if positional_baseline is not None
        else np.zeros(grid_shape, dtype=np.float32)
    )

    target_mask = scene.get_mask(target_object)
    target_grid_mask = (
        _resize_mask(target_mask, grid_shape) & valid_mask
        if target_mask is not None
        else np.zeros(grid_shape, dtype=bool)
    )
    background_grid_mask = _resize_mask(scene.background_mask, grid_shape) & valid_mask

    relocation_old_mask = None
    relocation_new_mask = None
    relocated_qk_maps = None
    if relocation_result is not None:
        metrics = relocation_result.metrics or {}
        shift = metrics.get("shift_pixels")
        if isinstance(shift, (list, tuple)) and len(shift) == 2:
            relocated_sample, relocated_masks = _make_relocated_sample(
                sample, image_key, scene, target_object, (int(shift[0]), int(shift[1])),
            )
            if relocated_sample is not None:
                relocated_artifact = compute_patch_text_similarity(
                    policy, relocated_sample, dataset, image_key, device, [target_object],
                    image_map=image_map, capture_qk=True,
                )
                if relocated_artifact is not None and "qk_key_maps" in relocated_artifact:
                    relocated_qk_maps = relocated_artifact["qk_key_maps"]
                    relocation_old_mask = _resize_mask(relocated_masks["old_anchor_mask"], grid_shape) & valid_mask
                    relocation_new_mask = _resize_mask(relocated_masks["moved_target_mask"], grid_shape) & valid_mask

    head_summaries: list[QKHeadSummary] = []
    for head_index, head_map in enumerate(qk_maps):
        semantic_corr = _pearson_corr(head_map, target_map, valid_mask)
        positional_corr = _pearson_corr(head_map, positional_grid, valid_mask)
        target_mass = _map_mass(head_map, target_grid_mask, valid_mask)
        background_mass = _map_mass(head_map, background_grid_mask, valid_mask)

        old_anchor_mass = None
        moved_object_mass = None
        relocation_bias = 0.0
        if relocated_qk_maps is not None and relocation_old_mask is not None and relocation_new_mask is not None:
            old_anchor_mass = _map_mass(relocated_qk_maps[head_index], relocation_old_mask, valid_mask)
            moved_object_mass = _map_mass(relocated_qk_maps[head_index], relocation_new_mask, valid_mask)
            relocation_bias = moved_object_mass - old_anchor_mass

        semantic_adv = semantic_corr - positional_corr
        if relocation_bias > 0.05:
            semantic_adv += 0.10
        elif relocation_bias < -0.05:
            semantic_adv -= 0.10

        if semantic_adv > 0.08:
            head_type = "semantic"
        elif semantic_adv < -0.08:
            head_type = "positional"
        else:
            head_type = "mixed"

        score = max(abs(semantic_adv), abs(relocation_bias), abs(target_mass - background_mass))
        head_summaries.append(QKHeadSummary(
            head_index=head_index,
            head_type=head_type,
            score=score,
            semantic_map_correlation=semantic_corr,
            positional_baseline_correlation=positional_corr,
            target_region_logit_mass=target_mass,
            background_logit_mass=background_mass,
            old_anchor_logit_mass=old_anchor_mass,
            moved_object_logit_mass=moved_object_mass,
        ))

    if not head_summaries:
        return None

    semantic_count = sum(1 for head in head_summaries if head.head_type == "semantic")
    positional_count = sum(1 for head in head_summaries if head.head_type == "positional")
    mixed_count = sum(1 for head in head_summaries if head.head_type == "mixed")
    total = len(head_summaries)

    semantic_fraction = semantic_count / total
    positional_fraction = positional_count / total
    mixed_fraction = mixed_count / total

    if semantic_fraction > positional_fraction + 0.15:
        dominant = "semantic"
        summary = (
            f"Most last-layer SigLIP heads align more with target semantics than with the positional baseline "
            f"({semantic_fraction:.0%} semantic vs {positional_fraction:.0%} positional)."
        )
    elif positional_fraction > semantic_fraction + 0.15:
        dominant = "positional"
        summary = (
            f"Most last-layer SigLIP heads align more with the positional baseline than with target semantics "
            f"({positional_fraction:.0%} positional vs {semantic_fraction:.0%} semantic)."
        )
    else:
        dominant = "mixed"
        summary = (
            f"Last-layer SigLIP heads split between semantic and positional signals "
            f"({semantic_fraction:.0%} semantic, {positional_fraction:.0%} positional, {mixed_fraction:.0%} mixed)."
        )

    top_heads = sorted(head_summaries, key=lambda head: head.score, reverse=True)[:3]
    return QKProbeReport(
        layer="siglip_last",
        summary=summary,
        dominant_head_type=dominant,
        semantic_head_fraction=semantic_fraction,
        positional_head_fraction=positional_fraction,
        mixed_head_fraction=mixed_fraction,
        top_heads=top_heads,
    )


def summarize_qk_probe(report: QKProbeReport | None) -> str:
    """Short summary string for prompts and logs."""
    if report is None:
        return "QK decomposition unavailable."
    return report.summary

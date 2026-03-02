"""LLM streaming analysis endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..config import Settings
from ..models.schemas import LLMAnalyzeRequest, LLMConfig
from ..services import (
    comparison,
    data_loader,
    llm_service,
    prompt_templates,
    run_scanner,
    stats_engine,
)

router = APIRouter(prefix="/api/llm", tags=["llm"])


def _settings() -> Settings:
    from ..main import get_settings
    return get_settings()


@router.get("/config", response_model=LLMConfig)
async def get_llm_config():
    s = _settings()
    return LLMConfig(
        provider=s.llm_provider,
        model=s.llm_model,
        api_key="***" if s.llm_api_key else "",
        base_url=s.llm_base_url,
    )


@router.post("/config")
async def set_llm_config(config: LLMConfig):
    s = _settings()
    s.llm_provider = config.provider
    s.llm_model = config.model
    if config.api_key and config.api_key != "***":
        s.llm_api_key = config.api_key
    s.llm_base_url = config.base_url
    return {"status": "ok"}


@router.get("/prompts/{analysis_type}")
async def get_prompt_template(analysis_type: str):
    template = prompt_templates.get_template(analysis_type)
    if not template:
        raise HTTPException(404, f"No template for {analysis_type}")
    return {"analysis_type": analysis_type, "template": template}


@router.post("/analyze")
async def analyze(req: LLMAnalyzeRequest):
    settings = _settings()

    # Load run data for context
    try:
        run_dir, manifest = run_scanner.load_manifest(settings.base_dir, req.run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {req.run_id} not found")

    model_info = manifest.get("model_info", {})
    dataset_info = manifest.get("dataset_info", {})

    # Compute stats if requested
    computed_stats_text = ""
    if req.include_stats:
        computed_stats_text = _compute_stats_text(req, run_dir, manifest, settings)

    # Build prompt
    if req.prompt:
        prompt = req.prompt
    elif req.analysis_type == "run_insights":
        prompt = _build_run_insights_prompt(run_dir, manifest, computed_stats_text)
    elif req.analysis_type == "multi_run_comparison":
        prompt = _build_multi_run_prompt(req, run_dir, manifest, settings, computed_stats_text)
    else:
        template = prompt_templates.get_template(req.analysis_type)
        prompt = prompt_templates.format_template(
            template,
            model_id=model_info.get("model_id", "unknown"),
            dataset_id=dataset_info.get("dataset_id", "unknown"),
            task_string=dataset_info.get("task_string", ""),
            episode_idx=dataset_info.get("episode_idx", 0),
            num_frames=dataset_info.get("num_frames", 0),
            method=manifest.get("cli_args", {}).get("method", "rollout"),
            action_dim_names=str(dataset_info.get("action_dim_names", [])),
            computed_stats=computed_stats_text,
        )

    # Collect relevant images
    image_paths: list[Path] = _collect_images(req, run_dir, manifest, settings)

    async def event_generator():
        async for token in llm_service.stream_analysis(prompt, image_paths, settings):
            # JSON-encode so newlines in tokens survive SSE transport
            yield f"data: {json.dumps(token)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _compute_stats_text(req: LLMAnalyzeRequest, run_dir: Path,
                        manifest: dict, settings: Settings) -> str:
    """Compute and format stats based on analysis type."""
    if req.analysis_type == "run_insights":
        summary = stats_engine.compute_run_summary_stats(run_dir, manifest)
        return stats_engine.format_stats_for_prompt(summary)

    if req.analysis_type == "multi_run_comparison":
        run_dirs = [run_dir]
        manifests = [manifest]
        for rid in (req.compare_run_ids or []):
            if rid == req.run_id:
                continue
            try:
                rd, mf = run_scanner.load_manifest(settings.base_dir, rid)
                run_dirs.append(rd)
                manifests.append(mf)
            except FileNotFoundError:
                continue
        if len(run_dirs) >= 2:
            cross = stats_engine.compute_cross_run_stats(run_dirs, manifests)
            return stats_engine.format_stats_for_prompt(cross.get("deltas", {}))
        return ""

    # Per-viz stats
    viz_type = req.viz_type or req.analysis_type.replace("single_viz_", "")
    stats = _compute_single_viz_stats(run_dir, viz_type)
    return stats_engine.format_stats_for_prompt(stats) if stats else ""


def _compute_single_viz_stats(run_dir: Path, viz_type: str) -> dict | None:
    """Compute stats for a single viz type."""
    heatmap_types = {
        "self_attention": lambda: data_loader.load_self_attention_heatmaps(run_dir),
        "cross_attention": lambda: data_loader.load_cross_attention_heatmaps(run_dir),
        "saliency": lambda: data_loader.load_gradient_data(run_dir, "saliency"),
        "gradcam_siglip": lambda: data_loader.load_gradient_data(run_dir, "gradcam_siglip"),
        "gradcam_connector": lambda: data_loader.load_gradient_data(run_dir, "gradcam_connector"),
    }
    if viz_type in heatmap_types:
        hm = heatmap_types[viz_type]()
        return stats_engine.compute_heatmap_stats(hm) if hm else None

    if viz_type == "per_head":
        ph = data_loader.load_per_head_data(run_dir)
        return stats_engine.compute_per_head_stats(
            ph.get("heads", []), ph.get("entropies")) if ph else None

    if viz_type == "vision_vs_state":
        vs = data_loader.load_vision_vs_state(run_dir)
        return stats_engine.compute_vision_vs_state_stats(vs) if vs else None

    if viz_type == "per_action_dim":
        dims = data_loader.load_per_action_dim(run_dir)
        return stats_engine.compute_per_action_dim_stats(dims) if dims else None

    return None


def _build_run_insights_prompt(run_dir: Path, manifest: dict,
                               computed_stats_text: str) -> str:
    """Build the run_insights prompt with all viz stats."""
    model_info = manifest.get("model_info", {})
    dataset_info = manifest.get("dataset_info", {})
    template = prompt_templates.get_template("run_insights")
    return prompt_templates.format_template(
        template,
        model_id=model_info.get("model_id", "unknown"),
        dataset_id=dataset_info.get("dataset_id", "unknown"),
        task_string=dataset_info.get("task_string", ""),
        episode_idx=dataset_info.get("episode_idx", 0),
        num_frames=dataset_info.get("num_frames", 0),
        computed_stats=computed_stats_text,
    )


def _build_multi_run_prompt(req: LLMAnalyzeRequest, run_dir: Path,
                            manifest: dict, settings: Settings,
                            computed_stats_text: str) -> str:
    """Build the multi_run_comparison prompt."""
    run_dirs = [run_dir]
    manifests = [manifest]
    for rid in (req.compare_run_ids or []):
        if rid == req.run_id:
            continue
        try:
            rd, mf = run_scanner.load_manifest(settings.base_dir, rid)
            run_dirs.append(rd)
            manifests.append(mf)
        except FileNotFoundError:
            continue

    # Run descriptions
    descriptions = []
    for i, (rd, mf) in enumerate(zip(run_dirs, manifests)):
        mi = mf.get("model_info", {})
        di = mf.get("dataset_info", {})
        descriptions.append(
            f"Run {i + 1} ({rd.name}): model={mi.get('model_id', '?')}, "
            f"dataset={di.get('dataset_id', '?')}, task=\"{di.get('task_string', '?')}\""
        )
    run_descriptions = "\n".join(descriptions)

    # Config diff
    config_diff = comparison.compute_config_diff(manifests)
    config_diff_text = ""
    if config_diff:
        lines = []
        for key, values in config_diff.items():
            val_strs = [f"Run {i + 1}: {v}" for i, v in enumerate(values)]
            lines.append(f"- {key}: {', '.join(val_strs)}")
        config_diff_text = "\n".join(lines)
    else:
        config_diff_text = "(No configuration differences detected.)"

    # Run notes
    notes_lines = []
    if req.run_notes:
        for rid, note in req.run_notes.items():
            if note.strip():
                # Find run name
                name = rid
                for rd, mf in zip(run_dirs, manifests):
                    if mf.get("_id", rd.name) == rid or rd.name == rid:
                        name = rd.name
                        break
                notes_lines.append(f"- {name}: {note}")
    run_notes_text = "\n".join(notes_lines) if notes_lines else "(No researcher notes provided.)"

    template = prompt_templates.get_template("multi_run_comparison")
    return prompt_templates.format_template(
        template,
        run_descriptions=run_descriptions,
        config_diff=config_diff_text,
        computed_stats=computed_stats_text,
        run_notes=run_notes_text,
    )


def _collect_images(req: LLMAnalyzeRequest, run_dir: Path,
                    manifest: dict, settings: Settings) -> list[Path]:
    """Collect relevant images for the LLM analysis."""
    image_paths: list[Path] = []
    if not req.include_images:
        return image_paths

    if req.analysis_type == "run_insights":
        # Collect 1-2 representative images per viz type (cap at 10)
        images = data_loader.list_images(run_dir)
        seen_types: dict[str, int] = {}
        for img in images:
            for vt_key, prefixes in _VIZ_IMAGE_MAPPING.items():
                name = img.split("/")[-1].lower()
                if any(name.startswith(p) for p in prefixes):
                    count = seen_types.get(vt_key, 0)
                    if count < 2:
                        resolved = data_loader.resolve_image_path(run_dir, img)
                        if resolved:
                            image_paths.append(resolved)
                            seen_types[vt_key] = count + 1
                    break
            if len(image_paths) >= 10:
                break
        return image_paths

    if req.analysis_type == "multi_run_comparison":
        # Collect a few images from each run
        all_run_dirs = [run_dir]
        for rid in (req.compare_run_ids or []):
            if rid == req.run_id:
                continue
            try:
                rd, _ = run_scanner.load_manifest(settings.base_dir, rid)
                all_run_dirs.append(rd)
            except FileNotFoundError:
                continue
        for rd in all_run_dirs:
            images = data_loader.list_images(rd)
            count = 0
            for img in images:
                if count >= 3:
                    break
                resolved = data_loader.resolve_image_path(rd, img)
                if resolved:
                    image_paths.append(resolved)
                    count += 1
            if len(image_paths) >= 10:
                break
        return image_paths

    # Standard per-viz image collection
    images = data_loader.list_images(run_dir)
    viz_type = req.viz_type or req.analysis_type.replace("single_viz_", "")
    for img in images:
        if _image_relevant(img, viz_type):
            resolved = data_loader.resolve_image_path(run_dir, img)
            if resolved:
                image_paths.append(resolved)
    return image_paths[:5]


_VIZ_IMAGE_MAPPING = {
    "self_attention": ["episode_dashboard", "attention_grid"],
    "per_head": ["per_head"],
    "cross_attention": ["per_step_cross_attn"],
    "per_step_cross_attention": ["per_step_cross_attn"],
    "gradcam_siglip": ["episode_dashboard"],
    "gradcam_connector": ["episode_dashboard"],
    "saliency": ["episode_dashboard"]
    "gradcam_vlm_layers": ["vlm_layers"],
    "per_action_dim": ["per_action_dim"],
    "language_diff": ["language_diff"],
    "vision_vs_state": ["vision_vs_state"],
}


def _image_relevant(image_path: str, viz_type: str) -> bool:
    name = image_path.split("/")[-1].lower()
    mapping = {
        **_VIZ_IMAGE_MAPPING,
        "model_health": ["model_health"],
        "health": ["model_health"],
    }
    prefixes = mapping.get(viz_type, [])
    return any(name.startswith(p) for p in prefixes)

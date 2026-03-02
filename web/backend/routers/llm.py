"""LLM streaming analysis endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..config import Settings
from ..models.schemas import LLMAnalyzeRequest, LLMConfig
from ..services import data_loader, llm_service, prompt_templates, run_scanner

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

    # Build prompt
    if req.prompt:
        prompt = req.prompt
    else:
        template = prompt_templates.get_template(req.analysis_type)
        model_info = manifest.get("model_info", {})
        dataset_info = manifest.get("dataset_info", {})
        prompt = prompt_templates.format_template(
            template,
            model_id=model_info.get("model_id", "unknown"),
            dataset_id=dataset_info.get("dataset_id", "unknown"),
            task_string=dataset_info.get("task_string", ""),
            episode_idx=dataset_info.get("episode_idx", 0),
            num_frames=dataset_info.get("num_frames", 0),
            method=manifest.get("cli_args", {}).get("method", "rollout"),
            action_dim_names=str(dataset_info.get("action_dim_names", [])),
        )

    # Collect relevant images
    image_paths: list[Path] = []
    if req.include_images:
        images = data_loader.list_images(run_dir)
        viz_type = req.viz_type or req.analysis_type.replace("single_viz_", "")
        for img in images:
            if _image_relevant(img, viz_type):
                resolved = data_loader.resolve_image_path(run_dir, img)
                if resolved:
                    image_paths.append(resolved)
        # Limit to 5 images to stay within token limits
        image_paths = image_paths[:5]

    async def event_generator():
        async for token in llm_service.stream_analysis(prompt, image_paths, settings):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _image_relevant(image_path: str, viz_type: str) -> bool:
    name = image_path.split("/")[-1].lower()
    mapping = {
        "self_attention": ["episode_dashboard", "attention_grid"],
        "per_head": ["per_head"],
        "cross_attention": ["episode_dashboard"],
        "per_step_cross_attention": ["per_step_cross_attn"],
        "gradcam_siglip": ["episode_dashboard"],
        "gradcam_connector": ["episode_dashboard"],
        "gradcam_vlm_layers": ["vlm_layers"],
        "per_action_dim": ["per_action_dim"],
        "language_diff": ["language_diff"],
        "vision_vs_state": ["vision_vs_state"],
        "model_health": ["model_health"],
        "health": ["model_health"],
    }
    prefixes = mapping.get(viz_type, [])
    return any(name.startswith(p) for p in prefixes)

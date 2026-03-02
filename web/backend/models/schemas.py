"""Pydantic models for API responses."""

from __future__ import annotations

from pydantic import BaseModel


class RunSummary(BaseModel):
    id: str
    name: str
    created_at: str | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    episode_idx: int | None = None
    num_frames: int | None = None
    available_visualizations: dict[str, bool] = {}
    is_legacy: bool = False


class RunDetail(BaseModel):
    id: str
    name: str
    manifest: dict


class FrameInfo(BaseModel):
    index: int
    image_url: str


class VizData(BaseModel):
    """Generic visualization data payload."""
    viz_type: str
    frames: list[int] = []
    # For patch-resolution data: JSON float arrays
    heatmaps: list[list[list[float]]] | None = None
    # For pixel-resolution data: image URLs
    image_urls: list[str] | None = None
    # For chart data (vision_vs_state, entropy, etc.)
    chart_data: list[dict] | None = None
    # Additional metadata
    metadata: dict = {}


class HealthData(BaseModel):
    weightwatcher: dict | None = None
    entropy: dict | None = None
    redundancy: dict | None = None


class CompareRequest(BaseModel):
    run_ids: list[str]
    viz_types: list[str] = []
    frame_indices: list[int] | None = None


class CompareResult(BaseModel):
    runs: list[dict]
    diffs: dict = {}


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""


class VizStats(BaseModel):
    viz_type: str
    per_frame: list[dict] = []
    aggregate: dict = {}


class LLMAnalyzeRequest(BaseModel):
    run_id: str
    analysis_type: str  # e.g. "self_attention", "health", "comparison"
    prompt: str | None = None  # user-edited prompt overrides default
    viz_type: str | None = None
    frame_indices: list[int] | None = None
    compare_run_ids: list[str] | None = None
    include_images: bool = True
    include_stats: bool = True
    run_notes: dict[str, str] | None = None


# -- LLM analysis cache (persisted to llm_analyses.json per run) --

class SaveLLMAnalysisRequest(BaseModel):
    analysis_type: str
    response: str
    model: str | None = None
    provider: str | None = None
    custom_prompt: bool = False


class LLMAnalysisEntry(BaseModel):
    response: str
    model: str = ""
    provider: str = ""
    custom_prompt: bool = False
    created_at: str = ""
    updated_at: str = ""


class LLMAnalysesResponse(BaseModel):
    analyses: dict[str, LLMAnalysisEntry] = {}

"""Application settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_dir: Path = Field(
        default=Path("./outputs"),
        description="Root directory to scan for run folders",
    )
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = ["http://localhost:5173"]
    cors_origin_regex: str = r"http://localhost:\d+"

    # LLM defaults
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: str = ""
    llm_base_url: str = ""  # custom base URL for OpenAI-compatible servers

    # Static files (built frontend)
    static_dir: Path | None = None

    model_config = {
        "env_prefix": "SMOLVLA_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

"""Agentic diagnostic system for smolvla-inspect."""

from __future__ import annotations

from .agent import DiagnosticAgent
from .models import DiagnosticReport
from .report import save_diagnostic_report, load_diagnostic_report

# Import modules with @register_primitive decorators so they register on load.
from . import scene as _scene  # noqa: F401
from . import counterfactual as _counterfactual  # noqa: F401
from . import temporal as _temporal  # noqa: F401
from . import occlusion as _occlusion  # noqa: F401

__all__ = [
    "DiagnosticAgent",
    "DiagnosticReport",
    "run_diagnostic",
    "save_diagnostic_report",
    "load_diagnostic_report",
]


async def run_diagnostic(
    policy=None,
    dataset=None,
    episode_idx: int = 0,
    image_key: str = "observation.images.top",
    device: str = "cpu",
    llm_settings: dict | None = None,
    config: dict | None = None,
    existing_run_dir: str | None = None,
    output_dir: str | None = None,
    image_map=None,
    progress_callback=None,
) -> DiagnosticReport:
    """Run the full diagnostic pipeline.

    Args:
        policy: SmolVLA policy (required for integrated mode + counterfactuals)
        dataset: LeRobot dataset
        episode_idx: Episode to analyze
        image_key: Dataset image key to use
        device: Device for computation
        llm_settings: LLM provider config dict
        config: Diagnostic config dict
        existing_run_dir: Path to completed run (post-hoc mode)
        output_dir: Where to save the report
        image_map: Optional image key mapping
        progress_callback: Callable(phase, detail) for progress updates

    Returns:
        DiagnosticReport with all findings
    """
    import os

    if llm_settings is None:
        llm_settings = {
            "provider": os.environ.get("SMOLVLA_LLM_PROVIDER", "anthropic"),
            "model": os.environ.get("SMOLVLA_LLM_MODEL", "claude-sonnet-4-20250514"),
            "api_key": os.environ.get("SMOLVLA_LLM_API_KEY", ""),
            "base_url": os.environ.get("SMOLVLA_LLM_BASE_URL", ""),
        }
        # Fallback to standard env vars
        if not llm_settings["api_key"]:
            if llm_settings["provider"] == "anthropic":
                llm_settings["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
            else:
                llm_settings["api_key"] = os.environ.get("OPENAI_API_KEY", "")

    if config is None:
        config = {}

    agent = DiagnosticAgent(
        policy=policy,
        dataset=dataset,
        episode_idx=episode_idx,
        image_key=image_key,
        device=device,
        llm_settings=llm_settings,
        config=config,
        existing_run_dir=existing_run_dir,
        image_map=image_map,
    )

    report = await agent.run(progress_callback=progress_callback)

    # Save report if output_dir provided
    if output_dir:
        run_dir = existing_run_dir or output_dir
        save_diagnostic_report(report, run_dir)

    return report

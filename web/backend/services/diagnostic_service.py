"""Backend service for loading and triggering diagnostic runs."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from ..config import Settings


def get_run_dir(run_id: str, settings: Settings) -> Path:
    """Resolve a run ID to its directory path."""
    # run_id is a path-safe identifier, typically the folder name
    base = Path(settings.base_dir)
    # Search for the run directory
    for parent in [base] + list(base.iterdir()) if base.is_dir() else [base]:
        if not parent.is_dir():
            continue
        candidate = parent / run_id
        if candidate.is_dir():
            return candidate
        # Also check subdirectories (user/run_name pattern)
        for child in parent.iterdir():
            if child.is_dir() and child.name == run_id:
                return child
    # Fallback: treat run_id as relative to base
    return base / run_id


def load_diagnostic_report(run_dir: Path) -> dict | None:
    """Load a saved diagnostic report from a run directory."""
    report_path = run_dir / "diagnostic" / "report.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        return json.load(f)


def load_diagnostic_matrix(run_dir: Path) -> dict | None:
    """Load the diagnostic matrix from a run directory."""
    matrix_path = run_dir / "diagnostic" / "matrix.json"
    if not matrix_path.exists():
        return None
    with open(matrix_path) as f:
        return json.load(f)


def load_scene_data(run_dir: Path) -> dict | None:
    """Load scene understanding data."""
    det_path = run_dir / "diagnostic" / "scene" / "detections.json"
    if not det_path.exists():
        return None
    with open(det_path) as f:
        data = json.load(f)
    # Check for annotated frame
    annotated = run_dir / "diagnostic" / "scene" / "annotated_frame.png"
    data["has_annotated_frame"] = annotated.exists()
    return data


def load_counterfactual_result(run_dir: Path, test_name: str) -> dict | None:
    """Load a specific counterfactual result."""
    result_path = run_dir / "diagnostic" / "counterfactuals" / test_name / "result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        data = json.load(f)
    comp = run_dir / "diagnostic" / "counterfactuals" / test_name / "comparison.png"
    data["has_comparison_image"] = comp.exists()
    return data


def diagnostic_exists(run_dir: Path) -> bool:
    """Check if a diagnostic report exists for this run."""
    return (run_dir / "diagnostic" / "report.json").exists()


async def run_diagnostic_async(
    run_dir: Path,
    dataset_id: str | None = None,
    model_id: str | None = None,
    max_counterfactuals: int = 3,
    skip_counterfactuals: bool = False,
    settings: Settings | None = None,
):
    """Run a diagnostic analysis on an existing run. Returns (report_dict, error)."""
    try:
        from smolvla_inspect.diagnostic import run_diagnostic

        # Load manifest for context
        manifest_path = run_dir / "run_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

        # Determine dataset
        ds_id = dataset_id or manifest.get("dataset_info", {}).get("dataset_id")
        if not ds_id:
            return None, "Dataset ID required"

        # Load dataset
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(ds_id)

        # Determine image key
        ds_info = manifest.get("dataset_info", {})
        image_keys = ds_info.get("image_keys", [])
        image_key = image_keys[0] if image_keys else None
        if image_key is None:
            from smolvla_inspect.data import find_image_keys
            found = find_image_keys(dataset)
            image_key = found[0] if found else "observation.images.top"

        episode_idx = ds_info.get("episode_idx", 0)

        # Optionally load model
        policy = None
        mdl_id = model_id or manifest.get("model_info", {}).get("model_id")
        if mdl_id and not skip_counterfactuals:
            try:
                import torch
                from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                device = "cuda" if torch.cuda.is_available() else "cpu"
                policy = SmolVLAPolicy.from_pretrained(mdl_id)
                policy.to(device)
                policy.eval()
            except Exception as e:
                print(f"  Warning: Could not load model for counterfactuals: {e}")

        device = "cuda" if policy and next(policy.parameters()).is_cuda else "cpu"

        # LLM settings
        llm_settings = {
            "provider": os.environ.get("SMOLVLA_LLM_PROVIDER",
                                       settings.llm_provider if settings else "anthropic"),
            "model": os.environ.get("SMOLVLA_LLM_MODEL",
                                    settings.llm_model if settings else "claude-sonnet-4-20250514"),
            "api_key": os.environ.get("SMOLVLA_LLM_API_KEY",
                                      settings.llm_api_key if settings else ""),
            "base_url": os.environ.get("SMOLVLA_LLM_BASE_URL",
                                       settings.llm_base_url if settings else ""),
        }
        if not llm_settings["api_key"]:
            if llm_settings["provider"] == "anthropic":
                llm_settings["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
            else:
                llm_settings["api_key"] = os.environ.get("OPENAI_API_KEY", "")

        config = {
            "max_counterfactuals": 0 if skip_counterfactuals else max_counterfactuals,
            "max_hypotheses": 5,
        }

        report = await run_diagnostic(
            policy=policy,
            dataset=dataset,
            episode_idx=episode_idx,
            image_key=image_key,
            device=device,
            llm_settings=llm_settings,
            config=config,
            existing_run_dir=str(run_dir),
            output_dir=str(run_dir),
        )

        # Reload the saved report JSON
        return load_diagnostic_report(run_dir), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)

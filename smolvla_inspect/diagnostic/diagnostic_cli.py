"""CLI subcommand: diagnose — run the diagnostic agent."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import yaml


def add_diagnose_args(parser: argparse.ArgumentParser):
    """Add diagnose subcommand arguments to an existing parser."""
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to existing run directory (post-hoc mode)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (HuggingFace repo or local path)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset ID (HuggingFace repo or local path)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to analyze")
    parser.add_argument("--image-key", type=str, default=None,
                        help="Dataset image key to use")
    parser.add_argument("--image-map", type=str, default=None,
                        help="Image key mapping (e.g. 'front=camera2,side=camera3')")
    parser.add_argument("--config", type=str, default=None,
                        help="Diagnostic config YAML path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda, mps, cpu)")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for the report")
    parser.add_argument("--max-counterfactuals", type=int, default=3,
                        help="Maximum counterfactual tests to run")
    parser.add_argument("--skip-counterfactuals", action="store_true",
                        help="Skip counterfactual testing")
    parser.add_argument("--max-hypotheses", type=int, default=5,
                        help="Maximum hypotheses to generate")


def diagnose_main(args: argparse.Namespace):
    """Main entry point for the diagnose subcommand."""
    print("\n" + "=" * 60)
    print("  SmolVLA Diagnostic Agent")
    print("=" * 60)

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # Override config with CLI args
    config["max_counterfactuals"] = args.max_counterfactuals
    config["max_hypotheses"] = args.max_hypotheses
    if args.skip_counterfactuals:
        config["max_counterfactuals"] = 0

    # Auto-detect device
    device = args.device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"\n  Device: {device}")

    # Determine mode
    policy = None
    dataset = None
    image_key = args.image_key
    existing_run_dir = args.run_dir

    if args.run_dir:
        # Post-hoc mode
        print(f"  Mode: Post-hoc analysis of {args.run_dir}")
        if not os.path.isdir(args.run_dir):
            print(f"  ERROR: Run directory not found: {args.run_dir}")
            sys.exit(1)

        # Load dataset (needed for scene detection + diversity)
        if args.dataset:
            print(f"  Loading dataset: {args.dataset}")
            dataset, image_key = _load_dataset(args.dataset, args.episode,
                                                args.image_key)
        else:
            # Try to get dataset info from manifest
            manifest_path = os.path.join(args.run_dir, "run_manifest.json")
            if os.path.exists(manifest_path):
                import json
                with open(manifest_path) as f:
                    manifest = json.load(f)
                ds_info = manifest.get("dataset_info", {})
                ds_id = ds_info.get("dataset_id")
                if ds_id:
                    print(f"  Loading dataset from manifest: {ds_id}")
                    dataset, image_key = _load_dataset(
                        ds_id, args.episode, args.image_key or ds_info.get("image_keys", [None])[0])
            if dataset is None:
                print("  ERROR: --dataset is required (could not infer from manifest)")
                sys.exit(1)

        # Optionally load model for counterfactuals
        if args.model and not args.skip_counterfactuals:
            print(f"  Loading model: {args.model}")
            policy = _load_policy(args.model, device)
    else:
        # Integrated mode
        if not args.model:
            print("  ERROR: --model is required for integrated mode (or use --run-dir for post-hoc)")
            sys.exit(1)
        if not args.dataset:
            print("  ERROR: --dataset is required")
            sys.exit(1)

        print(f"  Mode: Integrated (inspect + diagnose)")
        print(f"  Loading model: {args.model}")
        policy = _load_policy(args.model, device)

        print(f"  Loading dataset: {args.dataset}")
        dataset, image_key = _load_dataset(args.dataset, args.episode, args.image_key)

    if image_key is None:
        from ..data import find_image_keys
        image_keys = find_image_keys(dataset)
        image_key = image_keys[0] if image_keys else "observation.images.top"

    print(f"  Image key: {image_key}")
    print(f"  Episode: {args.episode}")

    # Parse image map
    image_map = None
    if args.image_map:
        from ..data import parse_image_map
        image_map = parse_image_map(args.image_map)

    # LLM settings from env
    llm_settings = {
        "provider": os.environ.get("SMOLVLA_LLM_PROVIDER", "anthropic"),
        "model": os.environ.get("SMOLVLA_LLM_MODEL", "claude-sonnet-4-20250514"),
        "api_key": os.environ.get("SMOLVLA_LLM_API_KEY", ""),
        "base_url": os.environ.get("SMOLVLA_LLM_BASE_URL", ""),
    }
    if not llm_settings["api_key"]:
        if llm_settings["provider"] == "anthropic":
            llm_settings["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            llm_settings["api_key"] = os.environ.get("OPENAI_API_KEY", "")

    if llm_settings["api_key"]:
        print(f"  LLM: {llm_settings['provider']} / {llm_settings['model']}")
    else:
        print("  LLM: Not configured (will use rule-based fallback)")

    print("\n" + "-" * 60)

    # Run diagnostic
    from . import run_diagnostic

    output_dir = args.output_dir
    if existing_run_dir:
        output_dir = existing_run_dir

    report = asyncio.run(run_diagnostic(
        policy=policy,
        dataset=dataset,
        episode_idx=args.episode,
        image_key=image_key,
        device=device,
        llm_settings=llm_settings,
        config=config,
        existing_run_dir=existing_run_dir,
        output_dir=output_dir,
        image_map=image_map,
    ))

    # Print summary
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC RESULTS")
    print("=" * 60)

    critical = [f for f in report.findings if f.severity == "critical"]
    warnings = [f for f in report.findings if f.severity == "warning"]
    info = [f for f in report.findings if f.severity == "info"]

    if critical:
        print(f"\n  CRITICAL ({len(critical)}):")
        for f in critical:
            print(f"    !! {f.title}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for f in warnings:
            print(f"    !  {f.title}")
    if info:
        print(f"\n  INFO ({len(info)}):")
        for f in info:
            print(f"       {f.title}")

    if not report.findings:
        print("\n  No issues detected.")

    print(f"\n  Full report: {output_dir}/diagnostic/report.md")
    print("=" * 60 + "\n")


def _load_policy(model_id: str, device: str):
    """Load a SmolVLA policy."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()
    return policy


def _load_dataset(dataset_id: str, episode_idx: int, image_key: str | None):
    """Load a LeRobot dataset and determine image key."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(dataset_id)

    if image_key is None:
        from ..data import find_image_keys
        image_keys = find_image_keys(dataset)
        image_key = image_keys[0] if image_keys else None

    return dataset, image_key

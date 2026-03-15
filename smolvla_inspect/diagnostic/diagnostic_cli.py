"""CLI subcommand: diagnose — run the diagnostic agent."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import yaml


def add_diagnose_args(parser: argparse.ArgumentParser, defaults=None):
    """Add diagnose subcommand arguments to an existing parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    defaults : dict | None
        Loaded YAML defaults (from ``load_defaults``).  When provided the
        full inspection-pipeline flags are also registered so that
        integrated mode (``--model`` + ``--dataset``) runs the complete
        pipeline before diagnosing.
    """
    if defaults is None:
        defaults = {}

    # Diagnose-specific args
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to existing run directory (post-hoc mode)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (HuggingFace repo or local path)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset ID (HuggingFace repo or local path)")
    parser.add_argument("--episode", type=int,
                        default=defaults.get("episode", 0),
                        help="Episode index to analyze")
    parser.add_argument("--config", type=str, default=None,
                        help="Diagnostic config YAML path")
    parser.add_argument("--max-counterfactuals", type=int, default=7,
                        help="Maximum counterfactual tests to run")
    parser.add_argument("--skip-counterfactuals", action="store_true",
                        help="Skip counterfactual testing")
    parser.add_argument("--max-hypotheses", type=int, default=7,
                        help="Maximum hypotheses to generate")

    # Pipeline flags (shared with the main CLI).  The exclude set avoids
    # clashes with args that diagnose already registers above.
    from ..cli import add_pipeline_args
    add_pipeline_args(parser, defaults, exclude={
        "--episode",
    })


def diagnose_main(args: argparse.Namespace):
    """Main entry point for the diagnose subcommand."""
    import torch

    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("  SmolVLA Diagnostic Agent")
    print("=" * 60)

    # Load diagnostic config
    diag_config = {}
    diag_config_path = args.config
    if diag_config_path is None:
        # Try the default diagnostic config
        _default_diag = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "configs", "diagnostic.yaml")
        if os.path.exists(_default_diag):
            diag_config_path = _default_diag
    if diag_config_path and os.path.exists(diag_config_path):
        with open(diag_config_path) as f:
            diag_config = yaml.safe_load(f) or {}

    # Override config with CLI args
    diag_config["max_counterfactuals"] = args.max_counterfactuals
    diag_config["max_hypotheses"] = args.max_hypotheses
    if args.skip_counterfactuals:
        diag_config["max_counterfactuals"] = 0

    # Auto-detect device
    device = args.device
    if device is None or device == "auto":
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
        # ── Post-hoc mode ──────────────────────────────────────
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
        # ── Integrated mode: full pipeline + diagnose ──────────
        if not args.model:
            print("  ERROR: --model is required for integrated mode (or use --run-dir for post-hoc)")
            sys.exit(1)
        if not args.dataset:
            print("  ERROR: --dataset is required")
            sys.exit(1)

        print(f"  Mode: Integrated (full pipeline + diagnose)")

        # --- Run the full inspection pipeline first ---
        from ..cli import run_inspection_pipeline, load_defaults
        from ..export import create_run_dir
        from ..data import parse_image_map

        # Force export on — the diagnostic agent reads the exported data
        args.export_data = True

        # Set up run directory
        base_output_dir = args.output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        run_dir = create_run_dir(base_output_dir, getattr(args, "run_name", None))
        images_dir = os.path.join(run_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        args.output_dir = images_dir  # PNGs go into images/ under the run folder

        # Resolve devices
        torch_device = torch.device(device)
        grad_device_str = getattr(args, "gradient_device", None) or device
        grad_device = torch.device(grad_device_str)

        # Ensure args.device is a string for the pipeline
        args.device = device

        # Auto-enable cross-attention if per-step is requested
        if getattr(args, "per_step_cross_attention", False) and not getattr(args, "cross_attention", False):
            args.cross_attention = True
        # Auto-enable gradient if language-diff is requested
        if getattr(args, "language_diff", None) and not getattr(args, "gradient", None):
            args.gradient = "gradcam"

        # Load model
        print(f"\n  Loading model: {args.model}")
        policy = _load_policy(args.model, device)

        # Load dataset
        print(f"  Loading dataset: {args.dataset}")
        dataset, image_key = _load_dataset(args.dataset, args.episode, args.image_key)

        # Parse image map
        image_map = parse_image_map(getattr(args, "image_map", None))

        print(f"\n  Running full inspection pipeline...")
        run_inspection_pipeline(args, policy, dataset, torch_device, grad_device, run_dir, image_map)

        # Now switch to post-hoc mode for the diagnostic agent
        existing_run_dir = run_dir
        # Restore output_dir for the diagnostic report path
        args.output_dir = base_output_dir

    if image_key is None:
        from ..data import find_image_keys
        image_keys = find_image_keys(dataset)
        image_key = image_keys[0] if image_keys else "observation.images.top"

    print(f"  Image key: {image_key}")
    print(f"  Episode: {args.episode}")

    # Parse image map (for post-hoc mode; integrated mode already parsed above)
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

    # Config summary
    print(f"\n  Config:")
    print(f"    Max hypotheses:      {diag_config.get('max_hypotheses', 5)}")
    print(f"    Max counterfactuals:  {diag_config.get('max_counterfactuals', 7)}")
    print(f"    Max expensive signals:{diag_config.get('max_expensive_signals', 4)}")
    print(f"    Num frames:          {diag_config.get('num_frames', 4)}")
    if diag_config.get("max_counterfactuals", 7) == 0:
        print(f"    ** Counterfactuals:  SKIPPED **")

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
        config=diag_config,
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


def add_compare_args(parser: argparse.ArgumentParser):
    """Add compare subcommand arguments to an existing parser."""
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Two or more run directories to compare")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Human-readable labels for each run (same order as --runs)")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for the comparison report")


def compare_main(args: argparse.Namespace):
    """Main entry point for the compare subcommand."""
    print("\n" + "=" * 60)
    print("  SmolVLA Run Comparison")
    print("=" * 60)

    run_dirs = args.runs
    labels = args.labels

    if len(run_dirs) < 2:
        print("  ERROR: Need at least 2 run directories to compare.")
        sys.exit(1)

    if labels and len(labels) != len(run_dirs):
        print(f"  ERROR: Got {len(labels)} labels but {len(run_dirs)} run directories.")
        sys.exit(1)

    # Validate run directories
    for rd in run_dirs:
        report_path = os.path.join(rd, "diagnostic", "report.json")
        if not os.path.exists(report_path):
            print(f"  ERROR: No diagnostic report found in {rd}")
            print(f"         Expected: {report_path}")
            print(f"         Run `diagnose` on this directory first.")
            sys.exit(1)

    print(f"\n  Comparing {len(run_dirs)} runs:")
    for i, rd in enumerate(run_dirs):
        lbl = labels[i] if labels else os.path.basename(os.path.normpath(rd))
        print(f"    {i + 1}. [{lbl}] {rd}")

    from .comparison import compare_runs

    report = compare_runs(run_dirs, labels)

    # Save
    output_dir = args.output_dir
    json_path, md_path = report.save(output_dir)

    # Print summary
    print("\n" + "-" * 60)
    print("  COMPARISON RESULTS")
    print("-" * 60)

    if report.weight_alpha_deltas:
        print("\n  Weight Spectral (alpha):")
        for wa in report.weight_alpha_deltas:
            arrow = "improved" if wa.delta < 0 and wa.values[0] > 4 else (
                "worsened" if wa.delta > 0.01 else "unchanged")
            pct = f" ({wa.pct_change:+.1f}%)" if wa.pct_change is not None else ""
            print(f"    {wa.component}: {wa.values[0]:.2f} → {wa.values[-1]:.2f} [{arrow}{pct}]")

    if report.counterfactual_deltas:
        print("\n  Counterfactuals:")
        for cd in report.counterfactual_deltas:
            v0 = f"{cd.values[0]:.4f}" if cd.values[0] is not None else "N/A"
            v1 = f"{cd.values[-1]:.4f}" if cd.values[-1] is not None else "N/A"
            if cd.delta is not None:
                sign = "+" if cd.delta > 0 else ""
                pct = f" ({cd.pct_change:+.1f}%)" if cd.pct_change is not None else ""
                print(f"    {cd.test_type}: {v0} → {v1} [{sign}{cd.delta:.4f}{pct}]")
            else:
                print(f"    {cd.test_type}: {v0} → {v1} [N/A — not run in both]")

    if report.symptom_summary:
        first_label = report.labels[0]
        last_label = report.labels[-1]
        first_set = set(report.symptom_summary.get(first_label, []))
        last_set = set(report.symptom_summary.get(last_label, []))
        resolved = first_set - last_set
        new_symp = last_set - first_set
        persistent = first_set & last_set
        if resolved:
            print(f"\n  Resolved symptoms: {', '.join(resolved)}")
        if new_symp:
            print(f"\n  New symptoms: {', '.join(new_symp)}")
        if persistent:
            print(f"\n  Persistent symptoms: {', '.join(persistent)}")

    if report.verdict:
        print(f"\n  Verdict:")
        for line in report.verdict.split("\n"):
            print(f"    {line}")

    if report.recommendations:
        print(f"\n  Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            # Strip markdown bold for terminal display
            clean = rec.replace("**", "")
            print(f"    {i}. {clean}")

    print(f"\n  Full report: {md_path}")
    print("=" * 60 + "\n")


def _load_policy(model_id: str, device: str):
    """Load a SmolVLA policy."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()
    return policy


def _load_dataset(dataset_id: str, episode_idx: int, image_key: str | None):
    """Load a LeRobot dataset and determine image key."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(dataset_id)

    if image_key is None:
        from ..data import find_image_keys
        image_keys = find_image_keys(dataset)
        image_key = image_keys[0] if image_keys else None

    return dataset, image_key

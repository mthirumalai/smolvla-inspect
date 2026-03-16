"""SmolVLA Inspector — package entry point."""

import sys


def main():
    """Dispatch to CLI, serve, diagnose, or compare subcommand."""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from .serve import serve_main
        serve_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        # Remove 'diagnose' from argv so argparse sees only the flags
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .cli import load_defaults
        from .diagnostic.diagnostic_cli import add_diagnose_args, diagnose_main
        import argparse

        # Pre-parse --config to load YAML defaults for pipeline flags
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("--config", type=str, default=None)
        pre_args, _ = pre_parser.parse_known_args()
        defaults = load_defaults(pre_args.config)

        parser = argparse.ArgumentParser(
            description="SmolVLA Diagnostic Agent — automated model diagnosis")
        add_diagnose_args(parser, defaults=defaults)
        args = parser.parse_args()
        diagnose_main(args)
    elif len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from .mcp import mcp_main
        mcp_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .diagnostic.diagnostic_cli import add_compare_args, compare_main
        import argparse
        parser = argparse.ArgumentParser(
            description="SmolVLA Run Comparison — compare diagnostic reports across runs")
        add_compare_args(parser)
        args = parser.parse_args()
        compare_main(args)
    else:
        from .cli import main as cli_main
        cli_main()


__all__ = ["main"]

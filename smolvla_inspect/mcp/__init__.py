"""MCP server for smolvla-inspect — ``smolvla-inspect mcp``."""

from __future__ import annotations

import argparse
import sys


def mcp_main(argv: list[str] | None = None):
    """Entry point for ``smolvla-inspect mcp``."""
    parser = argparse.ArgumentParser(
        description="Launch the smolvla-inspect MCP server (stdio transport).",
        prog="smolvla-inspect mcp",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./outputs",
        help="Root directory to scan for run folders (default: ./outputs)",
    )
    args = parser.parse_args(argv)

    from .context import init_context
    init_context(args.base_dir)

    from .server import run_server
    run_server()

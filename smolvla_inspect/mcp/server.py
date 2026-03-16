"""FastMCP server instance and startup."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("smolvla-inspect")


def run_server():
    """Import tool/resource modules (triggering registration) and start."""
    # Importing the tools package registers all @mcp.tool() decorators
    from . import tools  # noqa: F401
    from . import resources  # noqa: F401

    mcp.run(transport="stdio")

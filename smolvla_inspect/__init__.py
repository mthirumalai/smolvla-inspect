"""SmolVLA Inspector — package entry point."""

import sys


def main():
    """Dispatch to CLI or serve subcommand."""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from .serve import serve_main
        serve_main(sys.argv[2:])
    else:
        from .cli import main as cli_main
        cli_main()


__all__ = ["main"]

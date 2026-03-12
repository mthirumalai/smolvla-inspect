"""SmolVLA Inspector — package entry point."""

import sys


def main():
    """Dispatch to CLI, serve, or diagnose subcommand."""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from .serve import serve_main
        serve_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        # Remove 'diagnose' from argv so argparse sees only the flags
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .diagnostic.diagnostic_cli import add_diagnose_args, diagnose_main
        import argparse
        parser = argparse.ArgumentParser(
            description="SmolVLA Diagnostic Agent — automated model diagnosis")
        add_diagnose_args(parser)
        args = parser.parse_args()
        diagnose_main(args)
    else:
        from .cli import main as cli_main
        cli_main()


__all__ = ["main"]

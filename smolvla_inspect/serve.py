"""Launch the web viewer — `smolvla-inspect serve`."""

import argparse
import sys
from pathlib import Path


def serve_main(argv: list[str] | None = None):
    """Entry point for `smolvla-inspect serve`."""
    parser = argparse.ArgumentParser(
        description="Launch the smolvla-inspect web viewer.",
        prog="smolvla-inspect serve",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./outputs",
        help="Root directory to scan for run folders (default: ./outputs)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open browser automatically",
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run:")
        print('  pip install "uvicorn[standard]"')
        sys.exit(1)

    try:
        from web.backend.config import Settings
        from web.backend.main import create_app, set_settings
    except ImportError:
        # Try relative import (if running from project root)
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from web.backend.config import Settings
            from web.backend.main import create_app, set_settings
        except ImportError:
            print("ERROR: Web backend not found. Make sure you're running from the project root.")
            sys.exit(1)

    # Check for built frontend
    project_root = Path(__file__).resolve().parent.parent
    frontend_dist = project_root / "web" / "frontend" / "dist"
    static_dir = frontend_dist if frontend_dist.is_dir() else None

    settings = Settings(
        base_dir=Path(args.base_dir).resolve(),
        host=args.host,
        port=args.port,
        static_dir=static_dir,
    )
    set_settings(settings)
    app = create_app(settings)

    print(f"\nsmolvla-inspect web viewer")
    print(f"  Base directory: {settings.base_dir}")
    print(f"  Server: http://{args.host}:{args.port}")
    if static_dir:
        print(f"  Frontend: {static_dir}")
    else:
        print(f"  Frontend: not built (run `npm run build` in web/frontend/)")
        print(f"  For development, run `npm run dev` in web/frontend/ separately")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

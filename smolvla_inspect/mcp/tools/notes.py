"""Run notes tools (2 tools)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..server import mcp
from ..context import resolve_run
from ..helpers import success_response, error_response

NOTES_FILENAME = "run_notes.json"


@mcp.tool()
def get_run_notes(run_id: str) -> str:
    """Get notes for a run.

    Args:
        run_id: The run identifier.
    """
    try:
        run_dir, _ = resolve_run(run_id)
        notes_path = run_dir / NOTES_FILENAME
        if not notes_path.exists():
            return success_response({"notes": "", "updated_at": None})

        with open(notes_path) as f:
            data = json.load(f)
        return success_response({
            "notes": data.get("notes", ""),
            "updated_at": data.get("updated_at"),
        })
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")


@mcp.tool()
def set_run_notes(run_id: str, notes: str) -> str:
    """Set notes for a run. Overwrites existing notes.

    Args:
        run_id: The run identifier.
        notes: The notes text to save.
    """
    try:
        run_dir, _ = resolve_run(run_id)
        now = datetime.now(timezone.utc).isoformat()
        data = {"notes": notes, "updated_at": now}

        notes_path = run_dir / NOTES_FILENAME
        # Atomic write via temp file
        tmp_path = notes_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(notes_path)

        return success_response({"notes": notes, "updated_at": now})
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unexpected error: {e}")

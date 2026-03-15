#!/usr/bin/env python
"""Regenerate reports without needing torch/GPU dependencies.

Usage:
  python regenerate_report.py <run_dir>              # regenerate a single diagnostic report
  python regenerate_report.py compare <dir1> <dir2>  # generate a comparison report
"""

import importlib.util
import sys
import types

# ---------------------------------------------------------------------------
# Bootstrap: load only the lightweight modules (models, report, comparison)
# without triggering the full __init__.py → agent → torch import chain.
# ---------------------------------------------------------------------------

_pkg_root = "smolvla_inspect"
_diag = f"{_pkg_root}.diagnostic"


def _bootstrap():
    """Register package stubs and load models/report/comparison modules."""
    # Package stubs
    root = types.ModuleType(_pkg_root)
    root.__path__ = [_pkg_root]
    sys.modules[_pkg_root] = root

    diag = types.ModuleType(_diag)
    diag.__path__ = [f"{_pkg_root}/diagnostic"]
    diag.__package__ = _diag
    sys.modules[_diag] = diag

    # Helper to load a single module by file path
    def _load(mod_name, file_path):
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load(f"{_diag}.models", f"{_pkg_root}/diagnostic/models.py")
    _load(f"{_diag}.report", f"{_pkg_root}/diagnostic/report.py")
    _load(f"{_diag}.comparison", f"{_pkg_root}/diagnostic/comparison.py")


_bootstrap()

from smolvla_inspect.diagnostic.report import regenerate_report_markdown  # noqa: E402
from smolvla_inspect.diagnostic.comparison import compare_runs  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    if sys.argv[1] == "compare":
        if len(sys.argv) < 4:
            print("Usage: python regenerate_report.py compare <dir1> <dir2> [--labels l1 l2] [--output-dir dir]")
            sys.exit(1)

        # Parse args
        dirs = []
        labels = None
        output_dir = None
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--labels":
                labels = []
                i += 1
                while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                    labels.append(sys.argv[i])
                    i += 1
            elif sys.argv[i] == "--output-dir":
                i += 1
                output_dir = sys.argv[i]
                i += 1
            else:
                dirs.append(sys.argv[i])
                i += 1

        if len(dirs) < 2:
            print("ERROR: Need at least 2 run directories to compare.")
            sys.exit(1)

        if output_dir is None:
            import os
            output_dir = os.path.dirname(os.path.commonprefix(dirs)) or "."

        report = compare_runs(dirs, labels)
        json_path, md_path = report.save(output_dir)
        print(f"  Saved: {json_path}")
        print(f"  Saved: {md_path}")
    else:
        # Single report regeneration
        regenerate_report_markdown(sys.argv[1])


if __name__ == "__main__":
    main()

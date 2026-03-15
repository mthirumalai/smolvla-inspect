#!/usr/bin/env python
"""Regenerate report.md from an existing report.json."""

import sys
from smolvla_inspect.diagnostic.report import regenerate_report_markdown

if len(sys.argv) < 2:
    print("Usage: python regenerate_report.py <run_dir>")
    print("Example: python regenerate_report.py outputs/mthirumalai/finetuned_model")
    sys.exit(1)

regenerate_report_markdown(sys.argv[1])

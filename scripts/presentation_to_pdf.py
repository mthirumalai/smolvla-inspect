#!/usr/bin/env python3
"""Convert the HTML slide deck to a high-quality PDF using Playwright (Chromium).

Each slide is rendered at 1280x720 in a real browser, then assembled into
a single PDF with one slide per page.

Usage:
    .venv/bin/python scripts/presentation_to_pdf.py
"""

import os
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "diagnostic_agent_explained.html"
PDF_PATH = ROOT / "diagnostic_agent_explained.pdf"

WIDTH = 1920
HEIGHT = 1080


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        # Load the presentation
        page.goto(f"file://{HTML_PATH}", wait_until="networkidle")
        # Wait for fonts to load
        page.wait_for_timeout(2000)

        total = page.evaluate("document.querySelectorAll('.slide').length")
        print(f"Found {total} slides, rendering at {WIDTH}x{HEIGHT}...")

        slide_pdfs: list[str] = []
        tmpdir = tempfile.mkdtemp()

        for i in range(total):
            # Navigate to slide
            page.evaluate(f"go({i})")
            page.wait_for_timeout(300)  # let transitions settle

            # Hide nav and zoom controls for clean capture
            page.evaluate("""
                document.querySelectorAll('.nav, .zoom-controls, #progress').forEach(
                    el => el.style.display = 'none'
                );
            """)

            slide_pdf = os.path.join(tmpdir, f"slide_{i:03d}.pdf")
            page.pdf(
                path=slide_pdf,
                width=f"{WIDTH}px",
                height=f"{HEIGHT}px",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
                prefer_css_page_size=False,
            )
            slide_pdfs.append(slide_pdf)
            print(f"  Slide {i + 1}/{total}")

            # Restore UI (in case it affects next slide layout)
            page.evaluate("""
                document.querySelectorAll('.nav, .zoom-controls, #progress').forEach(
                    el => el.style.display = ''
                );
            """)

        browser.close()

    # Merge individual slide PDFs into one
    print("Merging slides...")
    _merge_pdfs(slide_pdfs, str(PDF_PATH))

    # Clean up temp files
    for f in slide_pdfs:
        os.remove(f)
    os.rmdir(tmpdir)

    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"\nDone: {PDF_PATH.relative_to(ROOT)} ({size_kb:.0f} KB, {total} pages)")


def _merge_pdfs(paths: list[str], output: str) -> None:
    """Merge multiple single-page PDFs into one using pypdf."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        from PyPDF2 import PdfWriter  # type: ignore[no-redef]

    writer = PdfWriter()
    for path in paths:
        writer.append(path)
    with open(output, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    main()

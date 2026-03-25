#!/usr/bin/env -S .venv/bin/python
"""Convert markdown reports and the HTML presentation to PDF.

Usage:
    python scripts/convert_to_pdf.py

Requires: markdown, weasyprint (pip install markdown weasyprint)
"""

import os
import sys

import markdown
from weasyprint import HTML

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")

# ── CSS for markdown reports ──────────────────────────────────────────────

REPORT_CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center { content: counter(page) " / " counter(pages); font-size: 9px; color: #999; }
}
body {
    font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #2c3042;
    max-width: 100%;
}
h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #2c3042;
    border-bottom: 2px solid #7b7ff0;
    padding-bottom: 6px;
    margin-top: 24pt;
    margin-bottom: 12pt;
}
h2 {
    font-size: 15pt;
    font-weight: 600;
    color: #3d4155;
    border-bottom: 1px solid #e6e4df;
    padding-bottom: 4px;
    margin-top: 20pt;
    margin-bottom: 10pt;
    page-break-after: avoid;
}
h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #5c6070;
    margin-top: 16pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}
p { margin: 6pt 0; }
blockquote {
    border-left: 3px solid #7b7ff0;
    margin: 10pt 0;
    padding: 8pt 12pt;
    background: #f8f7ff;
    color: #5c6070;
    font-size: 10pt;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0;
    font-size: 9.5pt;
    page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th {
    background: #f2f0ec;
    font-weight: 600;
    text-align: left;
    padding: 6px 8px;
    border: 1px solid #dbe4e8;
    color: #2c3042;
}
td {
    padding: 5px 8px;
    border: 1px solid #dbe4e8;
    color: #3d4155;
}
tr:nth-child(even) td { background: #fafaf8; }
code {
    background: #f2f0ec;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
    font-family: 'SF Mono', 'Menlo', monospace;
}
pre {
    background: #f2f0ec;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 9pt;
    line-height: 1.5;
    overflow-wrap: break-word;
    white-space: pre-wrap;
}
ul, ol { margin: 6pt 0; padding-left: 20pt; }
li { margin: 3pt 0; }
strong { color: #2c3042; }
hr { border: none; border-top: 1px solid #e6e4df; margin: 16pt 0; }
/* Severity badges */
em { font-style: italic; }
"""


def md_to_pdf(md_path: str, pdf_path: str) -> None:
    """Convert a single markdown file to PDF."""
    with open(md_path) as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{REPORT_CSS}</style></head>
<body>{html_body}</body></html>"""

    HTML(string=full_html, base_url=os.path.dirname(md_path)).write_pdf(pdf_path)
    print(f"  -> {os.path.relpath(pdf_path, ROOT)}")


# ── CSS overrides for presentation PDF ────────────────────────────────────

PRESENTATION_CSS = """
@page {
    size: 1280px 720px;
    margin: 0;
}
/* Show all slides, one per page */
.deck { position: static; width: 1280px; height: auto; }
.slide {
    position: static !important;
    display: flex !important;
    opacity: 1 !important;
    width: 1280px;
    height: 720px;
    page-break-after: always;
    page-break-inside: avoid;
    overflow: hidden;
}
/* Hide navigation and zoom controls */
.nav, .zoom-controls, #progress { display: none !important; }
"""


def presentation_to_pdf(html_path: str, pdf_path: str) -> None:
    """Convert the HTML slide deck to a multi-page PDF (one slide per page)."""
    with open(html_path) as f:
        html_text = f.read()

    # Inject our print CSS before the closing </style> or </head>
    inject = f"<style>{PRESENTATION_CSS}</style>"
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", f"{inject}</head>")

    HTML(string=html_text, base_url=os.path.dirname(html_path)).write_pdf(pdf_path)
    print(f"  -> {os.path.relpath(pdf_path, ROOT)}")


def main() -> None:
    # ── Collect all MD reports under outputs/ ─────────────────────────────
    md_files: list[str] = []
    for dirpath, _dirs, files in os.walk(OUTPUTS):
        for fn in files:
            if fn.endswith(".md"):
                md_files.append(os.path.join(dirpath, fn))
    md_files.sort()

    if not md_files:
        print("No markdown reports found under outputs/")
    else:
        print(f"Converting {len(md_files)} markdown report(s) to PDF...")
        for md_path in md_files:
            pdf_path = md_path.rsplit(".", 1)[0] + ".pdf"
            md_to_pdf(md_path, pdf_path)

    # ── Convert presentation ──────────────────────────────────────────────
    pres_html = os.path.join(ROOT, "diagnostic_agent_explained.html")
    pres_pdf = os.path.join(ROOT, "diagnostic_agent_explained.pdf")
    if os.path.exists(pres_html):
        print("\nConverting presentation to PDF...")
        presentation_to_pdf(pres_html, pres_pdf)
    else:
        print(f"Presentation not found: {pres_html}")

    print("\nDone.")


if __name__ == "__main__":
    main()

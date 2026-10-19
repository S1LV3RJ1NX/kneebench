"""PDF export with headless Chromium (Playwright), readable by people and LLMs.

Text stays text, every chart has a data table, the appendix lists every number, and the
Markdown and JSON reports are embedded as PDF attachments. Without Chromium, the web
report's "Print or save as PDF" button gives the same layout through the browser.
"""

from __future__ import annotations

import io
import subprocess
import sys

from pypdf import PdfReader, PdfWriter

INSTALL_HINT = (
    "PDF export needs a headless Chromium. Install it once with "
    "`uv run playwright install chromium` (on a bare Linux image also run "
    "`sudo uv run playwright install-deps chromium`), or use the report's "
    '"Print or save as PDF" button instead.'
)
READY_TIMEOUT_MS = 30_000


class PdfUnavailableError(RuntimeError):
    pass


def html_to_pdf(html: str) -> bytes:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - playwright is a core dependency
        raise PdfUnavailableError(INSTALL_HINT) from exc
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                page.wait_for_function("window.kneebenchReady === true", timeout=READY_TIMEOUT_MS)
                page.evaluate(
                    "document.querySelectorAll('details').forEach(d => d.setAttribute('open', ''))"
                )
                page.emulate_media(media="print")
                pdf: bytes = page.pdf(
                    format="A4",
                    print_background=True,
                    margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
                )
                return pdf
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
            raise PdfUnavailableError(INSTALL_HINT) from exc
        raise


def attach(pdf: bytes, attachments: dict[str, bytes]) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf)))
    for name, data in attachments.items():
        writer.add_attachment(name, data)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def chromium_installed() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            return bool(p.chromium.executable_path) and _exists(p.chromium.executable_path)
    except Exception:
        return False


def _exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()


def install_chromium() -> int:
    return subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"], check=False
    ).returncode

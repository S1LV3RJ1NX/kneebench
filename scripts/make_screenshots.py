"""Regenerate the README screenshots and the animated tour (dev tool).

    uv run python scripts/make_screenshots.py --db .kneebench/kneebench.sqlite3 \
        --sweep-run 3 --baseline-run 4

Works on a sanitized copy of the database (endpoint URLs and served model names are
replaced), serves the UI against it, and points the new-run form at a local fake vLLM
endpoint built from the recorded fixtures, so no real endpoint appears in any image.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from PIL import Image  # noqa: E402
from playwright.sync_api import Page, sync_playwright  # noqa: E402

from kneebench.store import repo  # noqa: E402
from kneebench.store.db import DB_FILENAME, connect  # noqa: E402
from kneebench.web.app import create_app  # noqa: E402
from tests.builders import ENDPOINT_FIXTURE  # noqa: E402
from tests.conftest import FakeServer, json_route, text_route  # noqa: E402

OUT = ROOT / "docs" / "images"
PLACEHOLDER_URL = "http://vllm.example:8000"
VIEWPORT = {"width": 1320, "height": 900}


def sanitized_home(db: Path, keep: dict[int, str]) -> Path:
    home = Path(tempfile.mkdtemp(prefix="kneebench-shots-"))
    shutil.copy(db, home / DB_FILENAME)
    conn = connect(home / DB_FILENAME)
    for row in list(conn.execute("SELECT id FROM runs")):
        if row["id"] not in keep:
            repo.delete_run(conn, row["id"])
    for run_id, label in keep.items():
        conn.execute("UPDATE runs SET label = ? WHERE id = ?", (label, run_id))
    conn.execute("UPDATE endpoints SET base_url = ?, name = ?", (PLACEHOLDER_URL, PLACEHOLDER_URL))
    conn.execute(
        "UPDATE snapshots SET served_model = 'llama-3.1-8b-instruct', metrics_text_z = NULL"
    )
    conn.close()
    return home


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def serve(home: Path) -> str:
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(home), host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return f"http://127.0.0.1:{port}"


def fake_vllm() -> FakeServer:
    server = FakeServer(
        {
            "/version": json_route({"version": "0.29.0"}),
            "/v1/models": text_route((ENDPOINT_FIXTURE / "models.json").read_text()),
            "/metrics": text_route((ENDPOINT_FIXTURE / "metrics.prom").read_text()),
            "/health": text_route(""),
        }
    )
    server.thread.start()
    return server


def shoot(page: Page, name: str, *, selector: str | None = None) -> Path:
    path = OUT / f"{name}.png"
    if selector:
        page.locator(selector).screenshot(path=str(path))
    else:
        page.screenshot(path=str(path))
    print("wrote", path.relative_to(ROOT))
    return path


def tour(frames: list[Path], out: Path, width: int = 1100, height: int = 750) -> None:
    images = []
    for frame in frames:
        img = Image.open(frame).convert("RGB")
        scale = width / img.width
        img = img.resize((width, int(img.height * scale)), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), (250, 249, 245))
        canvas.paste(img.crop((0, 0, width, min(height, img.height))), (0, 0))
        images.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=3200, loop=0, optimize=True
    )
    print("wrote", out.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--sweep-run", type=int, required=True)
    parser.add_argument("--baseline-run", type=int, required=True)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    home = sanitized_home(
        args.db, {args.sweep_run: "chat sweep, before tuning", args.baseline_run: "quick baseline"}
    )
    os.chdir(home)  # no kneebench.local.toml here, so the form starts empty
    base = serve(home)
    vllm = fake_vllm()
    frames: list[Path] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

            page.goto(f"{base}/runs/{args.sweep_run}")
            page.wait_for_function("window.kneebenchReady === true", timeout=30000)
            frames.append(shoot(page, "report"))
            page.locator("#capacity").scroll_into_view_if_needed()
            frames.append(shoot(page, "report-curve", selector="#rule-C02"))
            shoot(page, "report-walls", selector="#rule-W03")
            page.evaluate("window.kneebenchExpandAll()")
            page.wait_for_timeout(800)
            frames.append(shoot(page, "report-latency", selector="#rule-W02"))
            shoot(page, "report-baseline", selector="#rule-B01")

            page.goto(f"{base}/runs/new")
            page.fill("#url", vllm.url)
            page.click("text=Detect server")
            page.wait_for_selector("#detected table", timeout=30000)
            frames.append(shoot(page, "new-run"))

            page.goto(f"{base}/")
            frames.append(shoot(page, "runs"))
            browser.close()
    finally:
        vllm.httpd.shutdown()
    tour(frames, OUT / "tour.gif")
    shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sqlite3.register_adapter(bool, int)
    main()

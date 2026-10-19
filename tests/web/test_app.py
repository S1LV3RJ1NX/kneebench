"""Web UI routes, escaping, downloads, and a full wizard-to-report flow without AIPerf."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kneebench import service
from kneebench.detect.server import ModelMetadata
from kneebench.loadgen.base import Endpoint, PointResult, PointSpec
from kneebench.report.pdf import chromium_installed
from kneebench.store.db import DB_FILENAME, connect
from kneebench.web.app import create_app
from kneebench.web.manager import RunManager
from tests.builders import (
    BASELINE_FIXTURE,
    ENDPOINT_FIXTURE,
    fixture_snapshot,
    store_fixture_run,
)
from tests.conftest import FakeServer, json_route, text_route
from tests.fixture_data import LLAMA_8B, hf_fixture

HOSTILE = "<script>alert(1)</script>"


class ReplayLoadGenerator:
    """Pretends to run AIPerf by copying a recorded point's artifacts."""

    name = "replay"

    def version(self) -> str:
        return "0.13.0"

    def run_point(self, endpoint: Endpoint, spec: PointSpec, artifact_dir: Path) -> PointResult:
        shutil.copytree(BASELINE_FIXTURE, artifact_dir, dirs_exist_ok=True)
        return PointResult(
            spec=spec,
            artifact_dir=artifact_dir,
            exit_code=0,
            wall_time_s=1.0,
            argv_redacted=("replay",),
        )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def stored_run(home: Path) -> int:
    snap = fixture_snapshot()
    snap = replace(snap, models=[replace(snap.models[0], id=HOSTILE)])
    conn = connect(home / DB_FILENAME)
    run_id = store_fixture_run(conn, snapshot=snap)
    conn.close()
    return run_id


def test_pages_render(home: Path, stored_run: int) -> None:
    client = TestClient(create_app(home))
    assert client.get("/").status_code == 200
    assert client.get("/runs/new").status_code == 200
    page = client.get(f"/runs/{stored_run}")
    assert page.status_code == 200
    assert "Verdict" in page.text
    assert "{{" not in page.text
    assert client.get("/static/plotly.min.js").status_code == 200
    assert client.get("/static/kb.css").status_code == 200
    assert client.get("/runs/999").status_code == 404


def test_server_supplied_names_are_escaped(home: Path, stored_run: int) -> None:
    client = TestClient(create_app(home))
    html = client.get(f"/runs/{stored_run}").text + client.get("/").text
    assert HOSTILE not in html
    assert "&lt;script&gt;" in html


def test_downloads(home: Path, stored_run: int) -> None:
    client = TestClient(create_app(home))
    md = client.get(f"/runs/{stored_run}/report.md")
    assert md.status_code == 200
    assert "attachment" in md.headers["content-disposition"]
    assert md.text.startswith("# kneebench report")
    js = client.get(f"/runs/{stored_run}/report.json")
    assert js.json()["run_id"] == stored_run


def test_progress_of_a_finished_run_links_the_report(home: Path, stored_run: int) -> None:
    client = TestClient(create_app(home))
    html = client.get(f"/runs/{stored_run}/progress").text
    assert f"/runs/{stored_run}" in html
    assert "Finished" in html


def test_correct_gpu_recomputes_without_rerun(home: Path, stored_run: int) -> None:
    client = TestClient(create_app(home))
    resp = client.post(f"/runs/{stored_run}/gpu", data={"gpu": "h100-sxm"}, follow_redirects=False)
    assert resp.status_code == 303
    assert "H100 SXM" in client.get(f"/runs/{stored_run}/report.md").text


def _fake_endpoint(fake_server: type[FakeServer]) -> FakeServer:
    return fake_server(
        {
            "/version": json_route({"version": "0.29.0"}),
            "/v1/models": text_route((ENDPOINT_FIXTURE / "models.json").read_text()),
            "/metrics": text_route((ENDPOINT_FIXTURE / "metrics.prom").read_text()),
            "/health": text_route(""),
        }
    )


def test_detect_partial(home: Path, fake_server: type[FakeServer]) -> None:
    server = _fake_endpoint(fake_server)
    client = TestClient(create_app(home))
    html = client.post("/detect", data={"url": server.url, "api_key": ""}).text
    assert "0.29.0" in html
    assert "532,768" in html
    assert "unsloth/Meta-Llama-3.1-8B-Instruct" in html


def test_wizard_to_report(
    home: Path, fake_server: type[FakeServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _fake_endpoint(fake_server)
    config, params = hf_fixture(LLAMA_8B)
    monkeypatch.setattr(
        service, "fetch_model_metadata", lambda hf_id: ModelMetadata(hf_id, config, params)
    )
    manager = RunManager(home, loadgen=ReplayLoadGenerator())
    client = TestClient(create_app(home, manager))
    form = {
        "url": server.url,
        "gpu": "h100-nvl",
        "gpu_count": "1",
        "product": "chat",
        "slo_ttft_ms": "500",
        "slo_tpot_ms": "50",
        "slo_percentile": "99",
        "max_concurrency": "4",
        "enforce_eager": "",
        "api_key": "sk-never-stored",
    }

    plan = client.post("/plan", data=form).text
    assert "points" in plan

    resp = client.post("/runs", data=form, follow_redirects=False)
    assert resp.status_code == 303
    run_id = int(resp.headers["location"].split("/")[2])
    deadline = time.monotonic() + 30
    while manager.busy and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not manager.busy
    assert manager.active is not None and manager.active.error is None
    report = client.get(f"/runs/{run_id}")
    assert report.status_code == 200
    assert "How much load it takes" in report.text

    db_bytes = (home / DB_FILENAME).read_bytes()
    assert b"sk-never-stored" not in db_bytes


@pytest.mark.skipif(not chromium_installed(), reason="headless Chromium not installed")
def test_pdf_export_is_text_with_attachments(home: Path, stored_run: int) -> None:
    import io

    from pypdf import PdfReader

    client = TestClient(create_app(home))
    resp = client.get(f"/runs/{stored_run}/report.pdf")
    assert resp.status_code == 200
    reader = PdfReader(io.BytesIO(resp.content))
    assert set(reader.attachments) == {"report.md", "report.json"}
    text = "".join(page.extract_text() for page in reader.pages)
    assert "Verdict" in text
    assert "Appendix" in text

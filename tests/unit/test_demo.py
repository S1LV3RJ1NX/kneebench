"""The demo run: shipped sanitized, imported once, deletable, refreshable."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

from kneebench import demo
from kneebench.analysis.context import context_from_bundle
from kneebench.analysis.engine import evaluate
from kneebench.physics.gpus import catalog
from kneebench.store import repo
from kneebench.store.db import DB_FILENAME, connect
from kneebench.web.app import create_app
from tests.builders import store_fixture_run


def test_bundled_demo_is_sanitized_and_analyzes(tmp_path: Path) -> None:
    bundle = demo.bundled()
    assert bundle is not None
    text = json.dumps(bundle)
    public = {urlparse(g.source).hostname for g in catalog() if g.source}
    hosts = {urlparse(u).hostname for u in re.findall(r"https?://[^\s\"\\]+", text)}
    assert hosts <= {"vllm.example"} | public  # the placeholder and public GPU spec pages
    assert "/Users/" not in text and "/home/" not in text
    conn = connect(tmp_path / DB_FILENAME)
    run_id = demo.import_run(conn, bundle)
    ctx = context_from_bundle(repo.run_bundle(conn, run_id))
    rules = {f.rule_id for f in evaluate(ctx)}
    assert {"B01", "C02", "W03", "R01", "S06"} <= rules  # every phase is in the demo


def test_export_import_round_trip(tmp_path: Path) -> None:
    conn = connect(tmp_path / DB_FILENAME)
    run_id = store_fixture_run(conn)
    copy = demo.import_run(conn, demo.export_run(conn, run_id), label="copy")
    a = context_from_bundle(repo.run_bundle(conn, run_id))
    b = context_from_bundle(repo.run_bundle(conn, copy))
    assert copy != run_id
    assert a.points[0].metrics == b.points[0].metrics
    assert len(a.points[0].requests) == len(b.points[0].requests)


def test_seeded_once_and_stays_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KNEEBENCH_NO_DEMO")
    client = TestClient(create_app(tmp_path))
    demo_id = demo.demo_run_id(tmp_path)
    assert demo_id is not None
    assert "sample run shipped with kneebench" in client.get("/").text
    assert "This is the demo run" in client.get(f"/runs/{demo_id}").text
    client.post(f"/runs/{demo_id}/delete")
    TestClient(create_app(tmp_path))  # a restart doesn't bring it back
    conn = connect(tmp_path / DB_FILENAME)
    assert repo.get_run(conn, demo_id) is None
    assert repo.get_run(conn, demo.restore(conn, tmp_path)) is not None


def test_newer_demo_replaces_a_kept_one(tmp_path: Path) -> None:
    conn = connect(tmp_path / DB_FILENAME)
    first = demo.seed(conn, tmp_path)
    assert first is not None
    (tmp_path / demo.MARKER).write_text(json.dumps({"version": "old", "run_id": first}))
    assert demo.seed(conn, tmp_path) is not None
    assert len(repo.list_runs(conn)) == 1  # replaced, not added
    assert json.loads((tmp_path / demo.MARKER).read_text())["version"] != "old"
    assert demo.seed(conn, tmp_path) is None  # same version: nothing to do

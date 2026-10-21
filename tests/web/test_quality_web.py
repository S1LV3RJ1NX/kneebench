"""The quality steps of a guided experiment: record before, ask after, read the verdict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kneebench.quality import service as qservice
from kneebench.quality.ask import Answer
from kneebench.quality.evalset import EvalSet
from kneebench.store import repo
from kneebench.store.db import DB_FILENAME, connect
from kneebench.web.app import create_app
from tests.builders import store_fixture_run


def _experiment(home: Path) -> int:
    conn = connect(home / DB_FILENAME)
    run_id = store_fixture_run(conn)
    endpoint_id = conn.execute("SELECT endpoint_id FROM runs WHERE id = ?", (run_id,)).fetchone()[0]
    return repo.create_experiment(
        conn, endpoint_id=endpoint_id, base_run_id=run_id, rule_id="F02",
        title="Store the KV cache in FP8", reason="r", flag="--kv-cache-dtype fp8",
        setting="--kv-cache-dtype", value="fp8",
    )  # fmt: skip


def test_quality_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_ask(endpoint: Any, evalset: EvalSet, *_: Any, **__: Any) -> list[Answer]:
        calls.append(endpoint.api_key)
        after = len(calls) > 2
        return [
            # The change breaks the first five arithmetic answers.
            Answer(
                i.id, i.category, "wrong" if after and n < 5 else i.answer, not (after and n < 5)
            )
            for n, i in enumerate(evalset.items)
        ]

    monkeypatch.setattr(qservice, "ask_all", fake_ask)
    experiment = _experiment(tmp_path)
    client = TestClient(create_app(tmp_path))
    page = client.get(f"/experiments/{experiment}").text
    assert "Record answers" in page and "Record answers in step 02" in page

    resp = client.post(
        f"/experiments/{experiment}/quality/before",
        data={"evalset": "starter", "max_drop": "1.0", "max_flip_excess": "2.0", "api_key": "sk-q"},
    )
    assert resp.status_code == 204
    page = client.get(f"/experiments/{experiment}").text
    assert "Recorded 290 questions" in page and "flipped 0.0%" in page

    resp = client.post(f"/experiments/{experiment}/quality/after", data={"api_key": "sk-q"})
    assert resp.status_code == 204
    page = client.get(f"/experiments/{experiment}").text
    assert "Changed answers (5)" in page
    # 5 of 290 lost: -1.7 +/- 1.5 points straddles the 1-point budget.
    assert 'badge warn">inconclusive' in page and "Inconclusive: 290 questions" in page
    assert calls == ["sk-q", "sk-q", "sk-q"]
    assert b"sk-q" not in (tmp_path / DB_FILENAME).read_bytes()


def test_bad_upload_is_explained(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    client = TestClient(create_app(tmp_path))
    resp = client.post(
        f"/experiments/{experiment}/quality/before",
        data={"evalset": "file"},
        files={"file": ("mine.jsonl", b'{"prompt": "x"}\n', "application/json")},
    )
    assert "missing answer" in resp.text

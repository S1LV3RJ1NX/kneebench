"""Run status is always in the database: done, failed (with why), cancelled, interrupted."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from kneebench.domain import STANDARD_SHAPES, ProductType, ServerSetup, Slo, Workload
from kneebench.loadgen.base import Endpoint, PointResult, PointSpec
from kneebench.phases.plan import plan_run
from kneebench.phases.runner import RunInputs, execute_run
from kneebench.physics.gpus import get_gpu
from kneebench.service import delete_run_data
from kneebench.store import repo
from kneebench.store.db import connect_memory
from tests.builders import fixture_snapshot, store_fixture_run, theory_for


class Boom:
    name = "boom"

    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code

    def version(self) -> str:
        return "0.13.0"

    def run_point(
        self,
        endpoint: Endpoint,
        spec: PointSpec,
        artifact_dir: Path,
        timeout_s: float | None = None,
        cancel: threading.Event | None = None,
    ) -> PointResult:
        if self.exit_code is None:
            raise RuntimeError("disk full")
        return PointResult(
            spec=spec,
            artifact_dir=artifact_dir,
            exit_code=self.exit_code,
            wall_time_s=0.1,
            argv_redacted=("aiperf",),
            stderr_tail="Error: connection refused",
        )


def _inputs() -> RunInputs:
    snap = fixture_snapshot()
    return RunInputs(
        endpoint=Endpoint(base_url=snap.base_url, model="m", tokenizer="builtin"),
        snapshot=snap,
        served=snap.models[0],
        metadata=None,
        setup=ServerSetup(gpu=get_gpu("h100-nvl")),
        theory=theory_for(),
        workload=Workload(ProductType.CHAT, STANDARD_SHAPES, Slo(500, 50)),
    )


def test_unexpected_exception_marks_the_run_failed(tmp_path: Path) -> None:
    conn = connect_memory()
    plan = plan_run(theory_for(), ("baseline",))
    with pytest.raises(RuntimeError):
        execute_run(conn, _inputs(), plan, Boom(), tmp_path)
    run = conn.execute("SELECT status, error, planned_points FROM runs").fetchone()
    assert run["status"] == "failed"
    assert "disk full" in run["error"]
    assert run["planned_points"] == 1


def test_failed_point_records_why(tmp_path: Path) -> None:
    conn = connect_memory()
    run_id = execute_run(conn, _inputs(), plan_run(theory_for(), ("baseline",)), Boom(1), tmp_path)
    run = repo.get_run(conn, run_id)
    assert run is not None
    assert run["status"] == "failed"
    assert "exit code 1" in run["error"] and "connection refused" in run["error"]


def test_stale_running_runs_become_interrupted() -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn)
    conn.execute("UPDATE runs SET status = 'running' WHERE id = ?", (run_id,))
    assert repo.mark_interrupted(conn, live_run_ids={run_id}) == []
    assert repo.mark_interrupted(conn, live_run_ids=set()) == [run_id]
    run = repo.get_run(conn, run_id)
    assert run is not None
    assert run["status"] == "interrupted"
    assert run["error"]


def test_progress_comes_from_the_database() -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn)
    repo.set_planned_points(conn, run_id, 4)
    prog = repo.progress(conn, run_id)
    assert (prog["done"], prog["total"], prog["status"]) == (1, 4, "done")


def test_delete_removes_run_rows_and_artifacts(tmp_path: Path) -> None:
    conn = connect_memory()
    keep = store_fixture_run(conn)
    gone = store_fixture_run(conn)
    artifacts = tmp_path / "runs" / str(gone)
    artifacts.mkdir(parents=True)
    (artifacts / "x.json").write_text("{}")

    delete_run_data(conn, tmp_path, gone)

    assert repo.get_run(conn, gone) is None
    assert repo.get_run(conn, keep) is not None
    assert not artifacts.exists()
    for table in ("points", "findings"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (gone,))
        assert count.fetchone()[0] == 0, table
    for table in ("requests", "point_metrics", "server_metrics", "artifacts"):
        orphans = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE point_id NOT IN (SELECT id FROM points)"
        )
        assert orphans.fetchone()[0] == 0, table
    assert conn.execute("SELECT COUNT(*) FROM setups").fetchone()[0] == 1
    with pytest.raises(KeyError):
        repo.delete_run(conn, gone)

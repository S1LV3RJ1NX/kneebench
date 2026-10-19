"""SQLite round trip: a stored run regenerates a report with no network."""

from kneebench.analysis.context import context_from_bundle
from kneebench.domain import ProductType
from kneebench.report.build import build_report
from kneebench.store import repo
from kneebench.store.db import connect_memory, migrate
from tests.builders import fixture_snapshot, store_fixture_run, without_metrics


def test_migrations_are_idempotent() -> None:
    conn = connect_memory()
    latest = migrate(conn)
    assert latest >= 2
    assert migrate(conn) == latest
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(requests)")}
    assert "error_code" in columns


def test_fixture_run_round_trip() -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn)
    bundle = repo.run_bundle(conn, run_id)
    (p,) = bundle["points"]
    assert p["status"] == "done"
    assert p["metrics"]["request_count"]["avg"] == 30
    assert "vllm:kv_cache_usage_perc" in p["server"]
    assert len(repo.point_requests(conn, p["id"])) == 30
    stored = {r["name"] for r in conn.execute("SELECT name FROM artifacts")}
    assert {"profile_export_aiperf.json", "profile_export.jsonl"} <= stored

    ctx = context_from_bundle(bundle)
    assert ctx.baseline is not None
    assert ctx.theory.pool_tokens == 532_768
    assert ctx.http_rtt_ms == 272.0


def test_build_report_caches_findings() -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn)
    report = build_report(conn, run_id, now="2026-01-01T00:00:00+00:00")
    assert report.verdict
    cached = {r["rule_id"] for r in conn.execute("SELECT rule_id FROM findings")}
    assert {"T01", "B01", "H03", "H07"} <= cached


def test_no_metrics_run_reports_blind_spots() -> None:
    conn = connect_memory()
    run_id = store_fixture_run(
        conn, snapshot=without_metrics(fixture_snapshot()), product=ProductType.REASONING
    )
    report = build_report(conn, run_id, now="2026-01-01T00:00:00+00:00")
    ids = {f.rule_id for f in report.problems}
    assert "H08" in ids
    assert any("not reachable" in c for c in report.caveats)

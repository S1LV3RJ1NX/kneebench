"""Rendered reports for fixture workloads. Update with: uv run pytest --snapshot-update"""

import pytest
from syrupy.assertion import SnapshotAssertion

from kneebench.domain import ProductType
from kneebench.report.build import build_report
from kneebench.report.render_json import render_json
from kneebench.report.render_md import render_markdown
from kneebench.store.db import connect_memory
from tests.builders import fixture_snapshot, store_fixture_run, without_metrics

NOW = "2026-01-01T00:00:00+00:00"


@pytest.mark.parametrize("product", [ProductType.CHAT, ProductType.RAG, ProductType.REASONING])
def test_markdown_report_per_workload(product: ProductType, snapshot: SnapshotAssertion) -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn, product=product)
    assert render_markdown(build_report(conn, run_id, now=NOW)) == snapshot


def test_markdown_report_with_wrong_gpu_entry(snapshot: SnapshotAssertion) -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn, gpu="h100-sxm")
    report = build_report(conn, run_id, now=NOW)
    assert report.problems[0].rule_id == "T07"  # setup errors outrank everything
    assert render_markdown(report) == snapshot


def test_markdown_report_without_metrics(snapshot: SnapshotAssertion) -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn, snapshot=without_metrics(fixture_snapshot()))
    assert render_markdown(build_report(conn, run_id, now=NOW)) == snapshot


def test_json_report_carries_numbers_and_text() -> None:
    conn = connect_memory()
    run_id = store_fixture_run(conn)
    data = render_json(build_report(conn, run_id, now=NOW))
    assert '"rule_id": "B01"' in data
    assert '"mbu"' in data


@pytest.mark.parametrize("product", [ProductType.CHAT, ProductType.REASONING])
def test_sweep_report_renders_every_section(
    product: ProductType, snapshot: SnapshotAssertion
) -> None:
    from kneebench.analysis.engine import evaluate
    from kneebench.report.composer import compose
    from kneebench.report.render_html import render_html
    from tests.builders import ctx, healthy_sweep, point

    context = ctx(points=(point(), *healthy_sweep()), product=product)
    findings = evaluate(context)
    md = render_markdown(compose(context, findings, now=NOW))
    assert "How much load it takes" in md
    assert "comfortably handles about 32 users" in md
    html = render_html(compose(context, findings, now=NOW, style="html"))
    assert 'data-chart="C02-0"' in html
    assert 'data-chart="W03-0"' in html
    assert md == snapshot


@pytest.mark.parametrize("product", [ProductType.CHAT, ProductType.RAG])
def test_shape_report_renders_the_shape_section(
    product: ProductType, snapshot: SnapshotAssertion
) -> None:
    from kneebench.analysis.engine import evaluate
    from kneebench.report.composer import compose
    from kneebench.report.render_html import render_html
    from tests.builders import ctx, healthy_sweep, point, shape_sweep

    context = ctx(points=(point(), *healthy_sweep(), *shape_sweep()), product=product)
    findings = evaluate(context)
    report = compose(context, findings, now=NOW)
    assert report.shapes[0].rule_id == "S06" or any(p.rule_id == "S06" for p in report.problems)
    md = render_markdown(report)
    assert "How request size changes things" in md
    html = render_html(compose(context, findings, now=NOW, style="html"))
    for rid in ("S01", "S02", "S03", "S04", "S06"):
        assert f'data-chart="{rid}-0"' in html
    assert md == snapshot

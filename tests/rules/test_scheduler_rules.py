"""E01-E02: what the engine reports about queueing and its admission limits."""

from kneebench.analysis.context import PointView
from kneebench.analysis.engine import Status
from tests.builders import ctx, engine_metrics, point, run_rule, sweep_point


def _sweep() -> tuple[PointView, ...]:
    """Queueing starts at 16 users with room everywhere; at 64 the 8,192-token budget is full."""
    rows = [  # c, waiting max, engine metrics
        (1, 0.0, engine_metrics(0.07, 0.0, 0.0, 0.06, 1)),
        (16, 6.0, engine_metrics(0.17, 0.0005, 0.3, 0.14, 2200)),
        (64, 43.0, engine_metrics(0.45, 0.035, 3.1, 0.38, 7000)),
    ]
    return tuple(
        sweep_point(c, 150.0 * c, 6.0, waiting_max=w, engine=e, ttft_p50=300.0 + 100 * i, seq=i)
        for i, (c, w, e) in enumerate(rows)
    )


def test_e01_splits_first_token_time() -> None:
    f = run_rule("E01", ctx(points=(point(), *_sweep())))
    assert f is not None
    assert f.status is Status.INFO
    first = f.data["first"]
    assert first["engine_ms"] == 70.0
    assert first["outside_ms"] == 300.0 - 70.0
    assert f.data["queued_from"] is None  # the queue never outweighs the prompt read
    assert run_rule("E01", ctx(points=(point(),))) is None


def test_e02_names_the_limit_that_fills() -> None:
    f = run_rule("E02", ctx(points=(point(), *_sweep())))
    assert f is not None
    assert [r["gate"] for r in f.data["rows"]] == ["unknown", "token_budget"]
    assert f.data["first"]["concurrency"] == 16
    assert f.data["known"]["concurrency"] == 64
    assert f.data["safe_budget"] is not None
    quiet = engine_metrics(0.07, 0.0, 0.001, 0.06, 100)
    calm = tuple(sweep_point(c, 150.0 * c, 6.0, engine=quiet, seq=i) for i, c in enumerate((1, 8)))
    ok = run_rule("E02", ctx(points=(point(), *calm)))
    assert ok is not None and ok.status is Status.OK
    blind = tuple(sweep_point(c, 150.0 * c, 6.0, seq=i) for i, c in enumerate((1, 8)))
    assert run_rule("E02", ctx(points=(point(), *blind))) is None

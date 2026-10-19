"""R01-R04: random and bursty arrivals."""

import pytest

from kneebench.analysis.engine import Status
from kneebench.analysis.rate import capacity_rps
from tests.builders import ctx, healthy_sweep, point, rate_point, rate_sweep, run_rule


def _ctx(rates: tuple = ()):  # type: ignore[no-untyped-def,type-arg]
    return ctx(points=(point(), *healthy_sweep(), *rates))


def test_capacity_comes_from_the_concurrency_sweep() -> None:
    mu = capacity_rps(_ctx().points_in("concurrency"))
    assert mu == pytest.approx(3080 / 1024, rel=0.05)


def test_r01_max_rate_within_targets() -> None:
    f = run_rule("R01", _ctx(rate_sweep()))
    assert f is not None
    assert f.status is Status.INFO
    assert f.data["max_rps"] == pytest.approx(2.6)  # 3.2 req/s has a growing queue
    level = run_rule("R01", _ctx((rate_point(2.2), rate_point(3.1, seq=61))))
    assert level is not None
    assert level.data["max_rps"] == pytest.approx(2.2)  # 3.1 is above the 3.0 capacity
    assert f.data["knee_rps"] == pytest.approx(2.2)
    assert len(f.data["rows"]) == 5
    assert run_rule("R01", _ctx()) is None


def test_r01_warns_when_nothing_meets_the_target() -> None:
    f = run_rule("R01", _ctx((rate_point(0.7, ttft_ms=900),)))
    assert f is not None
    assert f.status is Status.WARN


def test_r02_run_length_dependence() -> None:
    f = run_rule("R02", _ctx(rate_sweep()))
    assert f is not None
    assert f.status is Status.INFO
    assert f.data["rate"] == pytest.approx(3.2)
    assert f.data["end_ms"] > 4 * f.data["start_ms"]
    early = run_rule("R02", _ctx((rate_point(1.0, 200, 6000),)))
    assert early is not None
    assert early.status is Status.WARN
    assert run_rule("R02", _ctx((rate_point(1.0),))).status is Status.OK  # type: ignore[union-attr]


def test_r03_client_shortfall() -> None:
    assert run_rule("R03", _ctx(rate_sweep())).status is Status.OK  # type: ignore[union-attr]
    f = run_rule("R03", _ctx((rate_point(3.0, issued_share=0.7),)))
    assert f is not None
    assert f.status is Status.WARN


def test_r04_burst_penalty() -> None:
    f = run_rule("R04", _ctx(rate_sweep()))
    assert f is not None
    assert f.status is Status.WARN  # 420 -> 1,400 ms at 2.6 req/s breaks the 500 ms target
    assert f.data["breaks"] is True
    assert f.data["worst"]["penalty"] == pytest.approx(1400 / 420, rel=0.01)
    assert run_rule("R04", _ctx((rate_point(1.0),))) is None

"""Steady-state throughput and sweep analysis on synthetic, exactly known traffic."""

import pytest

from kneebench.analysis.context import RequestView
from kneebench.analysis.steady import steady_state
from kneebench.analysis.sweep import (
    curve,
    first_crossing,
    meets,
    operating_point,
    saturation,
    steps,
)
from kneebench.domain import Slo
from tests.builders import closed_loop_requests, healthy_sweep


def test_closed_loop_steady_throughput_excludes_ramp_and_drain() -> None:
    # 8 slots x 5 rounds of 10 s requests with 1,000 tokens: 800 tok/s while fully loaded.
    reqs = closed_loop_requests(concurrency=8, rounds=5, e2e_s=10.0, osl=1000)
    s = steady_state(reqs, concurrency=8)
    assert s is not None
    assert s.output_tok_s == pytest.approx(800, rel=0.01)
    assert s.duration_s == pytest.approx(40, rel=0.01)  # last issue at 40 s, run ends at 50 s
    assert s.reliable


def test_ramp_up_is_not_counted() -> None:
    # Slots start 5 s apart: the naive average would include the half-empty ramp.
    base = closed_loop_requests(concurrency=4, rounds=6, e2e_s=10.0, osl=1000)
    staggered = tuple(
        RequestView(
            r.start_ns + slot * 5_000_000_000,
            r.end_ns + slot * 5_000_000_000,
            r.ttft_ms,
            r.tpot_ms,
            r.e2e_ms,
            r.isl,
            r.osl,
        )
        for slot, r in ((i // 6, r) for i, r in enumerate(base))
    )
    s = steady_state(staggered, concurrency=4)
    assert s is not None
    assert s.start_ns >= 15_000_000_000  # fully loaded once the 4th slot starts
    assert s.output_tok_s == pytest.approx(400, rel=0.05)


def test_short_run_is_flagged_unreliable() -> None:
    reqs = closed_loop_requests(concurrency=4, rounds=2, e2e_s=5.0, osl=500)
    s = steady_state(reqs, concurrency=4)
    assert s is not None
    assert not s.reliable


def test_too_few_requests() -> None:
    assert steady_state(closed_loop_requests(1, 1, 5.0, 100), concurrency=1) is None


def test_curve_knee_saturation_crossing_and_operating_point() -> None:
    pts = curve(healthy_sweep())
    assert [p.concurrency for p in pts] == [1, 2, 4, 8, 16, 32, 64]
    assert all(p.steady_reliable for p in pts)
    assert pts[0].output_tok_s == pytest.approx(170, rel=0.02)

    gains = steps(pts)
    assert gains[0].efficiency > 0.9  # 1 -> 2 users nearly doubles throughput
    assert gains[-1].gain < 0.05

    sat = saturation(pts)
    assert sat is not None
    assert sat.concurrency == 32

    slo = Slo(ttft_ms=500, tpot_ms=50, percentile=99)
    cross = first_crossing(pts, "tpot", 50, 99)
    assert cross is not None
    assert cross.last_pass is not None
    assert cross.last_pass.concurrency == 32
    assert cross.concurrency == pytest.approx(2**5.4, rel=0.01)
    op = operating_point(pts, slo)
    assert op is not None
    assert op.concurrency == 32
    assert first_crossing(pts, "ttft", 500, 99) is None


def test_errors_fail_the_target() -> None:
    pts = curve(healthy_sweep(errors=1))
    assert not any(meets(p, Slo(500, 50)) for p in pts)

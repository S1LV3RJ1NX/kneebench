"""Estimates for a request size the run didn't test."""

import pytest

from kneebench.analysis.context import RunContext
from kneebench.analysis.whatif import estimate
from tests.builders import ctx, healthy_sweep, point, shape_sweep


def _ctx() -> RunContext:
    return ctx(points=(point(), *healthy_sweep(), *shape_sweep()))


def test_estimate_inside_the_measured_range() -> None:
    e = estimate(_ctx(), 1024, 512)
    assert e is not None
    assert e.ttft_ms == pytest.approx(10 + 0.03 * 1024 - 1.0, rel=0.01)  # 1 ms round trip removed
    assert e.tpot_ms == pytest.approx(5.8, rel=0.01)
    assert e.e2e_ms == pytest.approx((e.ttft_ms or 0) + 5.8 * 511, rel=0.01)
    assert e.loaded_concurrency == 32
    assert e.loaded_tpot_ms is not None and 10.5 < e.loaded_tpot_ms < 18.5
    assert e.fits == 532_768 // 1536
    assert e.fits_context
    assert e.extrapolated == ()


def test_estimate_flags_extrapolation_and_context() -> None:
    e = estimate(_ctx(), 16_000, 9000)
    assert e is not None
    assert set(e.extrapolated) == {"prompt", "output"}
    assert not e.fits_context


def test_no_estimate_without_a_shape_sweep() -> None:
    assert estimate(ctx(points=(point(), *healthy_sweep())), 1000, 100) is None

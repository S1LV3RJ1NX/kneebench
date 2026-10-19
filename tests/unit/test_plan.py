"""Sweep planning, ETA and the early-stop policy."""

import pytest

from kneebench.analysis.sweep import curve
from kneebench.domain import ProductType, Slo
from kneebench.loadgen.aiperf.overhead import DEFAULT_STARTUP_OVERHEAD_S
from kneebench.phases.plan import concurrency_levels, estimate_e2e_ms, plan_run
from kneebench.phases.runner import sweep_stop_reason
from tests.builders import healthy_sweep, sweep_point, theory_for

SLO = Slo(ttft_ms=500, tpot_ms=50)


def test_levels_double_up_to_the_lowest_cap() -> None:
    t = theory_for()
    chat = t.shape("chat")
    assert chat is not None
    assert concurrency_levels(t, chat, 256, None) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert concurrency_levels(t, chat, 256, 16)[-1] == 32  # twice the server's own cap
    reasoning = t.shape("reasoning")
    assert reasoning is not None
    assert concurrency_levels(t, reasoning, 1024, None)[-1] == 256  # ~2x what memory holds


def test_plan_includes_baseline_sweep_and_end_baseline() -> None:
    plan = plan_run(theory_for(), ("baseline", "concurrency"), slo=SLO)
    phases = [p.phase for p in plan.points]
    assert phases[0] == "baseline"
    assert phases[-1] == "baseline_end"
    sweep = [p for p in plan.points if p.phase == "concurrency"]
    assert [p.spec.concurrency for p in sweep][-1] == 256
    assert all(
        p.spec.request_count and p.spec.request_count >= 4 * (p.spec.concurrency or 1)
        for p in sweep
    )
    assert all(p.spec.goodput for p in sweep)
    assert all(p.eta_s > DEFAULT_STARTUP_OVERHEAD_S for p in plan.points)


def test_product_picks_the_primary_shape() -> None:
    plan = plan_run(theory_for(), ("concurrency",), product=ProductType.RAG, slo=SLO)
    spec = next(p.spec for p in plan.points if p.phase == "concurrency")
    assert spec.isl > spec.osl  # summarization-style shape


def test_eta_grows_with_concurrency_and_context() -> None:
    t = theory_for()
    chat = t.shape("chat")
    assert chat is not None
    assert estimate_e2e_ms(t, chat, 64) > estimate_e2e_ms(t, chat, 1)
    assert estimate_e2e_ms(t, chat, 1024) > 3 * estimate_e2e_ms(t, chat, 256)  # memory wall


def test_budget_trims_the_top_of_the_sweep() -> None:
    full = plan_run(theory_for(), ("baseline", "concurrency"), slo=SLO)
    trimmed = plan_run(theory_for(), ("baseline", "concurrency"), slo=SLO, budget_s=full.eta_s / 3)
    assert trimmed.eta_s < full.eta_s
    assert trimmed.dropped_for_budget
    assert max(trimmed.dropped_for_budget) == 256


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        (healthy_sweep()[:3], None),
        (healthy_sweep(), None),  # one flat doubling is not enough
        ((*healthy_sweep(), sweep_point(128, 3100, 40.0, 160.0, seq=8)), "walls_found"),
        ((sweep_point(8, 1000, 6.0, errors=20, seq=1),), "errors"),
        ((sweep_point(8, 1000, 6.0, errors=5, error_code=429, seq=1),), "gateway_errors"),
    ],
)
def test_sweep_stops_early(points: tuple, reason: str | None) -> None:  # type: ignore[type-arg]
    assert sweep_stop_reason(curve(points), SLO) == reason

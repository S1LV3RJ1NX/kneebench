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


def test_rate_phase_follows_the_concurrency_sweep() -> None:
    plan = plan_run(theory_for(), ("baseline", "concurrency", "rate"), slo=SLO)
    rate = [p for p in plan.points if p.phase == "rate"]
    assert [p.rate_factor for p in rate] == [0.25, 0.5, 0.75, 0.9, 1.1, None, None, 0.75, 0.9]
    assert [p.refine for p in rate] == [False] * 5 + [True, True] + [False] * 2
    assert {p.spec.arrival.value for p in rate} == {"poisson", "gamma"}
    assert all(p.spec.duration_s == 90 for p in rate)
    phases = [p.phase for p in plan.points]
    assert phases.index("rate") > max(i for i, ph in enumerate(phases) if ph == "concurrency")
    assert phases[-1] == "baseline_end"
    with pytest.raises(ValueError, match="concurrency sweep"):
        plan_run(theory_for(), ("baseline", "rate"), slo=SLO)


def test_rate_points_are_resolved_from_measured_capacity() -> None:
    from kneebench.phases.runner import _resolve_rate

    plan = plan_run(theory_for(), ("concurrency", "rate"), slo=SLO)
    planned = next(p for p in plan.points if p.phase == "rate" and p.rate_factor == 0.5)
    resolved, skip = _resolve_rate(planned, curve(healthy_sweep()), None)
    assert skip is None
    assert resolved.spec.request_rate == pytest.approx(0.5 * 3080 / 1024, rel=0.05)
    _, skip = _resolve_rate(planned, [], None)
    assert skip == "no_capacity_measured"


def test_shape_phase_sweeps_one_axis_at_a_time() -> None:
    from kneebench.phases.plan import axis_lengths

    assert axis_lengths(8192, 256) == [128, 512, 2048, 6656]  # capped, with room for variation
    assert axis_lengths(32_768, 1024) == [128, 512, 2048, 8192, 27_136]
    plan = plan_run(theory_for(), ("baseline", "concurrency", "shape"), slo=SLO)
    prompt = [p for p in plan.points if p.phase == "shape_prompt"]
    output = [p for p in plan.points if p.phase == "shape_output"]
    shape_map = [p for p in plan.points if p.phase == "shape_map"]
    assert {p.spec.osl for p in prompt} == {256}
    assert {p.spec.isl for p in output} == {1024}
    assert sum(not p.at_operating_point for p in prompt) == len(prompt) // 2  # 1 user and loaded
    assert {(p.spec.isl, p.spec.osl) for p in shape_map} == {(1024, 5988), (5988, 1024)}
    assert all(p.at_operating_point for p in shape_map)
    # Three deviations of length variation still fit the context window.
    assert all((p.spec.isl + p.spec.osl) * 1.15 + 128 <= 8192 for p in prompt + output + shape_map)
    with pytest.raises(ValueError, match="concurrency sweep"):
        plan_run(theory_for(), ("baseline", "shape"), slo=SLO)


def test_shape_points_run_at_the_operating_point_sized_by_time() -> None:
    from kneebench.phases.runner import _resolve_operating_point

    plan = plan_run(theory_for(), ("concurrency", "shape"), slo=SLO)
    short = next(p for p in plan.points if p.phase == "shape_output" and p.at_operating_point)
    long = [p for p in plan.points if p.phase == "shape_output" and p.at_operating_point][-1]
    resolved, skip = _resolve_operating_point(short, curve(healthy_sweep()), SLO)
    assert skip is None
    assert resolved.spec.concurrency == 32  # the last level within the targets
    assert resolved.spec.request_count == 4 * 32
    resolved_long, _ = _resolve_operating_point(long, curve(healthy_sweep()), SLO)
    assert resolved_long.spec.request_count == 2 * 32  # long answers: fewer rounds
    _, skip = _resolve_operating_point(short, [], SLO)
    assert skip == "no_operating_point"


def test_interference_points_follow_lesson_44() -> None:
    from kneebench.loadgen.base import LoadMode

    plan = plan_run(theory_for(), ("baseline", "concurrency", "interference"), slo=SLO)
    alone, mixed = [p for p in plan.points if p.phase.startswith("interference")]
    assert alone.spec.mode is LoadMode.SCHEDULE and len(alone.spec.schedule) == 32
    long = [s for s in mixed.spec.schedule if s[1] > 1000]
    assert [s[0] for s in long] == [5000, 15000, 25000, 35000]
    assert all((s[1] + s[2]) * 1.15 + 128 <= 8192 for s in mixed.spec.schedule)
    with pytest.raises(ValueError, match="concurrency sweep"):
        plan_run(theory_for(), ("baseline", "interference"), slo=SLO)


def test_refine_rate_bisects_the_pass_fail_gap() -> None:
    from kneebench.phases.plan import refine_rate

    results = [(1.0, True), (2.0, True), (3.0, True), (3.6, False), (4.4, False)]
    assert refine_rate(results, 4.0) == 3.3
    assert refine_rate([*results, (3.3, True)], 4.0) == 3.45
    assert refine_rate([*results, (3.3, True), (3.45, False)], 4.0) is None  # gap under 5%
    assert refine_rate([(1.0, True), (2.0, True)], 4.0) is None  # nothing failed
    assert refine_rate([(1.0, False)], 4.0) == 0.5  # even the lowest rate failed

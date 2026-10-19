"""C01-C05, W01-W09 and the sweep hygiene rules H01, H02, H06, H09."""

import pytest

from kneebench.analysis.engine import Status
from kneebench.domain import ProductType
from tests.builders import CACHE_65GIB, ctx, healthy_sweep, point, run_rule, sweep_point, theory_for


def _ctx(points: tuple = (), **kw: object):  # type: ignore[no-untyped-def,type-arg]
    return ctx(points=(point(), *points), **kw)  # type: ignore[arg-type]


def test_c01_free_lunch() -> None:
    f = run_rule("C01", _ctx(healthy_sweep()))
    assert f is not None
    assert f.data["concurrency"] == 32
    assert f.data["multiple"] == pytest.approx(3000 / 170, rel=0.05)
    assert f.data["tpot_increase"] == pytest.approx(10.5 / 5.8 - 1, rel=0.01)
    assert run_rule("C01", _ctx()) is None


def test_c02_regions_and_knee() -> None:
    f = run_rule("C02", _ctx(healthy_sweep()))
    assert f is not None
    regions = [s["region"] for s in f.data["steps"]]
    assert regions[0] == "flat"
    assert regions[-1] == "wall"
    assert f.data["knee"] is not None
    assert len(f.data["points"]) == 7


def test_c03_operating_point_states() -> None:
    good = run_rule("C03", _ctx(healthy_sweep()))
    assert good is not None
    assert good.status is Status.OK
    assert good.data["op"]["concurrency"] == 32
    at_top = run_rule("C03", _ctx(healthy_sweep()[:5]))
    assert at_top is not None
    assert at_top.status is Status.WARN
    impossible = run_rule("C03", _ctx((sweep_point(1, 100, 80.0, 90.0, seq=1),)))
    assert impossible is not None
    assert impossible.status is Status.BAD


def test_c04_peak_vs_capacity() -> None:
    f = run_rule("C04", _ctx(healthy_sweep()))
    assert f is not None
    assert f.data["peak"]["concurrency"] == 64
    assert 0 < f.data["gap"] < 0.1


def test_c05_tail_breaks_before_median() -> None:
    f = run_rule("C05", _ctx(healthy_sweep()))
    assert f is not None
    assert f.status is Status.WARN
    assert f.data["metric"] == "tpot"
    assert f.data["median"] < 50 < f.data["tail"]
    never = run_rule("C05", _ctx(healthy_sweep()[:5]))
    assert never is not None
    assert never.status is Status.OK


def test_w01_saturation() -> None:
    f = run_rule("W01", _ctx(healthy_sweep()))
    assert f is not None
    assert f.data["reached"] is True
    assert f.data["concurrency"] == 32
    growing = run_rule("W01", _ctx(healthy_sweep()[:4]))
    assert growing is not None
    assert growing.data["reached"] is False


def test_w02_slo_wall() -> None:
    f = run_rule("W02", _ctx(healthy_sweep()))
    assert f is not None
    assert f.status is Status.INFO
    assert f.data["metric"] == "tpot"
    assert 32 < f.data["at"] < 64
    assert run_rule("W02", _ctx(healthy_sweep()[:5])).status is Status.OK  # type: ignore[union-attr]


def test_w03_capacity_is_the_lowest_wall() -> None:
    f = run_rule("W03", _ctx(healthy_sweep()))
    assert f is not None
    assert f.data["limiter"] == "saturation"
    assert f.data["c_star"] == 32
    assert f.data["op"] == 32
    assert f.data["walls"]["memory"] == pytest.approx(532_768 / 2048)


def test_w03_memory_wall_binds_for_long_requests() -> None:
    small = CACHE_65GIB.__class__(**{**CACHE_65GIB.__dict__, "kv_cache_size_tokens": 40_000})
    f = run_rule("W03", _ctx(healthy_sweep(), theory=theory_for(cache=small)))
    assert f is not None
    assert f.data["limiter"] == "memory"


def test_w04_preemption() -> None:
    pts = list(healthy_sweep())
    pts[-1] = sweep_point(64, 3080, 20.0, 400.0, kv_max=1.0, preemptions=212, seq=7)
    f = run_rule("W04", _ctx(tuple(pts)))
    assert f is not None
    assert f.status is Status.BAD
    assert f.data["concurrency"] == 64
    assert f.data["preemptions"] == 212
    assert run_rule("W04", _ctx(healthy_sweep())).status is Status.OK  # type: ignore[union-attr]
    blind = run_rule("W04", _ctx(healthy_sweep(), metrics_available=False))
    assert blind is not None
    assert blind.status is Status.SKIP


def test_w05_late_preemption_for_long_outputs() -> None:
    # Reasoning shape on this cache is 1,024 in / 7,040 out: ~66 fit at the end, 520 at start.
    pts = (
        sweep_point(128, 4000, 20.0, 300.0, isl=1024, osl=7040, kv_max=1.0, preemptions=50, seq=1),
    )
    f = run_rule("W05", _ctx(pts, product=ProductType.REASONING))
    assert f is not None
    assert f.data["fits_end"] < 128 <= f.data["fits_admission"]
    assert run_rule("W05", _ctx(healthy_sweep())) is None


def test_w06_max_num_seqs_recommendation() -> None:
    f = run_rule("W06", _ctx(healthy_sweep()))
    assert f is not None
    assert f.status is Status.WARN
    assert f.data["current"] == 1024 and f.data["is_default"]
    assert f.data["recommend"] <= f.data["sustain"]
    preempted = list(healthy_sweep())
    preempted[-1] = sweep_point(64, 3080, 20.0, 400.0, preemptions=5, seq=7)
    assert run_rule("W06", _ctx(tuple(preempted))).status is Status.BAD  # type: ignore[union-attr]


def test_w06_ok_when_cap_is_already_low() -> None:
    c = ctx(points=(point(), *healthy_sweep()))
    capped = c.__class__(
        **{**c.__dict__, "setup": c.setup.__class__(**{**c.setup.__dict__, "max_num_seqs": 32})}
    )
    assert run_rule("W06", capped).status is Status.OK  # type: ignore[union-attr]


def test_w07_errors() -> None:
    assert run_rule("W07", _ctx(healthy_sweep())).status is Status.OK  # type: ignore[union-attr]
    pts = list(healthy_sweep())
    pts[-1] = sweep_point(64, 3000, 20.0, 80.0, errors=40, error_code=504, seq=7)
    f = run_rule("W07", _ctx(tuple(pts)))
    assert f is not None
    assert f.status is Status.BAD
    assert "504" in f.data["points"][0]["codes"]


def test_w08_headroom() -> None:
    f = run_rule("W08", _ctx(healthy_sweep()))
    assert f is not None
    assert f.data["run_at"] == 25


def test_w09_preemption_cost() -> None:
    pts = list(healthy_sweep())
    pts[-1] = sweep_point(64, 3080, 20.0, 400.0, preemptions=10, seq=7)
    f = run_rule("W09", _ctx(tuple(pts)))
    assert f is not None
    assert f.data["per_eviction_ms"] > 0
    assert f.data["evictions"] == 10
    assert run_rule("W09", _ctx(healthy_sweep())) is None


def test_h01_steady_state() -> None:
    assert run_rule("H01", _ctx(healthy_sweep())).status is Status.OK  # type: ignore[union-attr]
    short = run_rule("H01", _ctx((sweep_point(8, 1000, 6.0, rounds=2, seq=1),)))
    assert short is not None
    assert short.status is Status.WARN


def test_h02_weak_client() -> None:
    assert run_rule("H02", _ctx(healthy_sweep())).status is Status.OK  # type: ignore[union-attr]
    weak = run_rule("H02", _ctx((sweep_point(64, 3000, 20.0, effective=40.0, seq=1),)))
    assert weak is not None
    assert weak.status is Status.WARN


def test_h06_prefix_caching_caveat() -> None:
    f = run_rule("H06", _ctx(healthy_sweep()))
    assert f is not None
    assert f.status is Status.INFO
    off = CACHE_65GIB.__class__(**{**CACHE_65GIB.__dict__, "enable_prefix_caching": False})
    assert run_rule("H06", _ctx(theory=theory_for(cache=off))) is None


def test_h09_gateway_codes_held_requests_and_timeouts() -> None:
    assert run_rule("H09", _ctx(healthy_sweep())).status is Status.OK  # type: ignore[union-attr]
    limited = run_rule(
        "H09", _ctx((sweep_point(64, 3000, 20.0, errors=10, error_code=429, seq=1),))
    )
    assert limited is not None
    assert "429" in limited.data["codes"]
    held = run_rule("H09", _ctx((sweep_point(64, 3000, 20.0, running_max=16.0, seq=1),)))
    assert held is not None
    assert held.data["held"][0]["concurrency"] == 64

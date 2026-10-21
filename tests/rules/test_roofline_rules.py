"""F01-F02: the measured sweep on the roofline, and what other formats would change."""

from dataclasses import replace

import pytest

from kneebench.analysis.engine import Status
from tests.builders import ctx, healthy_sweep, point, run_rule, theory_for


def test_f01_places_every_level_below_the_ridge() -> None:
    f = run_rule("F01", ctx(points=(point(), *healthy_sweep())))
    assert f is not None and f.status is Status.INFO
    assert f.data["ridge"] == 835 / 3.9
    assert f.data["all_memory_bound"]
    rows = f.data["rows"]
    assert [r["concurrency"] for r in rows] == [1, 2, 4, 8, 16, 32, 64]
    assert all(r["intensity"] < f.data["ridge"] for r in rows)
    assert rows[0]["intensity"] < rows[-1]["intensity"]  # batching moves right
    assert all(0 < r["share"] <= 1 for r in rows)


def test_f02_offers_only_what_the_gpu_supports() -> None:
    h100 = run_rule("F02", ctx(points=(point(), *healthy_sweep())))
    assert h100 is not None and h100.status is Status.INFO
    assert [r["id"] for r in h100.data["rows"]] == ["served", "fp8_kv", "fp8", "w4a16"]
    a100 = ctx(theory=theory_for(gpu="a100-sxm-80"), points=(point(), *healthy_sweep()))
    f = run_rule("F02", replace(a100, theory=replace(a100.theory, gpu_id="a100-sxm-80")))
    assert f is not None
    assert [r["id"] for r in f.data["rows"]] == ["served", "fp8_kv", "int8", "w4a16"]
    fp8 = {r["id"]: r for r in h100.data["rows"]}["fp8"]
    assert fp8["fits"] > h100.data["rows"][0]["fits"]
    assert 1.5 < fp8["decode_speedup"] <= 2.0
    assert fp8["prefill_speedup"] == pytest.approx(2.0, abs=0.01)


def test_f02_warns_when_the_cache_is_the_limit() -> None:
    full = ctx(points=(point(), *healthy_sweep(kv_max=0.95)))
    f = run_rule("F02", full)
    assert f is not None and f.status is Status.WARN
    served_fp8_cache = replace(full, theory=replace(full.theory, cache_dtype="fp8"))
    again = run_rule("F02", served_fp8_cache)
    assert again is not None and again.status is Status.INFO
    assert "fp8_kv" not in [r["id"] for r in again.data["rows"]]

"""P01-P03: what prompt reuse gives real traffic."""

from dataclasses import replace
from typing import Any

from kneebench.analysis.context import PointView, RunContext
from kneebench.analysis.engine import Status
from tests.builders import ctx, healthy_sweep, point, run_rule


def _reuse(phase: str, ttft_s: float, hits: float, queries: float, **spec: Any) -> PointView:
    server = {
        "vllm:time_to_first_token_seconds": {"p50_estimate": ttft_s},
        "vllm:prefix_cache_hits": {"total": hits},
        "vllm:prefix_cache_queries": {"total": queries},
    }
    return replace(point(phase=phase, isl=100, concurrency=8, seq=50), server=server, spec=spec)


def _ctx(*pts: PointView, caching: bool = True) -> RunContext:
    c = ctx(points=(point(), *healthy_sweep(), *pts))
    return replace(c, theory=replace(c.theory, prefix_caching=caching))


CONTROL = _reuse("reuse_control", 0.120, 30, 1600)
SHARED = _reuse("reuse_shared", 0.045, 1400, 1600, prefix_pool=1, prefix_len=1500)


def test_p01_shared_prompt_saves_first_token_time() -> None:
    f = run_rule("P01", _ctx(CONTROL, SHARED))
    assert f is not None
    assert f.status is Status.OK
    assert f.data["saving"] == 1 - 45 / 120
    assert f.data["shared"]["hit_rate"] == 1400 / 1600
    off = run_rule("P01", _ctx(CONTROL, SHARED, caching=False))
    assert off is not None and off.status is Status.WARN and off.data["case"] == "off"
    low = run_rule("P01", _ctx(CONTROL, _reuse("reuse_shared", 0.1, 100, 1600, prefix_len=1500)))
    assert low is not None and low.data["case"] == "low"


def test_p02_turns_stay_flat_with_reuse() -> None:
    flat = {
        "turns": [
            {"turn": t, "requests": 8, "isl": 1100 + 400 * t, "ttft_ms": 60.0} for t in range(1, 7)
        ]
    }
    grow = {
        "turns": [
            {"turn": t, "requests": 8, "isl": 1100 + 400 * t, "ttft_ms": 40.0 * t}
            for t in range(1, 7)
        ]
    }
    base = _reuse("reuse_turns", 0.06, 3000, 3400, turns=6)
    ok = run_rule("P02", _ctx(replace(base, series={"turns": flat})))
    assert ok is not None and ok.status is Status.OK and ok.data["turns"] == 6
    bad = run_rule("P02", _ctx(replace(base, series={"turns": grow})))
    assert bad is not None and bad.status is Status.WARN


def test_p04_sorted_passages_reuse_more() -> None:
    ranked = _reuse("rag_ranked", 0.090, 900, 2400)
    ordered = _reuse("rag_sorted", 0.030, 2300, 2400)
    f = run_rule("P04", _ctx(ranked, ordered))
    assert f is not None and f.status is Status.INFO
    assert f.data["saving"] == 1 - 30 / 90
    assert f.data["sorted"]["hit_rate"] == 2300 / 2400
    assert run_rule("P04", _ctx(ranked, ordered, caching=False)) is None


def test_p03_counts_prompts_the_cache_holds_in_whole_blocks() -> None:
    c = _ctx(SHARED)
    c = replace(c, theory=replace(c.theory, pool_tokens=160_000, block_size=16))
    f = run_rule("P03", c)
    assert f is not None and f.status is Status.INFO
    # 10,000 blocks, minus 8 running requests of 23 blocks, over 94 blocks per prompt.
    assert f.data["held"] == (10_000 - 8 * 23) // 94
    assert run_rule("P03", _ctx(SHARED, caching=False)) is None

"""P rules: what prompt reuse and prefix caching give real traffic (notes 46-49)."""

from __future__ import annotations

import math
from typing import Any

from kneebench.analysis.context import PointView, RunContext
from kneebench.analysis.engine import Finding, info, ok, rule, warn
from kneebench.loadgen.texts import RAG_INSTRUCTIONS, RAG_PASSAGE_LEN, RAG_PASSAGES

# Note 47 section 4: an 80% hit rate cut first-token time by about a third, not 80%.
GOOD_HIT_RATE = 0.5
# Note 48 section 3: with reuse a later turn reads only its new message, so its first token
# is no slower than the first turn's; this much slower means the history is recomputed.
TURN_GROWTH_WARN = 2.0


def _point(ctx: RunContext, phase: str) -> PointView | None:
    pts = [p for p in ctx.points_in(phase) if p.status == "done"]
    return pts[0] if pts else None


def _facts(ctx: RunContext, p: PointView) -> dict[str, Any]:
    """First-token time from the engine when it reports it (the network can't blur it)."""
    engine = p.engine_ms("engine_ttft")
    client = p.server_ttft(ctx.ttft_offset_ms).get("p50")
    hits = p.server_stat("prefix_cache_hits", "total")
    queries = p.server_stat("prefix_cache_queries", "total")
    prompt = p.server_stat("prompt_tokens", "total")
    cached = p.server_stat("prompt_tokens_cached", "total")
    return {
        "ttft_ms": engine if engine is not None else client,
        "from_engine": engine is not None,
        "hit_rate": hits / queries if hits is not None and queries else None,
        "cached_share": cached / prompt if cached is not None and prompt else None,
        "concurrency": p.concurrency,
    }


@rule("P01", tags={"ttft", "prefill", "setup"}, source="47#4")
def shared_prompt(ctx: RunContext) -> Finding | None:
    control, shared = _point(ctx, "reuse_control"), _point(ctx, "reuse_shared")
    if control is None or shared is None:
        return None
    a, b = _facts(ctx, control), _facts(ctx, shared)
    if a["ttft_ms"] is None or b["ttft_ms"] is None:
        return None
    data = {
        "control": a,
        "shared": b,
        "saving": 1 - b["ttft_ms"] / a["ttft_ms"] if a["ttft_ms"] else None,
        "prefix_len": shared.spec.get("prefix_len"),
        "question_len": shared.isl,
        "caching_on": ctx.theory.prefix_caching,
    }
    if ctx.theory.prefix_caching is False:
        return warn("P01", case="off", **data)
    if b["hit_rate"] is not None and b["hit_rate"] < GOOD_HIT_RATE:
        return warn("P01", case="low", **data)
    return ok("P01", case="works", **data)


@rule("P02", tags={"ttft", "prefill"}, source="48#3")
def conversations(ctx: RunContext) -> Finding | None:
    p = _point(ctx, "reuse_turns")
    if p is None or not p.series.get("turns"):
        return None
    turns = p.series["turns"]["turns"]
    first, last = turns[0], turns[-1]
    offset = ctx.ttft_offset_ms
    rows = [{**t, "ttft_ms": max(0.0, t["ttft_ms"] - offset)} for t in turns]
    growth = rows[-1]["ttft_ms"] / rows[0]["ttft_ms"] if rows[0]["ttft_ms"] else None
    data = {
        "rows": rows,
        "turns": len(rows),
        "first_isl": first["isl"],
        "last_isl": last["isl"],
        "prompt_growth": last["isl"] / first["isl"] if first["isl"] else None,
        "ttft_growth": growth,
        "first_ttft_ms": rows[0]["ttft_ms"],
        "last_ttft_ms": rows[-1]["ttft_ms"],
        **_facts(ctx, p),
    }
    if growth is not None and growth >= TURN_GROWTH_WARN:
        return warn("P02", **data)
    return ok("P02", **data)


@rule("P04", tags={"ttft", "prefill"}, source="48#5")
def passage_order(ctx: RunContext) -> Finding | None:
    ranked, ordered = _point(ctx, "rag_ranked"), _point(ctx, "rag_sorted")
    if ranked is None or ordered is None or ctx.theory.prefix_caching is False:
        return None
    a, b = _facts(ctx, ranked), _facts(ctx, ordered)
    if a["hit_rate"] is None or b["hit_rate"] is None:
        return None
    return info(
        "P04",
        ranked=a,
        sorted=b,
        saving=1 - b["ttft_ms"] / a["ttft_ms"] if a["ttft_ms"] and b["ttft_ms"] else None,
        passages=RAG_PASSAGES,
        passage_len=RAG_PASSAGE_LEN,
        instructions=RAG_INSTRUCTIONS,
        concurrency=ordered.concurrency,
    )


@rule("P03", tags={"memory", "kv"}, source="49#6")
def cache_churn(ctx: RunContext) -> Finding | None:
    shared = _point(ctx, "reuse_shared")
    pool = ctx.theory.pool_tokens
    prefix_len = shared.spec.get("prefix_len") if shared else None
    if shared is None or not pool or not prefix_len or ctx.theory.prefix_caching is False:
        return None
    block = ctx.theory.block_size or 1
    blocks = pool // block
    running = (shared.concurrency or 1) * math.ceil((shared.isl + shared.osl) / block)
    held = max(0, blocks - running) // math.ceil(prefix_len / block)
    return info(
        "P03",
        held=held,
        prefix_len=prefix_len,
        pool_tokens=pool,
        concurrency=shared.concurrency,
    )

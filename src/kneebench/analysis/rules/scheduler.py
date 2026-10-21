"""E rules: what the engine itself reports about queueing and its admission limits (notes 30-35)."""

from __future__ import annotations

from kneebench.analysis.context import PointView, RunContext
from kneebench.analysis.engine import Finding, info, ok, rule
from kneebench.analysis.rules.walls import _effective_max_num_seqs

# A level queues inside the engine when requests wait and the slowest waits are noticeable
# (kneebench estimate: 50 ms is a tenth of a typical first-token target).
WAITING_MIN = 1.0
QUEUE_TAIL_MS = 50.0
# Note 35 section 3: a step close to the token budget means the budget is what limits it.
BUDGET_SHARE = 0.8
# Note 41 section 1: the cache is the limit once it is nearly full or evicting.
KV_FULL = 0.95
GATE_LABELS = {
    "max_num_seqs": "max_num_seqs",
    "kv_cache": "KV cache",
    "token_budget": "token budget",
    "unknown": "none full",
}


def _levels(ctx: RunContext) -> list[PointView]:
    pts = [p for p in ctx.points_in("concurrency") if p.concurrency and p.status == "done"]
    return sorted(pts, key=lambda p: p.concurrency or 0)


@rule("E01", tags={"ttft", "latency"}, source="30#6")
def where_first_token_time_goes(ctx: RunContext) -> Finding | None:
    """Client first-token time split into network and front end, engine queue, and prefill."""
    rows: list[dict[str, float | int | None]] = []
    for p in _levels(ctx):
        engine = p.engine_ms("engine_ttft")
        client = p.m("time_to_first_token", "p50")
        if engine is None or client is None:
            continue
        rows.append(
            {
                "concurrency": p.concurrency,
                "client_ms": client,
                "engine_ms": engine,
                "outside_ms": max(0.0, client - engine),
                "queue_ms": p.engine_ms("queue_time") or 0.0,
                "queue_tail_ms": p.engine_ms("queue_time", "p99"),
                "prefill_ms": p.engine_ms("prefill_time"),
            }
        )
    if not rows:
        return None
    queued = next(
        (r for r in rows if float(r["queue_ms"] or 0) > float(r["prefill_ms"] or 0)), None
    )
    return info(
        "E01",
        rows=rows,
        first=rows[0],
        queued_from=queued["concurrency"] if queued else None,
    )


@rule("E02", tags={"capacity", "setup"}, source="35#3")
def admission_limit(ctx: RunContext) -> Finding | None:
    """Which of the scheduler's three limits made requests wait (notes 31 and 35)."""
    levels = _levels(ctx)
    if not levels or not ctx.metrics_available:
        return None
    if all(p.engine_ms("queue_time", "p99") is None for p in levels):
        return None  # an older vLLM without the queue histogram: nothing to say either way
    seqs, seqs_default = _effective_max_num_seqs(ctx)
    budget = ctx.theory.default_max_num_batched_tokens
    rows = [_gate(p, seqs, budget) for p in levels if _queues(p)]
    if not rows:
        return ok("E02", max_tested=levels[-1].concurrency)
    known = next((r for r in rows if r["gate"] != "unknown"), None)
    # Note 35 section 3: a step holding N prompt tokens takes about N times the per-token prefill
    # cost, so the per-token target caps how many prompt tokens a step can safely carry.
    target = ctx.workload.slo.tpot_ms
    per_token_ms = _prefill_ms_per_token(ctx)
    safe = round(target / per_token_ms / 128) * 128 if target and per_token_ms else None
    return info(
        "E02",
        rows=rows,
        first=rows[0],
        known=known,
        gate=known["gate"] if known else "unknown",
        max_num_seqs=seqs,
        seqs_default=seqs_default,
        budget=budget,
        tpot_target_ms=target,
        safe_budget=safe,
    )


def _queues(p: PointView) -> bool:
    return (p.server_stat("waiting", "max") or 0) >= WAITING_MIN and (
        p.engine_ms("queue_time", "p99") or 0
    ) >= QUEUE_TAIL_MS


def _gate(p: PointView, seqs: int, budget: int) -> dict[str, object]:
    running = p.server_stat("running", "max") or 0
    kv = p.server_stat("kv_cache_usage", "max") or 0
    preempted = (p.server_stat("preemptions", "total") or 0) > 0
    step = p.server_stat("iteration_tokens", "p99_estimate")
    if running >= seqs:
        gate = "max_num_seqs"
    elif kv >= KV_FULL or preempted:
        gate = "kv_cache"
    elif step is not None and step >= BUDGET_SHARE * budget:
        gate = "token_budget"
    else:
        gate = "unknown"
    return {
        "concurrency": p.concurrency,
        "gate": gate,
        "gate_label": GATE_LABELS[gate],
        "waiting_max": p.server_stat("waiting", "max"),
        "running_max": running,
        "kv_max": kv,
        "step_tokens": step,
        "queue_tail_ms": p.engine_ms("queue_time", "p99"),
    }


def _prefill_ms_per_token(ctx: RunContext) -> float | None:
    from kneebench.analysis.shape import axis, prompt_fit

    fit = prompt_fit(axis(ctx, "shape_prompt", single=True))
    if fit is not None and fit[0] > 0:
        return fit[0]
    t = ctx.theory
    # Without the sweep: the math floor at half the peak (note 26 section 1).
    return (
        2 * t.model_params / (t.gpu_peak_tflops * 1e12 * 0.5) * 1000 if t.gpu_peak_tflops else None
    )

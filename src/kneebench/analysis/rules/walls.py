"""W rules: the operating envelope and its three walls (note 27)."""

from __future__ import annotations

import math

from kneebench.analysis import settings
from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, bad, info, ok, rule, skip, warn
from kneebench.analysis.rate import capacity_rps, rate_points, sustained_rps
from kneebench.analysis.rules.concurrency import sweep
from kneebench.analysis.sweep import (
    Crossing,
    SweepPoint,
    first_crossing,
    operating_point,
    saturation,
)
from kneebench.domain import Slo
from kneebench.phases.theory import ShapeTheory
from kneebench.physics.floors import requests_that_fit

# Notes 27 section 5 and 43 section 5: plan a margin below capacity for bursts.
BURST_HEADROOM = 0.8
# Note 43 section 6: replica counts to show the "survive one failure" budget for.
FAILURE_EXAMPLES = (2, 4)
# Note 27 section 1: errors and timeouts count as failures.
ERROR_WARN = 0.01
ERROR_BAD = 0.05
# A cap this far above what the cache sustains invites preemption (note 27 section 4).
SEQS_SLACK = 1.2
# Note 26 section 1 / 27 section 4: prefill at ~50% of peak for re-prefill cost.
PREFILL_EFFICIENCY = 0.5
# Note 41 section 4: one eviction can be harmless; a steady stream is thrashing. 1% of
# requests evicted is where kneebench calls it steady (kneebench estimate).
THRASHING_PER_1000 = 10.0
# Note 42 section 5: size admission for a high-percentile request, then keep a margin
# (its example sustains 187 and uses about 160).
ADMISSION_PERCENTILE = 95
ADMISSION_MARGIN = 0.85


def _shape(ctx: RunContext) -> ShapeTheory | None:
    return ctx.theory.shape(ctx.primary_shape) or ctx.theory.shape("baseline")


def final_length(ctx: RunContext, shape: ShapeTheory) -> tuple[int, bool]:
    """The final length (prompt + output) that 95 of 100 requests stay under, as served.

    Measured from the concurrency sweep when it ran, including the tokens the server adds
    for the chat template; otherwise the planned shape. The flag says which.
    """
    lengths = sorted(
        (r.isl or 0) + (r.osl or 0)
        for p in ctx.points_in("concurrency")
        for r in p.requests
        if r.ok and r.isl and r.osl
    )
    if len(lengths) < 20:
        return shape.isl + shape.osl, False
    rank = math.ceil(len(lengths) * ADMISSION_PERCENTILE / 100) - 1
    return lengths[rank] + round(ctx.prompt_overhead_tokens or 0), True


def _wall_crossings(pts: list[SweepPoint], slo: Slo) -> dict[str, Crossing]:
    """Where each latency target is crossed, at or above the operating point.

    A noisy tail can fail one low level while higher ones pass (note 25 section 6); the
    wall is where load stops fitting, so it is never below the best load that met every target.
    """
    op = operating_point(pts, slo)
    above = [p for p in pts if op is None or p.concurrency >= op.concurrency]
    found = {}
    for metric, target in (("tpot", slo.tpot_ms), ("ttft", slo.ttft_ms)):
        cross = first_crossing(above, metric, target, slo.percentile)
        if cross:
            found[metric] = cross
    return found


def _effective_max_num_seqs(ctx: RunContext) -> tuple[int, bool]:
    s = settings.max_num_seqs(ctx)
    return s.value, s.is_default


@rule("W01", tags={"capacity", "saturation", "throughput"}, source="27#2")
def saturation_wall(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    if len(pts) < 2:
        return None
    sat = saturation(pts)
    peak = max(pts, key=lambda p: p.output_tok_s or 0)
    if sat is None:
        return info(
            "W01", reached=False, max_tested=pts[-1].concurrency, peak_tok_s=peak.output_tok_s
        )
    after = [p for p in pts if p.concurrency > sat.concurrency]
    return info(
        "W01",
        reached=True,
        concurrency=sat.concurrency,
        tok_s=sat.output_tok_s,
        peak_tok_s=peak.output_tok_s,
        waiting_after=max((p.waiting_max or 0 for p in after), default=None),
        metrics_available=ctx.metrics_available,
    )


@rule("W02", tags={"capacity", "latency", "tpot", "ttft"}, source="27#3")
def slo_wall(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    slo = ctx.workload.slo
    if not pts or (slo.ttft_ms is None and slo.tpot_ms is None):
        return None
    crossings = list(_wall_crossings(pts, slo).values())
    data = {
        "pct_label": slo.percentile,
        "slo_ttft_ms": slo.ttft_ms,
        "slo_tpot_ms": slo.tpot_ms,
        "max_tested": pts[-1].concurrency,
        "ttft_offset_ms": ctx.ttft_offset_ms,
    }
    if not crossings:
        return ok("W02", **data)
    first = min(crossings, key=lambda c: c.concurrency)
    return info(
        "W02",
        metric=first.metric,
        at=first.concurrency,
        target=first.target_ms,
        last_pass=first.last_pass.concurrency if first.last_pass else None,
        first_fail=first.first_fail.concurrency,
        value=first.first_fail.latency(first.metric, slo.percentile),
        crossings=[{"metric": c.metric, "at": c.concurrency} for c in crossings],
        **data,
    )


def capacity(ctx: RunContext) -> dict[str, object] | None:
    """c* = the lowest wall: SLO crossings, saturation, memory and the sequence cap."""
    pts = sweep(ctx)
    if not pts:
        return None
    slo = ctx.workload.slo
    walls: dict[str, float] = {}
    for metric, cross in _wall_crossings(pts, slo).items():
        walls[f"slo_{metric}"] = cross.concurrency
    sat = saturation(pts)
    if sat:
        walls["saturation"] = float(sat.concurrency)
    shape = _shape(ctx)
    if shape is not None and shape.fits:
        walls["memory"] = shape.fits
    seqs, _ = _effective_max_num_seqs(ctx)
    walls["max_num_seqs"] = float(seqs)
    limiter = min(walls, key=lambda k: walls[k])
    op = operating_point(pts, slo)
    return {
        "walls": walls,
        "limiter": limiter,
        "c_star": walls[limiter],
        "op": op.concurrency if op else None,
        "op_tok_s": op.output_tok_s if op else None,
        "max_tested": pts[-1].concurrency,
        "untested_above": walls[limiter] > pts[-1].concurrency,
    }


@rule("W03", tags={"capacity", "latency"}, source="27#3")
def capacity_wall(ctx: RunContext) -> Finding | None:
    cap = capacity(ctx)
    if cap is None:
        return None
    slo = ctx.workload.slo
    return info(
        "W03",
        pct_label=slo.percentile,
        slo_ttft_ms=slo.ttft_ms,
        slo_tpot_ms=slo.tpot_ms,
        shape_name=ctx.primary_shape,
        **cap,
    )


@rule("W04", tags={"memory", "preemption", "kv"}, source="41#4")
def preemption(ctx: RunContext) -> Finding | None:
    """Evictions as a rate (note 41 section 4): an occasional one past the load you would run
    at is not thrashing; any at the operating point, or a steady stream, is."""
    pts = sweep(ctx)
    if not pts:
        return None
    if not ctx.metrics_available:
        return skip("W04", "no_metrics")
    hit = [p for p in pts if (p.preemptions or 0) > 0]
    kv_max = max((p.kv_usage_max or 0 for p in pts), default=None)
    if not hit:
        return ok("W04", kv_max=kv_max, max_tested=pts[-1].concurrency)
    first = hit[0]
    before = [p for p in pts if p.concurrency < first.concurrency]
    op = operating_point(pts, ctx.workload.slo)
    per_1000 = (first.preemptions or 0) / first.requests * 1000 if first.requests else None
    at_planned_load = op is not None and first.concurrency <= op.concurrency
    thrashing = at_planned_load or per_1000 is None or per_1000 >= THRASHING_PER_1000
    return (bad if thrashing else warn)(
        "W04",
        per_1000=per_1000,
        at_planned_load=at_planned_load,
        operating_point=op.concurrency if op else None,
        concurrency=first.concurrency,
        preemptions=first.preemptions,
        total=sum(p.preemptions or 0 for p in hit),
        kv_max=first.kv_usage_max,
        tpot_tail=first.tpot.get(f"p{ctx.workload.slo.percentile}"),
        tpot_tail_before=before[-1].tpot.get(f"p{ctx.workload.slo.percentile}") if before else None,
        pct_label=ctx.workload.slo.percentile,
        errors=first.error_rate,
        shape_name=ctx.primary_shape,
    )


@rule("W05", tags={"memory", "preemption", "shape"}, source="26#4")
def late_preemption(ctx: RunContext) -> Finding | None:
    shape = _shape(ctx)
    pool = ctx.theory.pool_tokens
    if shape is None or not pool or shape.osl < 2 * shape.isl:
        return None
    hit = [p for p in sweep(ctx) if (p.preemptions or 0) > 0]
    if not hit:
        return None
    c = hit[0].concurrency
    block = ctx.theory.block_size
    at_admission = requests_that_fit(pool, shape.isl, 0, block)
    at_end = requests_that_fit(pool, shape.isl, shape.osl, block)
    if not at_end < c <= at_admission:
        return None
    # Tokens each request may hold before the pool is full, and when that happens.
    full_at = pool / c - shape.isl
    tpot = hit[0].tpot.get("p50")
    return info(
        "W05",
        concurrency=c,
        isl=shape.isl,
        osl=shape.osl,
        fits_admission=at_admission,
        fits_end=at_end,
        start_share=c * shape.isl / pool,
        full_after_tokens=full_at,
        full_after_s=full_at * tpot / 1000 if tpot else None,
    )


@rule("W06", tags={"memory", "capacity", "setup"}, source="27#4")
def max_num_seqs(ctx: RunContext) -> Finding | None:
    shape = _shape(ctx)
    pool = ctx.theory.pool_tokens
    if shape is None or not pool:
        return None
    seqs, is_default = _effective_max_num_seqs(ctx)
    length, measured = final_length(ctx, shape)
    sustain = math.floor(requests_that_fit(pool, length, 0, ctx.theory.block_size))
    admit = max(1, math.floor(sustain * ADMISSION_MARGIN))
    cap = capacity(ctx)
    slo_c = None
    if cap:
        walls = cap["walls"]
        assert isinstance(walls, dict)
        # Only the per-token wall: capping sequences keeps admitted users fast, but it makes a
        # first-token problem worse, since the extra requests wait in the queue instead.
        slo_c = walls.get("slo_tpot")
    recommend = max(1, min(admit, math.floor(slo_c) if slo_c else admit))
    preempted = any((p.preemptions or 0) > 0 for p in sweep(ctx))
    data = {
        "current": seqs,
        "is_default": is_default,
        "seqs_source": settings.max_num_seqs(ctx).source,
        "sustain": sustain,
        "admit": admit,
        "margin": 1 - ADMISSION_MARGIN,
        "recommend": recommend,
        "measured_length": measured,
        "percentile": ADMISSION_PERCENTILE,
        "block_size": ctx.theory.block_size,
        "slo_c": slo_c,
        "preempted": preempted,
        "shape_name": ctx.primary_shape,
        "final_len": length,
        "pool_tokens": pool,
        "vllm_version": ctx.vllm_version,
    }
    if seqs <= recommend * SEQS_SLACK:
        return ok("W06", **data)
    return bad("W06", **data) if preempted else warn("W06", **data)


@rule("W07", tags={"reliability", "capacity"}, source="27#1")
def errors(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    if not pts:
        return None
    over = [p for p in pts if p.error_rate > ERROR_WARN]
    if not over:
        return ok("W07", max_tested=pts[-1].concurrency)
    failing = [
        {
            "concurrency": p.concurrency,
            "rate": p.error_rate,
            "codes": ", ".join(f"{k} x{v}" for k, v in sorted(p.error_codes.items())) or "none",
        }
        for p in over
    ]
    worst = max(p.error_rate for p in over)
    verdict = bad if worst > ERROR_BAD else warn
    return verdict("W07", points=failing, worst=worst, first=over[0].concurrency)


@rule("W08", tags={"capacity"}, source="43#5")
def headroom(ctx: RunContext) -> Finding | None:
    """Plan on the request rate that meets the targets, less a burst and a failure margin.

    Note 43: the margin goes on open-loop capacity (requests per second); users in flight is
    the fallback when the rate sweep didn't run.
    """
    cap = capacity(ctx)
    if cap is None:
        return None
    c_star = cap["c_star"]
    assert isinstance(c_star, float)
    offset = ctx.ttft_offset_ms
    rps = sustained_rps(
        rate_points(ctx.points_in("rate"), offset),
        capacity_rps(ctx.points_in("concurrency"), offset),
        ctx.workload.slo,
    )
    planned = rps * BURST_HEADROOM if rps else None
    return ok(
        "W08",
        c_star=c_star,
        run_at=max(1, math.floor(c_star * BURST_HEADROOM)),
        limiter=cap["limiter"],
        headroom=1 - BURST_HEADROOM,
        slo_rps=rps,
        planned_rps=planned,
        failure=[
            {"replicas": n, "per_replica_rps": planned * (n - 1) / n} for n in FAILURE_EXAMPLES
        ]
        if planned
        else [],
    )


@rule("W09", tags={"memory", "preemption"}, source="27#4")
def preemption_cost(ctx: RunContext) -> Finding | None:
    shape = _shape(ctx)
    hit = [p for p in sweep(ctx) if (p.preemptions or 0) > 0]
    if shape is None or not hit:
        return None
    t = ctx.theory
    per_token_s = 2 * t.model_params / (PREFILL_EFFICIENCY * t.gpu_peak_tflops * 1e12 * t.gpu_count)
    tokens = shape.isl + shape.osl / 2  # on average an eviction loses half the answer
    per_eviction_ms = tokens * per_token_s * 1000
    total = sum(p.preemptions or 0 for p in hit)
    return info(
        "W09",
        per_eviction_ms=per_eviction_ms,
        tokens=tokens,
        evictions=total,
        total_s=per_eviction_ms * total / 1000,
    )

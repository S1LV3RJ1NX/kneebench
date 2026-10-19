"""C rules: the latency-throughput curve from the concurrency sweep (note 25)."""

from __future__ import annotations

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, bad, info, ok, rule, skip, warn
from kneebench.analysis.sweep import (
    FLAT_EFFICIENCY,
    SATURATION_GAIN,
    SweepPoint,
    curve,
    meets,
    operating_point,
    steps,
)


def sweep(ctx: RunContext) -> list[SweepPoint]:
    return curve(ctx.points_in("concurrency"), ctx.http_rtt_ms or 0.0)


def _row(p: SweepPoint, pct: int) -> dict[str, object]:
    return {
        "concurrency": p.concurrency,
        "tok_s": p.output_tok_s,
        "req_s": p.requests_s,
        "tpot_p50": p.tpot.get("p50"),
        "tpot_tail": p.tpot.get(f"p{pct}"),
        "ttft_p50": p.ttft.get("p50"),
        "ttft_tail": p.ttft.get(f"p{pct}"),
        "errors": p.error_rate,
        "kv_max": p.kv_usage_max,
        "preemptions": p.preemptions,
        "waiting_max": p.waiting_max,
    }


def _slo_data(ctx: RunContext) -> dict[str, object]:
    slo = ctx.workload.slo
    return {
        "pct_label": slo.percentile,
        "slo_ttft_ms": slo.ttft_ms,
        "slo_tpot_ms": slo.tpot_ms,
        "shape_name": ctx.primary_shape,
        "ttft_offset_ms": ctx.http_rtt_ms or 0.0,
    }


@rule("C01", tags={"capacity", "throughput"}, source="25#1")
def free_lunch(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    if len(pts) < 2 or not pts[0].output_tok_s or pts[0].concurrency != 1:
        return None
    target = operating_point(pts, ctx.workload.slo) or max(pts, key=lambda p: p.output_tok_s or 0)
    if target.concurrency == 1 or not target.output_tok_s:
        return None
    one = pts[0]
    base_tok_s = one.output_tok_s
    if not base_tok_s:
        return None
    t1, tn = one.tpot.get("p50"), target.tpot.get("p50")
    return info(
        "C01",
        concurrency=target.concurrency,
        multiple=target.output_tok_s / base_tok_s,
        tpot_increase=(tn / t1 - 1) if t1 and tn else None,
        tok_s_one=one.output_tok_s,
        tok_s=target.output_tok_s,
        tpot_one=t1,
        tpot=tn,
    )


@rule("C02", tags={"capacity", "throughput", "tpot"}, source="25#3")
def regions(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    if len(pts) < 3:
        return None
    pct = ctx.workload.slo.percentile
    by_c = {p.concurrency: p for p in pts}
    rows = []
    for s in steps(pts):
        region = (
            "flat"
            if s.efficiency >= FLAT_EFFICIENCY
            else "knee"
            if s.gain >= SATURATION_GAIN
            else "wall"
        )
        rows.append(
            {
                "c_from": s.c_from,
                "c_to": s.c_to,
                "gain": s.gain,
                "efficiency": s.efficiency,
                "region": region,
                "tpot_tail": by_c[s.c_to].tpot.get(f"p{pct}"),
            }
        )
    knee = next((r["c_from"] for r in rows if r["region"] != "flat"), None)
    return info("C02", steps=rows, knee=knee, points=[_row(p, pct) for p in pts], **_slo_data(ctx))


@rule("C03", tags={"capacity", "latency", "tpot", "ttft"}, source="25#3")
def operating(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    if not pts:
        return None
    slo = ctx.workload.slo
    if slo.ttft_ms is None and slo.tpot_ms is None:
        return skip("C03", "no_slo")
    op = operating_point(pts, slo)
    data = {**_slo_data(ctx), "max_tested": pts[-1].concurrency}
    if op is None:
        return bad("C03", first=_row(pts[0], slo.percentile), **data)
    row = _row(op, slo.percentile)
    if op is pts[-1]:
        return warn("C03", op=row, at_top=True, **data)
    return ok("C03", op=row, at_top=False, **data)


@rule("C04", tags={"capacity", "throughput"}, source="25#6")
def peak_is_not_capacity(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    op = operating_point(pts, ctx.workload.slo) if pts else None
    if op is None or not op.output_tok_s:
        return None
    peak = max(pts, key=lambda p: p.output_tok_s or 0)
    if peak.concurrency <= op.concurrency or not peak.output_tok_s:
        return None
    pct = ctx.workload.slo.percentile
    return info(
        "C04",
        peak=_row(peak, pct),
        op=_row(op, pct),
        gap=1 - op.output_tok_s / peak.output_tok_s,
        **_slo_data(ctx),
    )


@rule("C05", tags={"latency", "tpot", "ttft"}, source="25#6")
def tails_not_medians(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    slo = ctx.workload.slo
    if not pts or slo.percentile == 50:
        return None
    for p in pts:
        if meets(p, slo):
            continue
        for metric, target in (("tpot", slo.tpot_ms), ("ttft", slo.ttft_ms)):
            median = p.latency(metric, 50)
            tail = p.latency(metric, slo.percentile)
            if target and median is not None and tail is not None and median <= target < tail:
                return warn(
                    "C05",
                    concurrency=p.concurrency,
                    metric=metric,
                    median=median,
                    tail=tail,
                    target=target,
                    pct_label=slo.percentile,
                )
        return ok("C05", concurrency=p.concurrency, pct_label=slo.percentile)
    return ok("C05", concurrency=None, pct_label=slo.percentile)

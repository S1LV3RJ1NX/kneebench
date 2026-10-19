"""R rules: random arrivals (note 25 section 4)."""

from __future__ import annotations

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, info, ok, rule, warn
from kneebench.analysis.rate import RatePoint, capacity_rps, rate_points, same_rate

# Note 25 section 4: near capacity, TTFT bends up sharply as random arrivals bunch up.
KNEE_FACTOR = 2.0
# Note 25 section 6: compare achieved with requested rate to catch a weak client.
ISSUE_MIN_SHARE = 0.95
# A burst penalty this large means average-rate planning will disappoint.
BURST_PENALTY_WARN = 2.0


def _points(ctx: RunContext) -> tuple[list[RatePoint], float | None]:
    offset = ctx.http_rtt_ms or 0.0
    pts = rate_points(ctx.points_in("rate"), offset)
    return pts, capacity_rps(ctx.points_in("concurrency"), offset)


def _row(p: RatePoint, pct: int, mu: float | None) -> dict[str, object]:
    return {
        "requested": p.requested,
        "factor": p.requested / mu if mu else None,
        "issued": p.issued,
        "completed": p.completed,
        "arrival": p.arrival,
        "ttft_p50": p.ttft.get("p50"),
        "ttft_tail": p.ttft.get(f"p{pct}"),
        "tpot_tail": p.tpot.get(f"p{pct}"),
        "growth": p.growth,
        "past_capacity": p.past_capacity,
        "errors": p.error_rate,
        "trend": [list(t) for t in p.trend],
    }


@rule("R01", tags={"capacity", "ttft", "latency"}, source="25#4")
def rate_capacity(ctx: RunContext) -> Finding | None:
    pts, mu = _points(ctx)
    poisson = [p for p in pts if p.arrival == "poisson"]
    if not poisson:
        return None
    slo = ctx.workload.slo
    pct = slo.percentile
    lowest = poisson[0].ttft.get(f"p{pct}") or 0.0
    knee = next(
        (
            p.requested
            for p in poisson
            if lowest and (p.ttft.get(f"p{pct}") or 0) > KNEE_FACTOR * lowest
        ),
        None,
    )
    # Note 25 section 4: above capacity the queue grows without limit, however short runs look.
    passing = [p for p in poisson if p.meets(slo) and (mu is None or p.requested <= mu)]
    best = passing[-1].requested if passing else None
    data = {
        "rows": [_row(p, pct, mu) for p in poisson],
        "capacity_rps": mu,
        "knee_rps": knee,
        "max_rps": best,
        "pct_label": pct,
        "slo_ttft_ms": slo.ttft_ms,
        "slo_tpot_ms": slo.tpot_ms,
        "ttft_offset_ms": ctx.http_rtt_ms or 0.0,
        "shape_name": ctx.primary_shape,
    }
    return info("R01", **data) if best is not None else warn("R01", **data)


@rule("R02", tags={"capacity", "ttft", "reliability"}, source="25 derivation 4")
def run_length(ctx: RunContext) -> Finding | None:
    pts, mu = _points(ctx)
    if not pts:
        return None
    growing = [p for p in pts if p.past_capacity]
    rows = [_row(p, ctx.workload.slo.percentile, mu) for p in pts]
    if not growing:
        return ok("R02", rows=rows, capacity_rps=mu)
    first = min(growing, key=lambda p: p.requested)
    trend = first.trend
    data = {
        "rows": rows,
        "capacity_rps": mu,
        "rate": first.requested,
        "factor": first.requested / mu if mu else None,
        "arrival": first.arrival,
        "start_ms": trend[0][1],
        "end_ms": trend[-1][1],
        "seconds": trend[-1][0],
    }
    # Growing queues below the closed-loop capacity mean random arrivals find the limit sooner.
    early = mu is not None and first.requested < 0.9 * mu
    return warn("R02", early=early, **data) if early else info("R02", early=False, **data)


@rule("R03", tags={"reliability", "network"}, source="25#6")
def client_kept_up(ctx: RunContext) -> Finding | None:
    pts, _ = _points(ctx)
    if not pts:
        return None
    short = [
        {"requested": p.requested, "issued": p.issued, "arrival": p.arrival}
        for p in pts
        if p.issued is not None and p.issued < ISSUE_MIN_SHARE * p.requested
    ]
    if short:
        return warn("R03", points=short)
    return ok("R03", checked=len(pts))


@rule("R04", tags={"capacity", "ttft", "latency"}, source="25#5")
def bursts(ctx: RunContext) -> Finding | None:
    pts, mu = _points(ctx)
    slo = ctx.workload.slo
    pct = slo.percentile
    pairs = []
    for b in (p for p in pts if p.arrival == "gamma"):
        match = next(
            (p for p in pts if p.arrival == "poisson" and same_rate(p.requested, b.requested)), None
        )
        if match is None:
            continue
        smooth, bursty = match.ttft.get(f"p{pct}"), b.ttft.get(f"p{pct}")
        pairs.append(
            {
                "rate": b.requested,
                "factor": b.requested / mu if mu else None,
                "smooth_ms": smooth,
                "bursty_ms": bursty,
                "penalty": bursty / smooth if smooth and bursty else None,
                "smooth_meets": match.meets(slo),
                "bursty_meets": b.meets(slo),
            }
        )
    if not pairs:
        return None
    worst = max(pairs, key=lambda x: x["penalty"] or 0)
    breaks = any(x["smooth_meets"] and not x["bursty_meets"] for x in pairs)
    data = {
        "pairs": pairs,
        "worst": worst,
        "pct_label": pct,
        "breaks": breaks,
        "burstiness": next((p.burstiness for p in pts if p.arrival == "gamma"), None),
    }
    penalty = worst["penalty"]
    if breaks or (isinstance(penalty, float) and penalty >= BURST_PENALTY_WARN):
        return warn("R04", **data)
    return info("R04", **data)

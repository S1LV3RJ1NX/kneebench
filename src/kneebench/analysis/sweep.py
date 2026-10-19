"""Concurrency sweep analysis (notes 25 and 27): curve, knee, operating point and walls."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

from kneebench.analysis.context import PointView
from kneebench.analysis.steady import steady_state
from kneebench.domain import Slo

# Note 27 section 2: once doubling concurrency adds less than ~5%, you've saturated.
SATURATION_GAIN = 0.05
# Marginal efficiency per doubling (kneebench estimate); at or above this it is "flat".
FLAT_EFFICIENCY = 0.8
GATEWAY_CODES = {408, 429, 502, 503, 504}


@dataclass(frozen=True)
class SweepPoint:
    concurrency: int
    point_id: int
    output_tok_s: float | None  # steady-state
    requests_s: float | None
    steady_reliable: bool
    steady_share: float | None
    ttft: dict[str, float]  # p50, p90, p99 in ms
    tpot: dict[str, float]
    error_rate: float
    gateway_errors: int
    error_codes: dict[int, int]
    kv_usage_max: float | None
    preemptions: float | None
    waiting_max: float | None
    running_max: float | None
    requests: int

    def latency(self, metric: str, percentile: int) -> float | None:
        values = self.ttft if metric == "ttft" else self.tpot
        return values.get(f"p{percentile}")


def _pcts(point: PointView, metric: str, offset_ms: float = 0.0) -> dict[str, float]:
    stats = point.metrics.get(metric, {})
    return {
        k: max(v - offset_ms, 0.0) for k, v in stats.items() if k in ("p50", "p90", "p99", "avg")
    }


def _error_rate(point: PointView, errors: int, total: int) -> float:
    if total:
        return errors / total
    return (point.m("request_error_rate", "avg") or 0.0) / 100  # AIPerf reports a percentage


def summarize(point: PointView, ttft_offset_ms: float = 0.0) -> SweepPoint:
    """One concurrency level. TTFT has the client's network round trip removed, because
    that time doesn't change with load and users elsewhere add their own instead."""
    steady = steady_state(point.requests, point.concurrency) if point.requests else None
    total = len(point.requests)
    errors = [r for r in point.requests if not r.ok]
    codes: dict[int, int] = {}
    for r in errors:
        if r.error_code is not None:
            codes[r.error_code] = codes.get(r.error_code, 0) + 1
    return SweepPoint(
        concurrency=point.concurrency or 0,
        point_id=point.id,
        output_tok_s=steady.output_tok_s if steady else point.m("output_token_throughput", "avg"),
        requests_s=steady.requests_s if steady else point.m("request_throughput", "avg"),
        steady_reliable=bool(steady and steady.reliable),
        steady_share=steady.share if steady else None,
        ttft=point.server_ttft(ttft_offset_ms),
        tpot=_pcts(point, "inter_token_latency"),
        error_rate=_error_rate(point, len(errors), total),
        gateway_errors=sum(n for c, n in codes.items() if c in GATEWAY_CODES),
        error_codes=codes,
        kv_usage_max=point.server_stat("kv_cache_usage", "max"),
        preemptions=point.server_stat("preemptions", "total"),
        waiting_max=point.server_stat("waiting", "max"),
        running_max=point.server_stat("running", "max"),
        requests=total,
    )


def curve(points: Sequence[PointView], ttft_offset_ms: float = 0.0) -> list[SweepPoint]:
    return sorted(
        (summarize(p, ttft_offset_ms) for p in points if p.concurrency),
        key=lambda s: s.concurrency,
    )


@dataclass(frozen=True)
class Step:
    c_from: int
    c_to: int
    gain: float  # throughput ratio minus 1
    efficiency: float  # gain relative to ideal for this step (kneebench estimate)


def steps(points: Sequence[SweepPoint]) -> list[Step]:
    out: list[Step] = []
    for a, b in itertools.pairwise(points):
        if not a.output_tok_s or b.output_tok_s is None:
            continue
        ratio = b.output_tok_s / a.output_tok_s
        ideal = b.concurrency / a.concurrency
        out.append(Step(a.concurrency, b.concurrency, ratio - 1, (ratio - 1) / (ideal - 1)))
    return out


def saturation(points: Sequence[SweepPoint]) -> SweepPoint | None:
    """The first point whose doubling gained less than 5%: extra users only wait (note 27)."""
    by_c = {p.concurrency: p for p in points}
    for s in steps(points):
        if s.gain < SATURATION_GAIN * (math.log2(s.c_to / s.c_from)):
            return by_c[s.c_from]
    return None


def meets(point: SweepPoint, slo: Slo) -> bool:
    if point.error_rate > 0:
        return False  # errors count as failures (note 27 section 1)
    ttft = point.latency("ttft", slo.percentile)
    tpot = point.latency("tpot", slo.percentile)
    if slo.ttft_ms is not None and (ttft is None or ttft > slo.ttft_ms):
        return False
    return not (slo.tpot_ms is not None and (tpot is None or tpot > slo.tpot_ms))


def operating_point(points: Sequence[SweepPoint], slo: Slo) -> SweepPoint | None:
    """The rightmost point still under every SLO line (note 25 section 3)."""
    passing = [p for p in points if meets(p, slo)]
    return passing[-1] if passing else None


@dataclass(frozen=True)
class Crossing:
    metric: str  # ttft or tpot
    target_ms: float
    last_pass: SweepPoint | None
    first_fail: SweepPoint
    concurrency: float  # interpolated on a log2 concurrency axis


def first_crossing(
    points: Sequence[SweepPoint], metric: str, target_ms: float | None, percentile: int
) -> Crossing | None:
    if target_ms is None:
        return None
    prev: SweepPoint | None = None
    for p in points:
        value = p.latency(metric, percentile)
        if value is not None and value > target_ms:
            if prev is None:
                return Crossing(metric, target_ms, None, p, float(p.concurrency))
            v0 = prev.latency(metric, percentile) or 0.0
            x0, x1 = math.log2(prev.concurrency), math.log2(p.concurrency)
            frac = (target_ms - v0) / (value - v0) if value != v0 else 0.0
            return Crossing(metric, target_ms, prev, p, 2 ** (x0 + frac * (x1 - x0)))
        prev = p
    return None

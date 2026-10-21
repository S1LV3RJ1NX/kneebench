"""Request-rate sweep analysis (note 25 section 4): random arrivals near capacity.

The concurrency sweep only sends a new request when one finishes, so it slows down with
the server. Real users arrive whenever they like. Past capacity, the queue grows for as
long as the test runs, so first-token time depends on run length.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from kneebench.analysis.context import PointView, RequestView
from kneebench.analysis.sweep import curve
from kneebench.domain import Slo

NS = 1e9
TREND_BIN_S = 10.0
TREND_MIN_REQUESTS = 3
# Note 25 section 4: past capacity, the newest request waits for everyone ahead of it.
GROWTH_PAST_CAPACITY = 2.0
GROWTH_MIN_INCREASE_MS = 500.0
# Two rates within this share of each other count as the same load level.
SAME_RATE = 0.03


@dataclass(frozen=True)
class RatePoint:
    point_id: int
    arrival: str
    burstiness: float | None
    requested: float
    issued: float | None
    completed: float | None
    ttft: dict[str, float]
    tpot: dict[str, float]
    error_rate: float
    trend: tuple[tuple[float, float], ...] = field(default=())  # (seconds, median TTFT ms)

    @property
    def growth(self) -> float | None:
        if len(self.trend) < 3:
            return None
        first = statistics.median(v for _, v in self.trend[:2])
        last = statistics.median(v for _, v in self.trend[-2:])
        return last / first if first > 0 else None

    @property
    def past_capacity(self) -> bool:
        g = self.growth
        if g is None or len(self.trend) < 3:
            return False
        increase = self.trend[-1][1] - self.trend[0][1]
        return g >= GROWTH_PAST_CAPACITY and increase >= GROWTH_MIN_INCREASE_MS

    def latency(self, metric: str, percentile: int) -> float | None:
        return (self.ttft if metric == "ttft" else self.tpot).get(f"p{percentile}")

    def meets(self, slo: Slo) -> bool:
        if self.error_rate > 0 or self.past_capacity:
            return False
        ttft, tpot = self.latency("ttft", slo.percentile), self.latency("tpot", slo.percentile)
        if slo.ttft_ms is not None and (ttft is None or ttft > slo.ttft_ms):
            return False
        return not (slo.tpot_ms is not None and (tpot is None or tpot > slo.tpot_ms))


def _rate(times: Sequence[int], span_ns: int) -> float | None:
    return len(times) / (span_ns / NS) if span_ns > 0 else None


def _trend(requests: Sequence[RequestView], offset_ms: float) -> tuple[tuple[float, float], ...]:
    timed = [r for r in requests if r.ok and r.ttft_ms is not None]
    if not timed:
        return ()
    t0 = min(r.start_ns for r in timed)
    bins: dict[int, list[float]] = {}
    for r in timed:
        assert r.ttft_ms is not None
        bins.setdefault(int((r.start_ns - t0) / NS // TREND_BIN_S), []).append(
            max(r.ttft_ms - (r.connect_ms or 0.0) - offset_ms, 0.0)
        )
    return tuple(
        ((i + 0.5) * TREND_BIN_S, statistics.median(v))
        for i, v in sorted(bins.items())
        if len(v) >= TREND_MIN_REQUESTS
    )


def summarize_rate(point: PointView, ttft_offset_ms: float = 0.0) -> RatePoint:
    reqs = point.requests
    starts = sorted(r.start_ns for r in reqs)
    span = (starts[-1] - starts[0]) if len(starts) > 1 else 0
    ends = sorted(r.end_ns for r in reqs if r.ok)
    tpot = {
        k: v
        for k, v in point.metrics.get("inter_token_latency", {}).items()
        if k in ("p50", "p90", "p99", "avg")
    }
    errors = sum(1 for r in reqs if not r.ok)
    return RatePoint(
        point_id=point.id,
        arrival=point.arrival or "poisson",
        burstiness=point.burstiness,
        requested=point.request_rate or 0.0,
        issued=_rate(starts, span),
        completed=_rate(ends, (ends[-1] - starts[0]) if ends and starts else 0),
        ttft=point.server_ttft(ttft_offset_ms),
        tpot=tpot,
        error_rate=errors / len(reqs) if reqs else 0.0,
        trend=_trend(reqs, ttft_offset_ms),
    )


def capacity_rps(points: Sequence[PointView], ttft_offset_ms: float = 0.0) -> float | None:
    """Peak steady-state requests per second from the concurrency sweep."""
    rates = [p.requests_s for p in curve(points, ttft_offset_ms) if p.requests_s]
    return max(rates) if rates else None


def rate_points(points: Sequence[PointView], ttft_offset_ms: float = 0.0) -> list[RatePoint]:
    return sorted(
        (summarize_rate(p, ttft_offset_ms) for p in points if p.request_rate),
        key=lambda r: (r.arrival != "poisson", r.requested),
    )


def same_rate(a: float, b: float) -> bool:
    return abs(a - b) <= SAME_RATE * max(a, b)


def sustained_rps(points: Sequence[RatePoint], capacity: float | None, slo: Slo) -> float | None:
    """The highest random (Poisson) arrival rate that met the targets.

    Note 25 section 4: above capacity the queue grows without limit, however short runs look,
    so rates above it never count.
    """
    passing = [
        p.requested
        for p in points
        if p.arrival == "poisson" and p.meets(slo) and (capacity is None or p.requested <= capacity)
    ]
    return max(passing) if passing else None

"""Steady-state throughput: only the window where the load was really at its target.

At the start of a point, requests are still being added (ramp-up); at the end, no new
ones are issued and the in-flight count falls (drain). Counting either makes throughput
look lower than what the server sustains, most of all at high concurrency (note 25:
"too few requests" means you measure the ramp-up). Tokens are spread evenly over each
request's decode time to attribute them to the window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from kneebench.analysis.context import RequestView

NS = 1e9
# A window shorter than this, or than this share of the run, is not a steady state.
MIN_WINDOW_S = 20.0
MIN_WINDOW_SHARE = 0.4
# Concurrency mode reaches its target when in-flight requests reach this share of it.
RAMP_TARGET_SHARE = 0.95
# Rate mode has no in-flight target; skip this share of the issue period as ramp-up.
RATE_RAMP_SHARE = 0.1


@dataclass(frozen=True)
class SteadyState:
    start_ns: int
    end_ns: int
    output_tok_s: float
    requests_s: float
    completed: int
    run_s: float

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / NS

    @property
    def share(self) -> float:
        return self.duration_s / self.run_s if self.run_s > 0 else 0.0

    @property
    def reliable(self) -> bool:
        return self.duration_s >= MIN_WINDOW_S and self.share >= MIN_WINDOW_SHARE


def _ramp_end(requests: Sequence[RequestView], concurrency: int | None) -> int:
    starts = sorted(r.start_ns for r in requests)
    if not concurrency:
        span = starts[-1] - starts[0]
        return starts[0] + int(span * RATE_RAMP_SHARE)
    target = max(1, math.ceil(concurrency * RAMP_TARGET_SHARE))
    events = sorted([(r.start_ns, 1) for r in requests] + [(r.end_ns, -1) for r in requests])
    in_flight = 0
    for t, delta in events:
        in_flight += delta
        if in_flight >= target:
            return t
    return starts[-1]  # never reached: the window collapses and is flagged unreliable


def _tokens_in_window(r: RequestView, start: int, end: int) -> float:
    if not r.osl or r.ttft_ms is None:
        return 0.0
    first = r.start_ns + int(r.ttft_ms * 1e6)
    tokens = 0.0
    if start <= first <= end:
        tokens += 1.0
    remaining = r.osl - 1
    if remaining <= 0 or r.end_ns <= first:
        return tokens
    overlap = min(end, r.end_ns) - max(start, first)
    if overlap > 0:
        tokens += remaining * overlap / (r.end_ns - first)
    return tokens


def steady_state(requests: Sequence[RequestView], concurrency: int | None) -> SteadyState | None:
    ok = [r for r in requests if r.ok]
    if len(ok) < 2:
        return None
    first_start = min(r.start_ns for r in requests)
    last_end = max(r.end_ns for r in requests)
    start = _ramp_end(ok, concurrency)
    end = max(r.start_ns for r in requests)  # after the last request is issued, load drains
    run_s = (last_end - first_start) / NS
    if end <= start:
        return SteadyState(start, start, 0.0, 0.0, 0, run_s)
    tokens = sum(_tokens_in_window(r, start, end) for r in ok)
    completed = sum(1 for r in ok if start <= r.end_ns <= end)
    window_s = (end - start) / NS
    return SteadyState(start, end, tokens / window_s, completed / window_s, completed, run_s)

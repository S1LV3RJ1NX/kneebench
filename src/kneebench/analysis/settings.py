"""Scheduler settings the server doesn't publish, read from how the engine behaved (note 35).

vLLM reports neither ``max_num_seqs`` nor ``max_num_batched_tokens`` on /metrics. Each comes
from the user when they said so, otherwise from the run when the engine's behaviour pins it
down, otherwise from the defaults for the vLLM version and GPU. The source is kept, so text
can say "inferred" or "the default" instead of presenting a guess as a setting.
"""

from __future__ import annotations

from dataclasses import dataclass

from kneebench.analysis.context import PointView, RunContext

# A level counts as held at the cap when requests waited while the cache had room.
WAITING_MIN = 1.0
KV_ROOM = 0.9
# Two queued levels whose largest step agree this closely share one ceiling.
PLATEAU_SPREAD = 0.1


@dataclass(frozen=True)
class Setting:
    value: int
    source: str  # "set" (the user said), "inferred" (from the engine), "default"

    @property
    def is_default(self) -> bool:
        return self.source == "default"


def _levels(ctx: RunContext) -> list[PointView]:
    return sorted(
        (p for p in ctx.points_in("concurrency") if p.concurrency),
        key=lambda p: p.concurrency or 0,
    )


def _held(p: PointView) -> bool:
    """Requests waited although the cache had room and nothing was evicted."""
    return (
        (p.server_stat("waiting", "max") or 0) >= WAITING_MIN
        and (p.server_stat("kv_cache_usage", "max") or 0) < KV_ROOM
        and not (p.server_stat("preemptions", "total") or 0)
    )


def inferred_max_num_seqs(ctx: RunContext) -> int | None:
    """The running count flattening below the offered load while requests wait (note 31)."""
    held = [
        int(running)
        for p in _levels(ctx)
        if _held(p)
        and (running := p.server_stat("running", "max"))
        and running < (p.concurrency or 0)
    ]
    if len(held) < 2 or max(held) - min(held) > 1:
        return None
    return max(held)


def inferred_token_budget(ctx: RunContext, default: int) -> int | None:
    """The largest step under load, when it proves the default wrong (note 35 section 3).

    A step bigger than the default means the budget is larger. A ceiling well under it, at
    two or more queued levels, means it was set lower. Steps come from histogram estimates,
    so the value is approximate.
    """
    steps = [
        step
        for p in _levels(ctx)
        if (step := p.server_stat("iteration_tokens", "p99_estimate")) is not None
    ]
    if not steps:
        return None
    biggest = max(steps)
    if biggest > default * (1 + PLATEAU_SPREAD):
        return round(biggest)
    queued = [
        step
        for p in _levels(ctx)
        if _held(p) and (step := p.server_stat("iteration_tokens", "p99_estimate")) is not None
    ]
    if len(queued) >= 2:
        top = max(queued)
        if top < default * 0.5 and (top - min(queued)) <= PLATEAU_SPREAD * top:
            return round(top)
    return None


def max_num_seqs(ctx: RunContext) -> Setting:
    if ctx.setup.max_num_seqs:
        return Setting(ctx.setup.max_num_seqs, "set")
    seen = inferred_max_num_seqs(ctx)
    if seen is not None:
        return Setting(seen, "inferred")
    return Setting(ctx.theory.default_max_num_seqs, "default")


def token_budget(ctx: RunContext) -> Setting:
    default = ctx.theory.default_max_num_batched_tokens
    seen = inferred_token_budget(ctx, default)
    if seen is not None:
        return Setting(seen, "inferred")
    return Setting(default, "default")

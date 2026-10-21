"""A rules: what the server does when users leave mid-answer (note 36)."""

from __future__ import annotations

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, ok, rule, skip, warn
from kneebench.phases.plan import CANCEL_PHASE

# Note 36 section 3: an aborted request frees its slot on the next step, so it generates
# about what fits in the delay. Most of the full answer means nobody stopped it.
KEPT_GOING_SHARE = 0.8
STOPPED_SHARE = 0.5
ABORTED_SHARE = 0.8
# Requests finishing beyond ours mean someone else is using the server: counters are shared.
OTHER_TRAFFIC_SLACK = 2


@rule("A01", tags={"reliability", "capacity"}, source="36#3")
def abandoned_requests(ctx: RunContext) -> Finding | None:
    pts = ctx.points_in(CANCEL_PHASE)
    probe = pts[0].series.get("cancel") if pts else None
    if not probe:
        return None
    before, after = probe["before"], probe["after"]
    n = probe["requests"] or 0
    if not n:
        return None
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    finished = sum(v for k, v in delta.items() if k != "generated")
    generated = delta.get("generated", 0.0)
    per_request = generated / n
    data = {
        "requests": n,
        "aborted": delta.get("abort", 0.0),
        "finished": finished,
        "generated": generated,
        "per_request": per_request,
        "osl": probe["osl"],
        "cancel_after_s": probe["cancel_after_s"],
        "linger_s": probe["linger_s"],
        "concurrency": pts[0].concurrency,
    }
    if finished > n + OTHER_TRAFFIC_SLACK or generated > n * probe["osl"] * 1.2:
        return skip("A01", "other_traffic", **data)
    if delta.get("abort", 0.0) >= ABORTED_SHARE * n or per_request < STOPPED_SHARE * probe["osl"]:
        return ok("A01", **data)
    if per_request >= KEPT_GOING_SHARE * probe["osl"]:
        return warn("A01", wasted=generated, **data)
    return skip("A01", "unclear", **data)

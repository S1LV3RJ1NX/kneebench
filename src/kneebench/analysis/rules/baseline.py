"""B rules: the single-request baseline judged against physics (note 24)."""

from __future__ import annotations

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, bad, info, ok, rule, skip, warn

# Note 24 section 4: decode at 60-80% MBU means the engine is doing its job.
MBU_HEALTHY = 0.60
MBU_POOR = 0.40
# Note 24 section 5: TPOT around 4x the floor is a symptom; flag from 2.5x (MBU < 40%).
TPOT_SYMPTOM_RATIO = 2.5
# TTFT far from its floor while TPOT is fine points away from the GPU (note 24 section 5).
TTFT_ODD_RATIO = 10.0
TTFT_ODD_MIN_MS = 50.0
# Note 24 section 5: wide spread between runs means trust none of the numbers yet.
SPREAD_MAX_CV = 0.10
SPREAD_MAX_P99_OVER_P50 = 1.5
DRIFT_MAX = 0.10


def _baseline_tpot(ctx: RunContext) -> tuple[float, float] | None:
    base = ctx.baseline
    shape = ctx.theory.shape("baseline")
    if base is None or shape is None:
        return None
    tpot = base.m("inter_token_latency", "p50")
    return (tpot, shape.tpot_floor_ms) if tpot else None


@rule("B01", tags={"physics", "latency", "tpot"}, source="24#4")
def mbu_band(ctx: RunContext) -> Finding | None:
    measured = _baseline_tpot(ctx)
    if measured is None:
        return None
    tpot, floor = measured
    mbu = floor / tpot
    base = ctx.baseline
    shape = ctx.theory.shape("baseline")
    data = {
        "tpot_ms": tpot,
        "floor_ms": floor,
        "mbu": mbu,
        "ratio": tpot / floor,
        "gpu_name": ctx.setup.gpu_name,
        "requests": base.m("request_count", "avg") if base else None,
        # For the measured-vs-floor chart.
        "e2e_ms": base.m("request_latency", "p50") if base else None,
        "e2e_floor_ms": shape.e2e_floor_ms if shape else None,
        "server_ttft_ms": ctx.server_ttft_ms(base) if base else None,
        "ttft_floor_ms": shape.ttft_floor_ms if shape else None,
    }
    if mbu > 1.0 / 0.97:
        return skip("B01", "above_physics", **data)
    if mbu >= MBU_HEALTHY:
        return ok("B01", **data)
    if mbu >= MBU_POOR:
        return warn("B01", **data)
    return bad("B01", **data)


@rule("B02", tags={"physics", "tpot", "setup"}, source="24#5")
def slow_decode_suspects(ctx: RunContext) -> Finding | None:
    measured = _baseline_tpot(ctx)
    if measured is None:
        return None
    tpot, floor = measured
    ratio = tpot / floor
    if ratio < TPOT_SYMPTOM_RATIO:
        return None
    return bad(
        "B02",
        tpot_ms=tpot,
        floor_ms=floor,
        ratio=ratio,
        enforce_eager=ctx.setup.enforce_eager,
        fp32=ctx.theory.model_weight_dtype in ("float32", "float"),
    )


@rule("B03", tags={"latency", "ttft"}, source="24#5")
def ttft_odd_one_out(ctx: RunContext) -> Finding | None:
    base = ctx.baseline
    shape = ctx.theory.shape("baseline")
    if base is None or shape is None:
        return None
    server_ttft = ctx.server_ttft_ms(base)
    from_engine = False
    if server_ttft is None:
        # Too far away to subtract the network reliably; the server's own metrics still know.
        server_ttft = base.engine_ms("engine_ttft")
        from_engine = server_ttft is not None
    if server_ttft is None:
        return skip(
            "B03",
            "network_dominates",
            ttft_ms=base.m("time_to_first_token"),
            http_rtt_ms=ctx.http_rtt_ms,
        )
    ratio = server_ttft / shape.ttft_floor_ms
    data = {
        "from_engine": from_engine,
        "server_ttft_ms": server_ttft,
        "floor_ms": shape.ttft_floor_ms,
        "ratio": ratio,
        "isl": shape.isl,
    }
    tpot_ok = (measured := _baseline_tpot(ctx)) is not None and (
        measured[0] / measured[1] < TPOT_SYMPTOM_RATIO
    )
    if tpot_ok and ratio >= TTFT_ODD_RATIO and server_ttft >= TTFT_ODD_MIN_MS:
        return warn("B03", **data)
    return ok("B03", **data)


@rule("B04", tags={"reliability", "tpot"}, source="24#5")
def spread(ctx: RunContext) -> Finding | None:
    base = ctx.baseline
    if base is None:
        return None
    avg = base.m("inter_token_latency", "avg")
    std = base.m("inter_token_latency", "std")
    p50 = base.m("inter_token_latency", "p50")
    p99 = base.m("inter_token_latency", "p99")
    if not (avg and std is not None and p50 and p99):
        return None
    cv = std / avg
    data = {
        "cv": cv,
        "p50_ms": p50,
        "p99_ms": p99,
        "p99_over_p50": p99 / p50,
        "requests": base.m("request_count", "avg"),
        "samples": [round(r.tpot_ms, 3) for r in base.requests if r.ok and r.tpot_ms][:200],
    }
    if cv > SPREAD_MAX_CV or p99 / p50 > SPREAD_MAX_P99_OVER_P50:
        return warn("B04", **data)
    return ok("B04", **data)


@rule("B05", tags={"physics", "ttft", "prefill"}, source="24#4")
def prefill_mfu(ctx: RunContext) -> Finding | None:
    """Prefill's share of the GPU's math. From the prompt-length sweep when it ran: the cost of
    one more prompt token is a slope, so the network and fixed overheads drop out (note 59)."""
    from kneebench.analysis.shape import axis, prompt_fit

    t = ctx.theory
    fit = prompt_fit(axis(ctx, "shape_prompt", single=True))
    if fit is not None and fit[0] > 0:
        floor_us = 2 * t.model_params / (t.gpu_peak_tflops * 1e12) * 1e6
        return info(
            "B05",
            mfu=floor_us / (fit[0] * 1000),
            from_slope=True,
            per_token_us=fit[0] * 1000,
            floor_per_token_us=floor_us,
        )
    base = ctx.baseline
    shape = t.shape("baseline")
    if base is None or shape is None:
        return None
    server_ttft = ctx.server_ttft_ms(base)
    if server_ttft is None or server_ttft <= 0:
        return None
    return info(
        "B05",
        mfu=shape.ttft_floor_ms / server_ttft,
        from_slope=False,
        server_ttft_ms=server_ttft,
        floor_ms=shape.ttft_floor_ms,
        isl=shape.isl,
    )


@rule("B06", tags={"reliability", "tpot"}, source="24#5")
def drift(ctx: RunContext) -> Finding | None:
    start, end = ctx.baseline, next(iter(ctx.points_in("baseline_end")), None)
    if start is None or end is None:
        return None
    a = start.m("inter_token_latency", "p50")
    b = end.m("inter_token_latency", "p50")
    if not (a and b):
        return None
    change = (b - a) / a
    data = {"start_ms": a, "end_ms": b, "change": change}
    return warn("B06", **data) if abs(change) > DRIFT_MAX else ok("B06", **data)

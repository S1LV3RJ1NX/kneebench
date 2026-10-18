"""T rules: theory before traffic, and setup sanity checks against physics."""

from __future__ import annotations

import math

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, bad, info, ok, rule, skip, warn
from kneebench.physics.gpus import catalog

# Note 27 section 4: below 1x the engine refuses to start; under 2x leaves almost no room.
STARTUP_FIT_REFUSE = 1.0
STARTUP_FIT_TIGHT = 2.0
# Measured TPOT this far under the floor can't be noise: the bandwidth claim is wrong.
BELOW_FLOOR_TOLERANCE = 0.97


@rule("T01", tags={"physics", "latency"}, source="24#2")
def floors(ctx: RunContext) -> Finding:
    t = ctx.theory
    shapes = [
        {
            "name": s.name,
            "isl": s.isl,
            "osl": s.osl,
            "tpot_ms": s.tpot_floor_ms,
            "ttft_ms": s.ttft_floor_ms,
            "e2e_ms": s.e2e_floor_ms,
        }
        for s in t.shapes
    ]
    return info(
        "T01",
        shapes=shapes,
        baseline=next(s for s in shapes if s["name"] == "baseline"),
        weight_gb=t.model_weight_bytes / 1e9,
        bandwidth_tbps=t.gpu_bandwidth_tbps * t.gpu_count,
        peak_tflops=t.gpu_peak_tflops * t.gpu_count,
        gpu_count=t.gpu_count,
    )


@rule("T02", tags={"memory", "kv"}, source="26#2")
def kv_cache(ctx: RunContext) -> Finding:
    t = ctx.theory
    if t.pool_tokens is None:
        return skip("T02", "no_cache_config", kv_bytes_per_token=t.kv_bytes_per_token)
    return info(
        "T02",
        kv_bytes_per_token=t.kv_bytes_per_token,
        kv_kib_per_token=t.kv_bytes_per_token / 1024,
        pool_tokens=t.pool_tokens,
        kv_pool_gib=t.kv_pool_gib_per_gpu,
        cache_dtype=t.cache_dtype,
        gpu_memory_utilization=t.gpu_memory_utilization,
    )


@rule("T03", tags={"memory", "capacity"}, source="26#4")
def requests_that_fit(ctx: RunContext) -> Finding:
    t = ctx.theory
    if t.pool_tokens is None:
        return skip("T03", "no_cache_config")
    return info(
        "T03",
        pool_tokens=t.pool_tokens,
        shapes=[{"name": s.name, "isl": s.isl, "osl": s.osl, "fits": s.fits} for s in t.shapes],
    )


@rule("T04", tags={"memory", "startup"}, source="27#4")
def startup_fit(ctx: RunContext) -> Finding:
    t = ctx.theory
    if t.startup_fit is None:
        return skip("T04", "no_cache_config")
    data: dict[str, object] = {
        "startup_fit": t.startup_fit,
        "max_model_len": t.max_model_len,
        "pool_tokens": t.pool_tokens,
    }
    if t.pool_tokens:
        # Largest power of two that leaves room for STARTUP_FIT_TIGHT full-length requests.
        target = t.pool_tokens / STARTUP_FIT_TIGHT
        data["suggest_max_model_len"] = 2 ** max(int(math.log2(target)), 10)
    if t.startup_fit < STARTUP_FIT_REFUSE:
        return bad("T04", **data)
    if t.startup_fit < STARTUP_FIT_TIGHT:
        return warn("T04", **data)
    return ok("T04", **data)


@rule("T05", tags={"shape"}, source="26#5")
def adapted_shapes(ctx: RunContext) -> Finding | None:
    adapted = [s for s in ctx.theory.shapes if s.adapted]
    if not adapted:
        return None
    return info(
        "T05",
        max_model_len=ctx.theory.max_model_len,
        shapes=[
            {
                "name": s.name,
                "from_isl": s.original_isl,
                "from_osl": s.original_osl,
                "isl": s.isl,
                "osl": s.osl,
            }
            for s in adapted
        ],
    )


@rule("T06", tags={"physics", "setup"}, source="24#4")
def faster_than_physics(ctx: RunContext) -> Finding | None:
    base = ctx.baseline
    shape = ctx.theory.shape("baseline")
    if base is None or shape is None:
        return None
    tpot = base.m("inter_token_latency", "p50")
    if tpot is None:
        return None
    data = {
        "tpot_ms": tpot,
        "floor_ms": shape.tpot_floor_ms,
        "gpu_name": ctx.setup.gpu_name,
        "gpu_count": ctx.setup.gpu_count,
    }
    if tpot >= shape.tpot_floor_ms * BELOW_FLOOR_TOLERANCE:
        return ok("T06", **data)
    implied_tbps = ctx.theory.model_weight_bytes / (tpot / 1e3) / 1e12 / ctx.setup.gpu_count
    candidates = [g.name for g in catalog() if g.bandwidth_tbps >= implied_tbps]
    return bad("T06", implied_tbps=implied_tbps, candidates=candidates, **data)


@rule("T07", tags={"memory", "setup"}, source="27#4")
def implied_vram(ctx: RunContext) -> Finding:
    t = ctx.theory
    if t.implied_min_gpu_gib is None:
        return skip("T07", "no_cache_config")
    data = {
        "implied_min_gib": t.implied_min_gpu_gib,
        "gpu_memory_gib": t.gpu_memory_gib,
        "gpu_name": t.gpu_name,
        "kv_pool_gib": t.kv_pool_gib_per_gpu,
        "weights_gib_per_gpu": t.model_weight_bytes / t.gpu_count / 1024**3,
        "gpu_memory_utilization": t.gpu_memory_utilization,
    }
    if t.implied_min_gpu_gib <= t.gpu_memory_gib:
        return ok("T07", **data)
    candidates = [g.name for g in catalog() if g.memory_gib >= t.implied_min_gpu_gib]
    return bad("T07", candidates=candidates, **data)


@rule("T08", tags={"physics", "setup"}, source="24#5")
def fp32_weights(ctx: RunContext) -> Finding | None:
    if ctx.theory.model_weight_dtype in ("float32", "float"):
        return warn("T08", weight_gb=ctx.theory.model_weight_bytes / 1e9)
    return None

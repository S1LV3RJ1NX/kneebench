"""F rules: where the measured load sits on the roofline, and what other number formats would
change (notes 50, 51, 53 and 61). Physics on top of measurements; no extra traffic."""

from __future__ import annotations

from typing import Any

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, info, rule, warn
from kneebench.analysis.rules.concurrency import sweep
from kneebench.analysis.sweep import operating_point
from kneebench.phases.theory import Theory
from kneebench.physics.floors import requests_that_fit
from kneebench.physics.gpus import get_gpu
from kneebench.physics.model import FP8_KV_DTYPES
from kneebench.physics.roofline import (
    WHAT_IFS,
    Chip,
    Model,
    project,
    ridge_batch,
    step_s,
)

# Note 42 section 6: a cache past ~90% full under load is the memory wall.
KV_FULL = 0.9
# Bits per weight at or below which a format is already as small as that what-if.
EIGHT_BIT = 8.5
FOUR_BIT = 5.0
WEIGHT_LABELS = {"fp8": "FP8 weights", "int8": "INT8 weights", "w4a16": "4-bit weights"}


def _chip(t: Theory) -> Chip | None:
    bf16, fp8 = t.gpu_bf16_tflops, t.gpu_fp8_tflops
    if bf16 is None:  # runs stored before these fields existed
        try:
            gpu = get_gpu(t.gpu_id)
        except KeyError:
            return None
        bf16, fp8 = gpu.bf16_tflops, gpu.fp8_tflops
    n = t.gpu_count
    return Chip(t.gpu_bandwidth_tbps * 1e12 * n, bf16 * 1e12 * n, fp8 * 1e12 * n if fp8 else None)


def _model(t: Theory) -> Model:
    return Model(t.model_params, t.model_weight_bytes, t.kv_bytes_per_token, t.model_attn_width)


def _context(ctx: RunContext) -> tuple[int, int, float] | None:
    shape = ctx.theory.shape(ctx.primary_shape) or ctx.theory.shape("baseline")
    if shape is None:
        return None
    # Halfway through its answer, the average request has read the prompt and half the output.
    return shape.isl, shape.osl, shape.isl + shape.osl / 2


@rule("F01", tags={"physics", "throughput", "tpot"}, source="61#4")
def roofline(ctx: RunContext) -> Finding | None:
    t, chip, shape = ctx.theory, _chip(ctx.theory), _context(ctx)
    pts = [p for p in sweep(ctx) if p.tpot.get("p50")]
    if chip is None or shape is None or not pts:
        return None
    _, _, context = shape
    model, compute = _model(t), t.model_compute
    ridge = chip.ridge(compute)
    rows = []
    for p in pts:
        b = p.concurrency
        intensity = model.intensity(b, context)
        achieved = model.decode_flops(b, context) / (p.tpot["p50"] / 1000)
        ceiling = min(chip.peak(compute), chip.bandwidth_bytes_s * intensity)
        rows.append(
            {
                "concurrency": b,
                "intensity": intensity,
                "achieved_tflops": achieved / 1e12,
                "ceiling_tflops": ceiling / 1e12,
                "share": achieved / ceiling,
                "floor_ms": step_s(model, chip, compute, b, context) * 1000,
                "tpot_ms": p.tpot["p50"],
            }
        )
    return info(
        "F01",
        rows=rows,
        ridge=ridge,
        peak_tflops=chip.peak(compute) / 1e12,
        bandwidth_tbps=chip.bandwidth_bytes_s / 1e12,
        compute=compute,
        context=context,
        limit=model.intensity_limit(context),
        ridge_batch=ridge_batch(model, chip, compute, context),
        all_memory_bound=all(r["intensity"] < ridge for r in rows),
        top=rows[-1],
        gpu_name=ctx.setup.gpu_name,
    )


@rule("F02", tags={"physics", "memory", "capacity"}, source="50#5")
def precision_what_ifs(ctx: RunContext) -> Finding | None:
    t, chip, shape = ctx.theory, _chip(ctx.theory), _context(ctx)
    if chip is None or shape is None:
        return None
    isl, osl, context = shape
    served = _model(t)
    bits = t.model_weight_bytes * 8 / t.model_params
    pts = sweep(ctx)
    op = operating_point(pts, ctx.workload.slo) if pts else None
    batch = op.concurrency if op else 1
    measured = op.tpot.get("p50") if op else None
    fp8_kv = t.cache_dtype.lower() in FP8_KV_DTYPES

    def applies(what_id: str) -> bool:
        return {
            "fp8_kv": not fp8_kv,
            "fp8": chip.fp8_flops is not None and bits > EIGHT_BIT,
            "int8": chip.fp8_flops is None and bits > EIGHT_BIT,
            "w4a16": bits > FOUR_BIT,
        }[what_id]

    def fits(pool: float | None) -> float | None:
        return requests_that_fit(int(pool), isl, osl, t.block_size) if pool else None

    rows: list[dict[str, Any]] = [
        {
            "id": "served",
            "weights_gb": t.model_weight_bytes / 1e9,
            "pool_tokens": t.pool_tokens,
            "fits": fits(t.pool_tokens),
            "tpot_ms": measured,
            "decode_speedup": 1.0,
            "decode_speedup_one": 1.0,
            "prefill_speedup": 1.0,
        }
    ]
    for what in WHAT_IFS:
        if not applies(what.id):
            continue
        p = project(served, chip, t.model_compute, t.pool_tokens, batch, context, what)
        rows.append(
            {
                "id": what.id,
                "weights_gb": p.weight_bytes / 1e9,
                "pool_tokens": p.pool_tokens,
                "fits": fits(p.pool_tokens),
                "tpot_ms": measured * p.tpot_ratio if measured else None,
                "decode_speedup": 1 / p.tpot_ratio,
                "decode_speedup_one": 1 / p.tpot_ratio_one,
                "prefill_speedup": 1 / p.ttft_ratio,
            }
        )
    if len(rows) == 1:
        return None
    weights = [r for r in rows if r["id"] in WEIGHT_LABELS]
    best = max(weights, key=lambda r: r["decode_speedup"], default=None)
    kv_max = max((p.kv_usage_max or 0 for p in pts), default=0)
    preempted = any((p.preemptions or 0) > 0 for p in pts)
    data = {
        "rows": rows,
        "bits": bits,
        "scheme": t.model_quant_scheme,
        "batch": batch,
        "measured": measured is not None,
        "has_fp8": chip.fp8_flops is not None,
        "gpu_name": ctx.setup.gpu_name,
        "kv_max": kv_max,
        "preempted": preempted,
        "fp8_kv_row": next((r for r in rows if r["id"] == "fp8_kv"), None),
        "best": best,
        "best_label": WEIGHT_LABELS[best["id"]] if best else None,
        "isl": isl,
        "osl": osl,
    }
    if not fp8_kv and (preempted or kv_max >= KV_FULL):
        return warn("F02", **data)
    return info("F02", **data)

"""S rules: prompt and output length (note 26)."""

from __future__ import annotations

import math

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, info, ok, rule, warn
from kneebench.analysis.shape import (
    ShapePoint,
    axis,
    fit_line,
    meets_slo,
    operating_concurrency,
    prompt_fit,
    shape_map,
)
from kneebench.domain import USER_SHAPE_PREFIX
from kneebench.physics.floors import requests_that_fit

# Note 26 section 2: attention adds ~13% at 8k; well above that means prefill is bending up.
TTFT_BEND_WARN = 1.3
# Note 26 section 2: long prompts slow decode for the whole batch.
DECODE_TAX_WARN = 1.25
# Note 26 section 3: requests finished per second falls with output length.
SLOT_DROP_INFO = 2.0


def _tail(ctx: RunContext) -> int:
    return ctx.workload.slo.percentile


def _rows(points: list[ShapePoint], pct: int) -> list[dict[str, object]]:
    return [p.row(pct) for p in points]


@rule("S01", tags={"ttft", "prefill", "shape"}, source="26#2")
def ttft_vs_prompt(ctx: RunContext) -> Finding | None:
    single = axis(ctx, "shape_prompt", single=True)
    fit = prompt_fit(single)
    if fit is None:
        return None
    slope, intercept = fit
    longest = single[-1]
    measured = longest.point.ttft.get("p50") or 0.0
    predicted = slope * longest.isl + intercept
    bend = measured / predicted if predicted > 0 else None
    base = ctx.theory.shape("baseline")
    floor_us = base.ttft_floor_ms / base.isl * 1000 if base and base.isl else None
    data = {
        "rows": _rows(single, _tail(ctx)),
        "per_1k_ms": slope * 1000,
        "per_token_us": slope * 1000,
        "floor_per_token_us": floor_us,
        "longest_isl": longest.isl,
        "longest_ttft_ms": measured,
        "bend": bend,
        "fixed_osl": longest.osl,
    }
    if bend is not None and bend > TTFT_BEND_WARN:
        return warn("S01", case="ttft_bends", **data)
    return info("S01", case="linear", **data)


@rule("S02", tags={"tpot", "kv", "shape"}, source="26#2")
def decode_tax(ctx: RunContext) -> Finding | None:
    loaded = axis(ctx, "shape_prompt", single=False)
    single = axis(ctx, "shape_prompt", single=True)
    pts = loaded or single
    if len(pts) < 2:
        return None
    pct = _tail(ctx)
    first, last = pts[0].point.tpot.get("p50"), pts[-1].point.tpot.get("p50")
    if not first or not last:
        return None
    growth = last / first
    target = ctx.workload.slo.tpot_ms
    tail = pts[-1].point.tpot.get(f"p{pct}")
    data = {
        "rows": _rows(pts, pct),
        "single_rows": _rows(single, pct),
        "concurrency": pts[0].concurrency,
        "shortest_isl": pts[0].isl,
        "longest_isl": pts[-1].isl,
        "tpot_short_ms": first,
        "tpot_long_ms": last,
        "growth": growth,
        "tpot_target_ms": target,
        "pct_label": pct,
        "misses_target": bool(target and tail and tail > target),
    }
    if growth >= DECODE_TAX_WARN and data["misses_target"]:
        return warn("S02", case="misses", **data)
    if growth >= DECODE_TAX_WARN:
        return info("S02", case="taxed", **data)
    return ok("S02", case="flat", **data)


@rule("S03", tags={"tpot", "e2e", "shape", "reasoning"}, source="26#3")
def e2e_vs_output(ctx: RunContext) -> Finding | None:
    single = axis(ctx, "shape_output", single=True)
    pairs = [(p.osl, p.e2e_p50_ms) for p in single if p.e2e_p50_ms is not None]
    fit = fit_line([float(x) for x, _ in pairs], [float(y) for _, y in pairs])
    if fit is None:
        return None
    slope, intercept = fit
    longest = single[-1]
    return info(
        "S03",
        case="linear",
        **{
            "rows": _rows(single, _tail(ctx)),
            "per_token_ms": slope,
            "intercept_ms": intercept,
            "longest_osl": longest.osl,
            "longest_e2e_ms": longest.e2e_p50_ms,
            "fixed_isl": longest.isl,
        },
    )


@rule("S04", tags={"capacity", "kv", "shape", "reasoning"}, source="26#3")
def long_outputs_hold_slots(ctx: RunContext) -> Finding | None:
    loaded = axis(ctx, "shape_output", single=False)
    if len(loaded) < 2:
        return None
    first, last = loaded[0].point, loaded[-1].point
    if not first.requests_s or not last.requests_s:
        return None
    drop = first.requests_s / last.requests_s
    tok_ratio = (last.output_tok_s or 0) / first.output_tok_s if first.output_tok_s else None
    data = {
        "rows": _rows(loaded, _tail(ctx)),
        "concurrency": loaded[0].concurrency,
        "shortest_osl": loaded[0].osl,
        "longest_osl": loaded[-1].osl,
        "req_s_short": first.requests_s,
        "req_s_long": last.requests_s,
        "drop": drop,
        "tok_ratio": tok_ratio,
        "preemptions": sum(p.point.preemptions or 0 for p in loaded),
        # Note 26 section 4: how many requests of the longest shape the cache holds at full length.
        "sustain": (
            math.floor(
                requests_that_fit(
                    ctx.theory.pool_tokens, loaded[-1].isl, loaded[-1].osl, ctx.theory.block_size
                )
            )
            if ctx.theory.pool_tokens
            else None
        ),
    }
    if data["preemptions"]:
        return warn("S04", case="preempted", **data)
    return info("S04", case="drops" if drop >= SLOT_DROP_INFO else "steady", **data)


@rule("S05", tags={"shape"}, source="26#6")
def your_traffic(ctx: RunContext) -> Finding | None:
    mine = [s for s in ctx.theory.shapes if s.name.startswith(USER_SHAPE_PREFIX)]
    if not mine and not ctx.points_in("concurrency"):
        return None  # the baseline uses its own fixed shape
    if not mine:
        return info("S05", case="standard_only", **{"primary": ctx.primary_shape})
    trimmed = [s for s in mine if s.adapted]
    data = {
        "shapes": [
            {
                "name": s.name,
                "isl": s.isl,
                "osl": s.osl,
                "original_isl": s.original_isl,
                "original_osl": s.original_osl,
            }
            for s in mine
        ],
        "trimmed": len(trimmed),
    }
    if trimmed:
        return warn("S05", case="trimmed", **data)
    return None


@rule("S06", tags={"shape", "capacity", "slo"}, source="26#6")
def map_of_shapes(ctx: RunContext) -> Finding | None:
    shapes = shape_map(ctx)
    if len(shapes) < 2:
        return None
    pct = _tail(ctx)
    rows = []
    for s in shapes:
        row = s.row(pct)
        row["meets"] = meets_slo(ctx, s)
        rows.append(row)
    failing = [r["name"] for r in rows if not r["meets"]]
    data = {
        "rows": rows,
        "concurrency": operating_concurrency(ctx),
        "primary": shapes[0].name,
        "failing": failing,
        "primary_meets": rows[0]["meets"],
        "pct_label": pct,
        "has_user": any(s.name.startswith(USER_SHAPE_PREFIX) for s in shapes),
        "tpot_target_ms": ctx.workload.slo.tpot_ms,
        "ttft_target_ms": ctx.workload.slo.ttft_ms,
        "has_slo": ctx.workload.slo.tpot_ms is not None or ctx.workload.slo.ttft_ms is not None,
    }
    if data["has_slo"] and failing and rows[0]["meets"]:
        return warn("S06", case="others_fail", **data)
    return info("S06", case="map", **data)


@rule("S07", tags={"prefill", "shape"}, source="26#1")
def exchange_rate(ctx: RunContext) -> Finding | None:
    single = axis(ctx, "shape_prompt", single=True)
    fit = prompt_fit(single)
    base = next((p for p in ctx.points_in("baseline")), None)
    tpot = base.m("inter_token_latency", "p50") if base else None
    if fit is None or not tpot or fit[0] <= 0:
        return None
    rate = tpot / fit[0]
    shape = ctx.theory.shape("baseline")
    theory = (
        shape.tpot_floor_ms / (shape.ttft_floor_ms / shape.isl)
        if shape and shape.isl and shape.ttft_floor_ms
        else None
    )
    return info(
        "S07",
        case="rate",
        **{"rate": rate, "theory_rate": theory, "tpot_ms": tpot, "per_token_us": fit[0] * 1000},
    )

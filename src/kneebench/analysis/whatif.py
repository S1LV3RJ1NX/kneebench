"""What-if estimates for a request size the run didn't test, from the shape sweep (note 26)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from kneebench.analysis.context import RunContext
from kneebench.analysis.shape import axis, fit_line, operating_concurrency, prompt_fit
from kneebench.domain import CONTEXT_MARGIN_TOKENS


@dataclass(frozen=True)
class Estimate:
    isl: int
    osl: int
    ttft_ms: float | None  # one user, server side (the network round trip removed)
    tpot_ms: float | None  # one user
    e2e_ms: float | None
    loaded_concurrency: int | None
    loaded_tpot_ms: float | None  # median at the operating point
    fits: int | None  # requests of this size the cache holds at full length
    fits_context: bool
    max_model_len: int | None
    extrapolated: tuple[str, ...]  # which axes go past what was measured


def _interpolate(xs: list[float], ys: list[float], x: float) -> float | None:
    """Piecewise-linear inside the measured range; None outside it."""
    pairs = sorted(zip(xs, ys, strict=True))
    for (x0, y0), (x1, y1) in pairwise(pairs):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def estimate(ctx: RunContext, isl: int, osl: int) -> Estimate | None:
    prompt = axis(ctx, "shape_prompt", single=True)
    output = axis(ctx, "shape_output", single=True)
    fit = prompt_fit(prompt)
    if fit is None:
        return None
    extrapolated: list[str] = []

    measured = [(p.isl, p.point.ttft.get("p50")) for p in prompt]
    pts = [(float(x), float(y)) for x, y in measured if y is not None]
    ttft = _interpolate([x for x, _ in pts], [y for _, y in pts], isl)
    if ttft is None:
        extrapolated.append("prompt")
        ttft = fit[1] + fit[0] * isl

    pairs = [(float(p.osl), p.e2e_p50_ms) for p in output if p.e2e_p50_ms is not None]
    line = fit_line([x for x, _ in pairs], [float(y) for _, y in pairs])
    base = next(iter(ctx.points_in("baseline")), None)
    tpot = line[0] if line else (base.m("inter_token_latency", "p50") if base else None)
    if pairs and not (min(x for x, _ in pairs) <= osl <= max(x for x, _ in pairs)):
        extrapolated.append("output")

    # Decode rereads the whole context, so loaded speed follows the mean context length.
    loaded = axis(ctx, "shape_prompt", single=False)
    ctxs = [float(p.isl + p.osl / 2) for p in loaded]
    tpots = [p.point.tpot.get("p50") for p in loaded]
    usable = [(c, t) for c, t in zip(ctxs, tpots, strict=True) if t is not None]
    mean_ctx = isl + osl / 2
    loaded_tpot = _interpolate([c for c, _ in usable], [t for _, t in usable], mean_ctx)
    if loaded_tpot is None and len(usable) >= 2:
        slope_icpt = fit_line([c for c, _ in usable], [t for _, t in usable])
        loaded_tpot = slope_icpt[1] + slope_icpt[0] * mean_ctx if slope_icpt else None

    pool = ctx.theory.pool_tokens
    limit = ctx.max_model_len
    return Estimate(
        isl=isl,
        osl=osl,
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_ms=ttft + tpot * (osl - 1) if tpot is not None else None,
        loaded_concurrency=operating_concurrency(ctx) if loaded else None,
        loaded_tpot_ms=loaded_tpot,
        fits=pool // (isl + osl) if pool else None,
        fits_context=limit is None or isl + osl + CONTEXT_MARGIN_TOKENS <= limit,
        max_model_len=limit,
        extrapolated=tuple(extrapolated),
    )

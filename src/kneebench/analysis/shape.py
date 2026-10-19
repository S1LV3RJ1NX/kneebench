"""Shape sweep analysis (note 26): prompt length and output length, one axis at a time."""

from __future__ import annotations

from dataclasses import dataclass

from kneebench.analysis.context import PointView, RunContext
from kneebench.analysis.sweep import (
    SweepPoint,
    curve,
    meets,
    operating_point,
    saturation,
    summarize,
)
from kneebench.domain import Slo


@dataclass(frozen=True)
class ShapePoint:
    name: str
    isl: int
    osl: int
    concurrency: int
    point: SweepPoint
    e2e_p50_ms: float | None

    def row(self, pct: int) -> dict[str, object]:
        p = self.point
        return {
            "name": self.name,
            "isl": self.isl,
            "osl": self.osl,
            "concurrency": self.concurrency,
            "ttft_p50": p.ttft.get("p50"),
            "ttft_tail": p.ttft.get(f"p{pct}"),
            "tpot_p50": p.tpot.get("p50"),
            "tpot_tail": p.tpot.get(f"p{pct}"),
            "e2e_p50": self.e2e_p50_ms,
            "tok_s": p.output_tok_s,
            "req_s": p.requests_s,
            "kv_max": p.kv_usage_max,
            "preemptions": p.preemptions,
            "errors": p.error_rate,
        }


def _shape(ctx: RunContext, point: PointView, name: str) -> ShapePoint:
    return ShapePoint(
        name=name,
        isl=point.isl,
        osl=point.osl,
        concurrency=point.concurrency or 1,
        point=summarize(point, ctx.http_rtt_ms or 0.0),
        e2e_p50_ms=point.m("request_latency", "p50"),
    )


def axis(ctx: RunContext, phase: str, single: bool) -> list[ShapePoint]:
    """One axis of the shape sweep, at 1 user (single) or at the operating point."""
    pts = [p for p in ctx.points_in(phase) if (p.concurrency == 1) == single]
    key = (lambda p: p.isl) if phase == "shape_prompt" else (lambda p: p.osl)
    return [_shape(ctx, p, phase) for p in sorted(pts, key=key)]


def comparison_load(pts: list[SweepPoint], slo: Slo) -> SweepPoint | None:
    """Where shapes are compared: the operating point, else where throughput saturates."""
    if not pts:
        return None
    return operating_point(pts, slo) or saturation(pts) or pts[-1]


def operating_concurrency(ctx: RunContext) -> int | None:
    op = comparison_load(
        curve(ctx.points_in("concurrency"), ctx.http_rtt_ms or 0.0), ctx.workload.slo
    )
    return op.concurrency if op else None


def shape_map(ctx: RunContext) -> list[ShapePoint]:
    """Every traffic shape at the same load: the primary shape plus each mapped one."""
    names = {(s.isl, s.osl): s.name for s in ctx.theory.shapes if s.name != "baseline"}
    out = [
        _shape(ctx, p, names.get((p.isl, p.osl), f"{p.isl}/{p.osl}"))
        for p in ctx.points_in("shape_map")
    ]
    c = operating_concurrency(ctx)
    primary = ctx.theory.shape(ctx.primary_shape)
    if c is not None and primary is not None:
        match = next((p for p in ctx.points_in("concurrency") if p.concurrency == c), None)
        if match is not None:
            out.insert(0, _shape(ctx, match, names.get((match.isl, match.osl), primary.name)))
    return out


def meets_slo(ctx: RunContext, sp: ShapePoint) -> bool:
    return meets(sp.point, ctx.workload.slo)


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Least-squares slope and intercept."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return slope, my - slope * mx


def prompt_fit(points: list[ShapePoint]) -> tuple[float, float] | None:
    """TTFT P50 against prompt length; the slope is per prompt token and network-free."""
    pairs = [(p.isl, p.point.ttft.get("p50")) for p in points]
    usable = [(x, y) for x, y in pairs if y is not None]
    if len(usable) < 3:
        return None
    # Fit the short prompts, where prefill is still a straight line.
    head = usable[:-1] if len(usable) > 3 else usable
    return fit_line([float(x) for x, _ in head], [float(y) for _, y in head])

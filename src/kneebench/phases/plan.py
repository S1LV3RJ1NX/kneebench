"""Turn the requested phases into concrete AIPerf points, with an ETA."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from kneebench.domain import (
    LENGTH_VARIATION,
    ProductType,
    Slo,
    context_budget,
    primary_shape_name,
)
from kneebench.loadgen.aiperf.overhead import estimate_point_overhead_s
from kneebench.loadgen.base import Arrival, LoadMode, PointSpec
from kneebench.phases.theory import ShapeTheory, Theory

BASELINE_WARMUP = 10  # note 24: warm up, discarded
BASELINE_REPEATS = 30  # note 24: about 30 repeats give a median and a spread
# Real requests run slower than the floor; assume this until measurements exist.
UNMEASURED_SLOWDOWN = 2.0
# Note 25 section 2: N = 1, 2, 4 ... doubling up to 256 or until something breaks.
DEFAULT_MAX_CONCURRENCY = 256
# Rounds of requests per concurrency level: enough for a steady-state window (note 25).
SWEEP_ROUNDS = 4
SWEEP_MIN_REQUESTS = 16
SWEEP_MAX_WARMUP = 32
# Length variation per request (note 26 section 5): identical lengths make closed-loop
# requests finish and restart in lockstep, which turns into unrealistic prefill bursts.
SWEEP_LENGTH_VARIATION = LENGTH_VARIATION
# Note 25 section 4: random arrivals below, near and past capacity, then bursty ones.
POISSON_FACTORS = (0.25, 0.5, 0.75, 0.9, 1.1)
BURSTY_FACTORS = (0.75, 0.9)
BURSTINESS = 0.5  # gamma shape below 1 clusters arrivals; 1.0 is Poisson
# Long enough to see whether first-token time keeps growing (run-length dependence).
RATE_DURATION_S = 90.0
# A few request lifetimes at the target rate, so profiling doesn't start from an idle server.
RATE_WARMUP_S = 20.0
# Open-loop arrivals need room to queue; past this, the client is the limit.
RATE_CONCURRENCY_CAP = 2048
# Note 26 section 5: one axis at a time, in powers of two (here powers of four, to keep the
# run short), with the other axis fixed.
SHAPE_AXIS = (128, 512, 2048, 8192)
SHAPE_FIXED_OSL = 256  # for the prompt axis (note 26 section 2)
SHAPE_FIXED_ISL = 1024  # for the output axis (note 26 section 3)
SHAPE_SINGLE_REQUESTS = 6
# One long answer already gives thousands of per-token samples; three are enough for a median.
SHAPE_SINGLE_REQUESTS_LONG = 3
SHAPE_LONG_OUTPUT = 2048
# Points at the operating point run about this long, with 2 to SWEEP_ROUNDS rounds.
SHAPE_POINT_TARGET_S = 150
SHAPE_MIN_ROUNDS = 2
SHAPE_PHASES = ("shape_prompt", "shape_output", "shape_map")
# Safety net so one stuck point can't eat the whole budget.
POINT_DURATION_CAP_S = 900.0
SLICE_S = 10.0


@dataclass(frozen=True)
class PlannedPoint:
    phase: str
    spec: PointSpec
    eta_s: float
    # Rate points run at this multiple of the capacity the concurrency sweep measures.
    rate_factor: float | None = None
    # Shape points that run at the operating point the concurrency sweep finds.
    at_operating_point: bool = False


@dataclass(frozen=True)
class RunPlan:
    points: tuple[PlannedPoint, ...]
    dropped_for_budget: tuple[int, ...] = ()

    @property
    def eta_s(self) -> float:
        return sum(p.eta_s for p in self.points)


def baseline_spec(theory: Theory) -> PointSpec:
    shape = theory.shape("baseline")
    assert shape is not None
    return PointSpec(
        isl=shape.isl,
        osl=shape.osl,
        mode=LoadMode.CONCURRENCY,
        concurrency=1,
        request_count=BASELINE_REPEATS,
        warmup_count=BASELINE_WARMUP,
        slice_duration_s=SLICE_S,
    )


def concurrency_levels(
    theory: Theory, shape: ShapeTheory, user_max: int, max_num_seqs: int | None
) -> list[int]:
    """Powers of two up to the user's cap, the server's sequence limit and well past memory."""
    caps = [user_max]
    seqs = max_num_seqs or theory.default_max_num_seqs
    caps.append(2 * seqs)
    if shape.fits:
        caps.append(2 ** math.ceil(math.log2(max(shape.fits * 2, 2))))
    top = min(caps)
    return [2**i for i in range(int(math.log2(max(top, 1))) + 1)]


def sweep_spec(shape: ShapeTheory, concurrency: int, slo: Slo) -> PointSpec:
    return PointSpec(
        isl=shape.isl,
        osl=shape.osl,
        mode=LoadMode.CONCURRENCY,
        concurrency=concurrency,
        isl_stddev=round(shape.isl * SWEEP_LENGTH_VARIATION, 1),
        osl_stddev=round(shape.osl * SWEEP_LENGTH_VARIATION),
        request_count=max(SWEEP_ROUNDS * concurrency, SWEEP_MIN_REQUESTS),
        warmup_count=min(concurrency, SWEEP_MAX_WARMUP),
        duration_s=POINT_DURATION_CAP_S,
        slice_duration_s=SLICE_S,
        goodput=slo.goodput(),
    )


def rate_spec(shape: ShapeTheory, arrival: str, slo: Slo, rate: float = 1.0) -> PointSpec:
    """A rate point; the real rate is filled in once capacity has been measured."""
    return PointSpec(
        isl=shape.isl,
        osl=shape.osl,
        mode=LoadMode.RATE,
        request_rate=rate,
        arrival=Arrival(arrival),
        burstiness=BURSTINESS if arrival == "gamma" else None,
        concurrency=RATE_CONCURRENCY_CAP,
        isl_stddev=round(shape.isl * SWEEP_LENGTH_VARIATION, 1),
        osl_stddev=round(shape.osl * SWEEP_LENGTH_VARIATION),
        warmup_s=RATE_WARMUP_S,
        duration_s=RATE_DURATION_S,
        slice_duration_s=SLICE_S,
        goodput=slo.goodput(),
    )


def axis_lengths(max_model_len: int | None, fixed: int) -> list[int]:
    """Axis values that fit the context window next to the fixed side."""
    room = context_budget(max_model_len or 32_768) - fixed
    values = [v for v in SHAPE_AXIS if v <= room]
    largest = room // 256 * 256
    if values and largest >= values[-1] * 2 and largest not in values:
        values.append(min(largest, SHAPE_AXIS[-1] * 4))
    return values


def shape_spec(isl: int, osl: int, concurrency: int, slo: Slo) -> PointSpec:
    single = concurrency == 1
    return PointSpec(
        isl=isl,
        osl=osl,
        mode=LoadMode.CONCURRENCY,
        concurrency=concurrency,
        isl_stddev=round(isl * SWEEP_LENGTH_VARIATION, 1),
        osl_stddev=round(osl * SWEEP_LENGTH_VARIATION),
        request_count=(
            SHAPE_SINGLE_REQUESTS_LONG if osl >= SHAPE_LONG_OUTPUT else SHAPE_SINGLE_REQUESTS
        )
        if single
        else max(SWEEP_ROUNDS * concurrency, SWEEP_MIN_REQUESTS),
        warmup_count=1 if single else min(concurrency, SWEEP_MAX_WARMUP),
        duration_s=POINT_DURATION_CAP_S,
        slice_duration_s=SLICE_S,
        goodput=slo.goodput(),
    )


def estimate_e2e_ms(
    theory: Theory, shape: ShapeTheory, concurrency: int, slowdown: float = UNMEASURED_SLOWDOWN
) -> float:
    """Every decode step reads the weights once plus each user's cache (note 26 deriv. 4)."""
    weights = theory.model_weight_bytes
    avg_context = shape.isl + shape.osl / 2
    kv_read = concurrency * avg_context * theory.kv_bytes_per_token
    tpot = shape.tpot_floor_ms * slowdown * (weights + kv_read) / weights
    ttft = shape.ttft_floor_ms * slowdown * math.sqrt(concurrency)
    e2e = ttft + (shape.osl - 1) * tpot
    if shape.fits and concurrency > shape.fits:
        e2e *= concurrency / shape.fits  # past the memory wall the rest wait in the queue
    return e2e


def estimate_point_s(
    spec: PointSpec, e2e_ms: float, overhead_observed: list[float], first: bool
) -> float:
    concurrency = spec.concurrency or 1
    requests = (spec.request_count or 0) + (spec.warmup_count or 0)
    traffic = requests * e2e_ms / 1000 / concurrency
    if spec.duration_s:
        traffic = min(traffic, spec.duration_s) if spec.request_count else spec.duration_s
    return traffic + estimate_point_overhead_s(overhead_observed, first_launch=first)


def plan_run(
    theory: Theory,
    phases: tuple[str, ...],
    first_launch: bool = True,
    product: ProductType = ProductType.CHAT,
    slo: Slo | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_num_seqs: int | None = None,
    budget_s: float | None = None,
    measured_slowdown: float | None = None,
) -> RunPlan:
    points: list[PlannedPoint] = []
    slowdown = measured_slowdown or UNMEASURED_SLOWDOWN

    def add(phase: str, spec: PointSpec, shape: ShapeTheory) -> None:
        e2e = estimate_e2e_ms(theory, shape, spec.concurrency or 1, slowdown)
        first = first_launch and not points
        points.append(PlannedPoint(phase, spec, estimate_point_s(spec, e2e, [], first)))

    base_shape = theory.shape("baseline")
    assert base_shape is not None
    if "baseline" in phases:
        add("baseline", baseline_spec(theory), base_shape)

    dropped: list[int] = []
    for needs_sweep in ("rate", "shape"):
        if needs_sweep in phases and "concurrency" not in phases:
            raise ValueError(
                f"the {needs_sweep} sweep needs the concurrency sweep to find capacity first"
            )
    if "concurrency" in phases:
        primary = primary_shape_name(product, tuple(s.name for s in theory.shapes))
        shape = theory.shape(primary) or base_shape
        levels = concurrency_levels(theory, shape, max_concurrency, max_num_seqs)
        sweep = [(c, sweep_spec(shape, c, slo or Slo(None, None))) for c in levels]
        for _, spec in sweep:
            add("concurrency", spec, shape)
        if "rate" in phases:
            for arrival, factors in (("poisson", POISSON_FACTORS), ("gamma", BURSTY_FACTORS)):
                for factor in factors:
                    spec = rate_spec(shape, arrival, slo or Slo(None, None))
                    tail_s = estimate_e2e_ms(theory, shape, 64, slowdown) / 1000
                    eta = (
                        RATE_WARMUP_S
                        + RATE_DURATION_S
                        + tail_s
                        + estimate_point_overhead_s([], False)
                    )
                    points.append(PlannedPoint("rate", spec, eta, rate_factor=factor))
        if "shape" in phases:
            points.extend(_shape_points(theory, shape, slo or Slo(None, None), slowdown))
        # Re-measuring the baseline at the end catches a GPU that changed mid-run (B06).
        add("baseline_end", replace(baseline_spec(theory), warmup_count=2), base_shape)
        if budget_s is not None:
            dropped = _trim(points, budget_s)

    return RunPlan(tuple(points), tuple(dropped))


def _shape_points(
    theory: Theory, primary: ShapeTheory, slo: Slo, slowdown: float
) -> list[PlannedPoint]:
    """Prompt and output axes at 1 user and at the operating point, then the shape map."""
    out: list[PlannedPoint] = []
    # Concurrency at the operating point is unknown until the sweep runs; assume 32 for ETAs.
    eta_c = 32

    def point(phase: str, isl: int, osl: int, at_op: bool) -> PlannedPoint:
        shape = theory.shape("baseline")
        assert shape is not None
        fake = ShapeTheory(
            name=phase,
            isl=isl,
            osl=osl,
            adapted=False,
            original_isl=isl,
            original_osl=osl,
            tpot_floor_ms=shape.tpot_floor_ms,
            ttft_floor_ms=shape.ttft_floor_ms * isl / shape.isl,
            e2e_floor_ms=0,
            fits=None,
        )
        c = eta_c if at_op else 1
        spec = shape_spec(isl, osl, 1, slo)
        e2e = estimate_e2e_ms(theory, fake, c, slowdown)
        if at_op:
            rounds = max(
                SHAPE_MIN_ROUNDS, min(SWEEP_ROUNDS, int(SHAPE_POINT_TARGET_S * 1000 / e2e))
            )
            spec_eta = replace(spec, concurrency=c, request_count=rounds * c)
        else:
            spec_eta = spec
        eta = estimate_point_s(spec_eta, e2e, [], False)
        return PlannedPoint(phase, spec, eta, at_operating_point=at_op)

    for isl in axis_lengths(theory.max_model_len, SHAPE_FIXED_OSL):
        out += [
            point("shape_prompt", isl, SHAPE_FIXED_OSL, False),
            point("shape_prompt", isl, SHAPE_FIXED_OSL, True),
        ]
    for osl in axis_lengths(theory.max_model_len, SHAPE_FIXED_ISL):
        out += [
            point("shape_output", SHAPE_FIXED_ISL, osl, False),
            point("shape_output", SHAPE_FIXED_ISL, osl, True),
        ]
    for s in theory.shapes:  # already fitted to the context window
        if s.name not in ("baseline", primary.name):
            out.append(point("shape_map", s.isl, s.osl, True))
    return out


def _trim(points: list[PlannedPoint], budget_s: float) -> list[int]:
    """Drop the highest concurrency levels until the plan fits, keeping at least 1 to 8."""
    dropped: list[int] = []
    while sum(p.eta_s for p in points) > budget_s:
        sweep = [p for p in points if p.phase == "concurrency"]
        if len(sweep) <= 4:
            break
        top = max(sweep, key=lambda p: p.spec.concurrency or 0)
        points.remove(top)
        dropped.append(top.spec.concurrency or 0)
    return sorted(dropped)

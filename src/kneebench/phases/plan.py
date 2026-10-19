"""Turn the requested phases into concrete AIPerf points, with an ETA."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from kneebench.domain import PRIMARY_SHAPE, ProductType, Slo
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
SWEEP_LENGTH_VARIATION = 0.05
# Note 25 section 4: random arrivals below, near and past capacity, then bursty ones.
POISSON_FACTORS = (0.25, 0.5, 0.75, 0.9, 1.1)
BURSTY_FACTORS = (0.75, 0.9)
BURSTINESS = 0.5  # gamma shape below 1 clusters arrivals; 1.0 is Poisson
# Long enough to see whether first-token time keeps growing (run-length dependence).
RATE_DURATION_S = 90.0
# Open-loop arrivals need room to queue; past this, the client is the limit.
RATE_CONCURRENCY_CAP = 2048
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
        duration_s=RATE_DURATION_S,
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
    if "rate" in phases and "concurrency" not in phases:
        raise ValueError("the request-rate sweep needs the concurrency sweep to find capacity")
    if "concurrency" in phases:
        shape = theory.shape(PRIMARY_SHAPE[product]) or base_shape
        levels = concurrency_levels(theory, shape, max_concurrency, max_num_seqs)
        sweep = [(c, sweep_spec(shape, c, slo or Slo(None, None))) for c in levels]
        for _, spec in sweep:
            add("concurrency", spec, shape)
        if "rate" in phases:
            for arrival, factors in (("poisson", POISSON_FACTORS), ("gamma", BURSTY_FACTORS)):
                for factor in factors:
                    spec = rate_spec(shape, arrival, slo or Slo(None, None))
                    tail_s = estimate_e2e_ms(theory, shape, 64, slowdown) / 1000
                    eta = RATE_DURATION_S + tail_s + estimate_point_overhead_s([], False)
                    points.append(PlannedPoint("rate", spec, eta, rate_factor=factor))
        # Re-measuring the baseline at the end catches a GPU that changed mid-run (B06).
        add("baseline_end", replace(baseline_spec(theory), warmup_count=2), base_shape)
        if budget_s is not None:
            dropped = _trim(points, budget_s)

    return RunPlan(tuple(points), tuple(dropped))


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

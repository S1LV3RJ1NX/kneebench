"""Turn the requested phases into concrete AIPerf points, with an ETA."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

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
from kneebench.physics.floors import requests_that_fit

BASELINE_WARMUP = 10  # note 24: warm up, discarded
BASELINE_REPEATS = 30  # note 24: about 30 repeats give a median and a spread
# Real requests run slower than the floor; assume this until measurements exist.
UNMEASURED_SLOWDOWN = 2.0
# Note 25 section 2: N = 1, 2, 4 ... doubling up to 256 or until something breaks.
DEFAULT_MAX_CONCURRENCY = 256
# Rounds of requests per concurrency level: enough for a steady-state window (note 25).
SWEEP_ROUNDS = 4
SWEEP_MIN_REQUESTS = 16
# A level's window at full load must last 20 s (analysis/steady.py); aim for half again more,
# so short answers get more rounds instead of a throughput number with no steady state.
STEADY_TARGET_S = 30.0
SWEEP_MAX_ROUNDS = 40
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
# Lesson 43 section 2: SLO capacity is the highest rate that passes; bisect the gap between
# the best passing and the first failing rate, until it is under 5% of capacity.
REFINE_POINTS = 2
REFINE_MIN_GAP = 0.05
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
# Lesson 44 section 2: 32 streaming chats with short prompts and long answers, one long
# prompt every 10 seconds, the first at 5 s; once without the long prompts to compare.
INTERFERENCE_PHASES = ("interference_base", "interference")
INTERFERENCE_STREAMS = 32
INTERFERENCE_STREAM_ISL = 128
INTERFERENCE_AT_S = (5, 15, 25, 35)
INTERFERENCE_ISL = 8192
INTERFERENCE_OSL = 16
# Answers must still be streaming after the last long prompt.
INTERFERENCE_WINDOW_S = 45.0
# Lessons 47-48: the same load with prompts that never repeat, one shared system prompt, and
# six-turn conversations that resend their history. Cache churn (lesson 49) is estimated, not
# run: proving it needs every prompt to come back after the cache filled, over 1,000 requests.
REUSE_PHASES = ("reuse_control", "reuse_shared", "reuse_turns")
REUSE_PREFIX = 1500  # lesson 47 section 4: 1,500 shared tokens
REUSE_QUESTION = 100
REUSE_OSL = 128
REUSE_SYSTEM = 1000  # lesson 48 section 3: 1,000 system, 100 question, 300 answer, 6 turns
REUSE_TURN_OSL = 300
REUSE_TURNS = 6
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
    # Rate points placed between the best passing and the first failing rate (lesson 43).
    refine: bool = False


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


def sweep_rounds(e2e_ms: float | None) -> int:
    """Back-to-back rounds per user so the window at full load outlasts STEADY_TARGET_S.

    In a closed loop the window runs from full load to the last request issued, about one
    round short of the whole point.
    """
    if not e2e_ms or e2e_ms <= 0:
        return SWEEP_ROUNDS
    need = math.ceil(STEADY_TARGET_S * 1000 / e2e_ms) + 1
    return max(SWEEP_ROUNDS, min(need, SWEEP_MAX_ROUNDS))


def sweep_spec(
    shape: ShapeTheory, concurrency: int, slo: Slo, e2e_ms: float | None = None
) -> PointSpec:
    return PointSpec(
        isl=shape.isl,
        osl=shape.osl,
        mode=LoadMode.CONCURRENCY,
        concurrency=concurrency,
        isl_stddev=round(shape.isl * SWEEP_LENGTH_VARIATION, 1),
        osl_stddev=round(shape.osl * SWEEP_LENGTH_VARIATION),
        request_count=max(sweep_rounds(e2e_ms) * concurrency, SWEEP_MIN_REQUESTS),
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


def refine_rate(results: list[tuple[float, bool]], capacity: float | None) -> float | None:
    """The next rate to try: halfway between the best passing and the first failing rate.

    None when nothing failed above the best passing rate, or the gap is already narrow.
    """
    passing = [r for r, passed in results if passed]
    low = max(passing) if passing else 0.0
    above = [r for r, passed in results if not passed and r > low]
    if not above:
        return None
    high = min(above)
    if capacity and (high - low) / capacity < REFINE_MIN_GAP:
        return None
    return round((low + high) / 2, 3)


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
    """Every decode step reads the weights once plus each user's cache (note 26 section 2)."""
    weights = theory.model_weight_bytes
    kv_one = (shape.isl + shape.osl / 2) * theory.kv_bytes_per_token
    # The floor already includes one user's cache; each extra user adds their own.
    tpot = shape.tpot_floor_ms * slowdown * (weights + concurrency * kv_one) / (weights + kv_one)
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
    min_concurrency: int = 1,
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
    for needs_sweep in ("rate", "shape", "interference", "reuse"):
        if needs_sweep in phases and "concurrency" not in phases:
            raise ValueError(
                f"the {needs_sweep} sweep needs the concurrency sweep to find capacity first"
            )
    if "concurrency" in phases:
        primary = primary_shape_name(product, tuple(s.name for s in theory.shapes))
        shape = theory.shape(primary) or base_shape
        levels = concurrency_levels(theory, shape, max_concurrency, max_num_seqs)
        # A verification run only sweeps the region a change can move (lesson 55, step 5).
        levels = [c for c in levels if c >= min_concurrency] or levels[-1:]
        sweep = [
            (
                c,
                sweep_spec(
                    shape, c, slo or Slo(None, None), estimate_e2e_ms(theory, shape, c, slowdown)
                ),
            )
            for c in levels
        ]
        for _, spec in sweep:
            add("concurrency", spec, shape)
        if "rate" in phases:
            tail_s = estimate_e2e_ms(theory, shape, 64, slowdown) / 1000
            eta = RATE_WARMUP_S + RATE_DURATION_S + tail_s + estimate_point_overhead_s([], False)
            for arrival, factors in (("poisson", POISSON_FACTORS), ("gamma", BURSTY_FACTORS)):
                for factor in factors:
                    spec = rate_spec(shape, arrival, slo or Slo(None, None))
                    points.append(PlannedPoint("rate", spec, eta, rate_factor=factor))
                if arrival == "poisson":
                    points.extend(
                        PlannedPoint(
                            "rate",
                            rate_spec(shape, "poisson", slo or Slo(None, None)),
                            eta,
                            refine=True,
                        )
                        for _ in range(REFINE_POINTS)
                    )
        if "shape" in phases:
            points.extend(_shape_points(theory, shape, slo or Slo(None, None), slowdown))
        if "interference" in phases:
            points.extend(_interference_points(theory, slowdown))
        if "reuse" in phases:
            points.extend(_reuse_points(theory, slo or Slo(None, None), slowdown))
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


def interference_schedules(
    theory: Theory,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...], int, int]:
    """The background streams alone, and with long prompts injected (lesson 44 section 2)."""
    base = theory.shape("baseline")
    assert base is not None
    budget = context_budget(theory.max_model_len or 32_768)
    streams = INTERFERENCE_STREAMS
    # Size answers with the fastest per-token time physics allows, so even a fast server is
    # still streaming when the last long prompt arrives.
    weights = theory.model_weight_bytes
    kv_one = INTERFERENCE_STREAM_ISL * theory.kv_bytes_per_token
    tpot_ms = base.tpot_floor_ms * (weights + streams * kv_one) / (weights + kv_one)
    osl = math.ceil(INTERFERENCE_WINDOW_S * 1000 / tpot_ms)
    osl = max(256, min(osl, budget - INTERFERENCE_STREAM_ISL))
    if theory.pool_tokens:
        # Evictions would muddy the stalls, so the streams fill at most 80% of the cache.
        fit = requests_that_fit(theory.pool_tokens, INTERFERENCE_STREAM_ISL, osl, theory.block_size)
        streams = max(4, min(streams, int(fit * 0.8)))
    inject_isl = min(INTERFERENCE_ISL, (budget - INTERFERENCE_OSL) // 256 * 256)
    background = tuple((0, INTERFERENCE_STREAM_ISL, osl) for _ in range(streams))
    injected = tuple((at * 1000, inject_isl, INTERFERENCE_OSL) for at in INTERFERENCE_AT_S)
    return background, background + injected, osl, inject_isl


def _reuse_points(theory: Theory, slo: Slo, slowdown: float) -> list[PlannedPoint]:
    """Run at the operating point the concurrency sweep finds; see REUSE_PHASES."""
    base = theory.shape("baseline")
    assert base is not None
    budget = context_budget(theory.max_model_len or 32_768)
    common: dict[str, Any] = {
        "mode": LoadMode.CONCURRENCY,
        "concurrency": 1,
        "request_count": SWEEP_MIN_REQUESTS,
        "duration_s": POINT_DURATION_CAP_S,
        "slice_duration_s": SLICE_S,
        "goodput": slo.goodput(),
    }
    single = REUSE_PREFIX + REUSE_QUESTION
    specs: list[tuple[str, PointSpec]] = []
    if single + REUSE_OSL <= budget:
        specs.append(("reuse_control", PointSpec(isl=single, osl=REUSE_OSL, **common)))
        specs.append(
            (
                "reuse_shared",
                PointSpec(
                    isl=REUSE_QUESTION,
                    osl=REUSE_OSL,
                    cache_bust=False,
                    prefix_pool=1,
                    prefix_len=REUSE_PREFIX,
                    random_seed=43,
                    **common,
                ),
            )
        )
    turns = REUSE_TURNS
    while turns > 2 and REUSE_SYSTEM + turns * (REUSE_QUESTION + REUSE_TURN_OSL) > budget:
        turns -= 1
    if REUSE_SYSTEM + turns * (REUSE_QUESTION + REUSE_TURN_OSL) <= budget:
        specs.append(
            (
                "reuse_turns",
                PointSpec(
                    isl=REUSE_QUESTION,
                    osl=REUSE_TURN_OSL,
                    cache_bust=False,
                    system_len=REUSE_SYSTEM,
                    turns=turns,
                    random_seed=45,
                    **{**common, "request_count": SWEEP_MIN_REQUESTS * turns},
                ),
            )
        )
    # Each user runs a few requests back to back at the operating point (assume 32 users),
    # with the answer time stretched by load.
    out = []
    for phase, spec in specs:
        answer_s = estimate_e2e_ms(theory, base, 32, slowdown) * spec.osl / base.osl / 1000
        rounds = 2 * (spec.turns or 1) if spec.turns else SWEEP_ROUNDS
        eta = rounds * answer_s + estimate_point_overhead_s([], first_launch=False)
        out.append(PlannedPoint(phase, spec, eta, at_operating_point=True))
    return out


def _interference_points(theory: Theory, slowdown: float) -> list[PlannedPoint]:
    base = theory.shape("baseline")
    assert base is not None
    alone, mixed, osl, _ = interference_schedules(theory)
    # The streams run as long as their answers; the planner's usual slowdown applies.
    seconds = osl * base.tpot_floor_ms * slowdown / 1000
    eta = seconds + estimate_point_overhead_s([], first_launch=False)
    return [
        PlannedPoint(
            phase,
            PointSpec(
                isl=INTERFERENCE_STREAM_ISL,
                osl=osl,
                mode=LoadMode.SCHEDULE,
                schedule=schedule,
                slice_duration_s=SLICE_S,
            ),
            eta,
        )
        for phase, schedule in zip(INTERFERENCE_PHASES, (alone, mixed), strict=True)
    ]


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

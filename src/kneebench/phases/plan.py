"""Turn the requested phases into concrete AIPerf points, with an ETA."""

from __future__ import annotations

from dataclasses import dataclass

from kneebench.loadgen.aiperf.overhead import estimate_point_overhead_s
from kneebench.loadgen.base import LoadMode, PointSpec
from kneebench.phases.theory import Theory

BASELINE_WARMUP = 10  # note 24: warm up, discarded
BASELINE_REPEATS = 30  # note 24: about 30 repeats give a median and a spread
# Real requests run slower than the floor; assume this until measurements exist.
UNMEASURED_SLOWDOWN = 2.0


@dataclass(frozen=True)
class PlannedPoint:
    phase: str
    spec: PointSpec
    eta_s: float


@dataclass(frozen=True)
class RunPlan:
    points: tuple[PlannedPoint, ...]

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
        slice_duration_s=10.0,
    )


def estimate_point_s(
    spec: PointSpec, e2e_ms: float, overhead_observed: list[float], first: bool
) -> float:
    concurrency = spec.concurrency or 1
    requests = (spec.request_count or 0) + (spec.warmup_count or 0)
    traffic = requests * e2e_ms / 1000 / concurrency
    if spec.duration_s:
        traffic = min(traffic, spec.duration_s) if spec.request_count else spec.duration_s
    return traffic + estimate_point_overhead_s(overhead_observed, first_launch=first)


def plan_run(theory: Theory, phases: tuple[str, ...], first_launch: bool = True) -> RunPlan:
    points: list[PlannedPoint] = []
    if "baseline" in phases:
        spec = baseline_spec(theory)
        shape = theory.shape("baseline")
        assert shape is not None
        e2e = shape.e2e_floor_ms * UNMEASURED_SLOWDOWN
        points.append(PlannedPoint("baseline", spec, estimate_point_s(spec, e2e, [], first_launch)))
    return RunPlan(tuple(points))

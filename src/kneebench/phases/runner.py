"""Execute a run: record the setup, run each planned point, parse and store it."""

from __future__ import annotations

import platform
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from sqlite3 import Connection

from kneebench import __version__
from kneebench.analysis.context import point_from_dict, ttft_offset
from kneebench.analysis.shape import comparison_load
from kneebench.analysis.sweep import (
    SATURATION_GAIN,
    SweepPoint,
    meets,
    steps,
    summarize,
)
from kneebench.detect.server import ModelMetadata, ServedModel, ServerSnapshot
from kneebench.domain import ServerSetup, Slo, Workload
from kneebench.loadgen.aiperf.overhead import point_overhead_s
from kneebench.loadgen.aiperf.parse import parse_point
from kneebench.loadgen.base import Endpoint, LoadGenerator, PointResult
from kneebench.phases.plan import (
    INTERFERENCE_PHASES,
    SHAPE_MIN_ROUNDS,
    SHAPE_PHASES,
    SHAPE_POINT_TARGET_S,
    SWEEP_MAX_WARMUP,
    SWEEP_MIN_REQUESTS,
    SWEEP_ROUNDS,
    PlannedPoint,
    RunPlan,
    sweep_rounds,
)
from kneebench.phases.theory import Theory
from kneebench.store import repo

# Stop a sweep early: past these, higher levels only repeat the failure.
MAX_ERROR_RATE = 0.10
GATEWAY_MIN_ERRORS = 3
GATEWAY_ERROR_SHARE = 0.05


@dataclass(frozen=True)
class RunInputs:
    endpoint: Endpoint
    snapshot: ServerSnapshot
    served: ServedModel
    metadata: ModelMetadata | None
    setup: ServerSetup
    workload: Workload
    theory: Theory
    label: str | None = None


@dataclass(frozen=True)
class PointEvent:
    index: int
    total: int
    planned: PlannedPoint
    result: PointResult | None = None
    skipped: str | None = None
    summary: SweepPoint | None = None


def client_info() -> dict[str, str]:
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }


def sweep_stop_reason(curve: list[SweepPoint], slo: Slo) -> str | None:
    """Why the remaining concurrency levels can be skipped, or None to keep going."""
    last = curve[-1]
    if last.gateway_errors >= max(GATEWAY_MIN_ERRORS, GATEWAY_ERROR_SHARE * last.requests):
        return "gateway_errors"
    if last.error_rate > MAX_ERROR_RATE:
        return "errors"
    recent = steps(curve)[-2:]
    saturated = len(recent) == 2 and all(s.gain < SATURATION_GAIN for s in recent)
    if saturated and not meets(last, slo):
        return "walls_found"
    return None


def _size_by_measured_e2e(planned: PlannedPoint, e2e_ms: float | None) -> PlannedPoint:
    """Add rounds when the last level's answers were shorter than the plan assumed.

    Answers only get slower as users are added, so the previous level's time is a safe floor.
    """
    c = planned.spec.concurrency or 1
    needed = sweep_rounds(e2e_ms) * c
    if e2e_ms is None or needed <= (planned.spec.request_count or 0):
        return planned
    return replace(planned, spec=replace(planned.spec, request_count=needed))


def _resolve_operating_point(
    planned: PlannedPoint, curve: list[SweepPoint], slo: Slo
) -> tuple[PlannedPoint, str | None]:
    op = comparison_load(curve, slo)
    if op is None:
        return planned, "no_operating_point"
    c = op.concurrency
    # Long outputs at high load take minutes per request: size the point by time, keeping
    # at least two rounds so there is a steady-state window.
    e2e_s = ((op.tpot.get("p50") or 10.0) * planned.spec.osl) / 1000
    rounds = max(SHAPE_MIN_ROUNDS, min(SWEEP_ROUNDS, int(SHAPE_POINT_TARGET_S / e2e_s)))
    spec = replace(
        planned.spec,
        concurrency=c,
        request_count=max(rounds * c, SWEEP_MIN_REQUESTS),
        warmup_count=min(c, SWEEP_MAX_WARMUP),
    )
    return replace(planned, spec=spec), None


def _resolve_rate(
    planned: PlannedPoint, curve: list[SweepPoint], stopped: str | None
) -> tuple[PlannedPoint, str | None]:
    """Set a rate point's arrival rate from the capacity the concurrency sweep measured."""
    if stopped:
        return planned, stopped
    peak = max((p.requests_s for p in curve if p.requests_s), default=None)
    if not peak or planned.rate_factor is None:
        return planned, "no_capacity_measured"
    rate = round(planned.rate_factor * peak, 3)
    return replace(planned, spec=replace(planned.spec, request_rate=rate)), None


def execute_run(
    conn: Connection,
    inputs: RunInputs,
    plan: RunPlan,
    loadgen: LoadGenerator,
    data_dir: Path,
    on_event: Callable[[PointEvent], None] | None = None,
    on_created: Callable[[int], None] | None = None,
    cancel: threading.Event | None = None,
) -> int:
    endpoint_id = repo.upsert_endpoint(conn, inputs.snapshot.base_url)
    snapshot_id = repo.insert_snapshot(
        conn, endpoint_id, inputs.snapshot, inputs.served, inputs.metadata
    )
    setup_id = repo.insert_setup(conn, endpoint_id, snapshot_id, inputs.setup)
    workload_id = repo.insert_workload(conn, inputs.workload)
    run_id = repo.create_run(
        conn,
        endpoint_id=endpoint_id,
        setup_id=setup_id,
        workload_id=workload_id,
        kneebench_version=__version__,
        aiperf_version=loadgen.version(),
        vllm_version=inputs.snapshot.vllm_version,
        client=client_info(),
        network=asdict(inputs.snapshot.network) if inputs.snapshot.network else None,
        theory=inputs.theory.to_json(),
        label=inputs.label,
    )
    repo.set_planned_points(conn, run_id, len(plan.points))
    if on_created:
        on_created(run_id)

    status = "done"
    error: str | None = None
    curve: list[SweepPoint] = []
    rtt = inputs.snapshot.network.http_rtt_ms if inputs.snapshot.network else None
    offset = rtt or 0.0
    last_e2e_ms: float | None = None
    stopped: str | None = None
    rate_stopped: str | None = None
    total = len(plan.points)
    try:
        for i, planned in enumerate(plan.points):
            if cancel is not None and cancel.is_set():
                status = "cancelled"
                break
            if planned.phase == "concurrency" and stopped:
                if on_event:
                    on_event(PointEvent(i, total, planned, skipped=stopped))
                continue
            if planned.phase == "concurrency":
                planned = _size_by_measured_e2e(planned, last_e2e_ms)
            if planned.at_operating_point:
                planned, op_skip = _resolve_operating_point(planned, curve, inputs.workload.slo)
                if op_skip:
                    if on_event:
                        on_event(PointEvent(i, total, planned, skipped=op_skip))
                    continue
            if planned.phase == "rate":
                planned, rate_skip = _resolve_rate(planned, curve, rate_stopped)
                if rate_skip:
                    if on_event:
                        on_event(PointEvent(i, total, planned, skipped=rate_skip))
                    continue
            if on_event:
                on_event(PointEvent(i, total, planned))
            point_id = repo.create_point(conn, run_id, i, planned.phase, planned.spec)
            artifact_dir = data_dir / "runs" / str(run_id) / f"{i:03d}-{planned.phase}"
            result = loadgen.run_point(inputs.endpoint, planned.spec, artifact_dir, cancel=cancel)
            if cancel is not None and cancel.is_set():
                repo.finish_point(conn, point_id, result, None, None, stop_reason="cancelled")
                status = "cancelled"
                break
            parsed = parse_point(result.artifact_dir) if result.ok else None
            overhead = point_overhead_s(result.wall_time_s, result.artifact_dir)
            repo.finish_point(conn, point_id, result, parsed, overhead)

            summary = None
            if planned.phase == "baseline" and result.ok:
                base_shape = inputs.theory.shape("baseline")
                offset = ttft_offset(
                    rtt,
                    point_from_dict(repo.point_dict(conn, point_id)),
                    base_shape.ttft_floor_ms if base_shape else None,
                )
            if planned.phase == "concurrency" and result.ok:
                view = point_from_dict(repo.point_dict(conn, point_id))
                last_e2e_ms = view.m("request_latency", "p50") or last_e2e_ms
                summary = summarize(view, offset)
                curve.append(summary)
                stopped = sweep_stop_reason(curve, inputs.workload.slo)
                if stopped:
                    repo.set_stop_reason(conn, point_id, stopped)
            if on_event:
                on_event(PointEvent(i, total, planned, result, summary=summary))
            if not result.ok and planned.phase == "rate":
                rate_stopped = "point_failed"
                repo.set_stop_reason(conn, point_id, rate_stopped)
                continue
            if not result.ok and planned.phase in (*SHAPE_PHASES, *INTERFERENCE_PHASES):
                repo.set_stop_reason(conn, point_id, "point_failed")
                continue
            if not result.ok:
                if planned.phase != "concurrency":
                    status = "failed"
                    detail = (result.stderr_tail or result.stdout_tail).strip().splitlines()[-3:]
                    error = (
                        f"The {planned.phase} point failed (AIPerf exit code "
                        f"{result.exit_code}). " + " ".join(detail)
                    ).strip()
                    break
                stopped = stopped or "point_failed"
                repo.set_stop_reason(conn, point_id, stopped)
    except KeyboardInterrupt:
        status = "cancelled"
        raise
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        repo.finish_run(conn, run_id, status, error)
    return run_id

"""Execute a run: record the setup, run each planned point, parse and store it."""

from __future__ import annotations

import platform
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from sqlite3 import Connection

from kneebench import __version__
from kneebench.analysis.context import point_from_dict
from kneebench.analysis.sweep import SATURATION_GAIN, SweepPoint, meets, steps, summarize
from kneebench.detect.server import ModelMetadata, ServedModel, ServerSnapshot
from kneebench.domain import ServerSetup, Slo, Workload
from kneebench.loadgen.aiperf.overhead import point_overhead_s
from kneebench.loadgen.aiperf.parse import parse_point
from kneebench.loadgen.base import Endpoint, LoadGenerator, PointResult
from kneebench.phases.plan import PlannedPoint, RunPlan
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
    stopped: str | None = None
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
            if planned.phase == "concurrency" and result.ok:
                rtt = inputs.snapshot.network.http_rtt_ms if inputs.snapshot.network else None
                summary = summarize(point_from_dict(repo.point_dict(conn, point_id)), rtt or 0.0)
                curve.append(summary)
                stopped = sweep_stop_reason(curve, inputs.workload.slo)
                if stopped:
                    repo.set_stop_reason(conn, point_id, stopped)
            if on_event:
                on_event(PointEvent(i, total, planned, result, summary=summary))
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

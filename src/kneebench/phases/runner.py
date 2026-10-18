"""Execute a run: record the setup, run each planned point, parse and store it."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from sqlite3 import Connection

from kneebench import __version__
from kneebench.detect.server import ModelMetadata, ServedModel, ServerSnapshot
from kneebench.domain import ServerSetup, Workload
from kneebench.loadgen.aiperf.overhead import point_overhead_s
from kneebench.loadgen.aiperf.parse import parse_point
from kneebench.loadgen.base import Endpoint, LoadGenerator, PointResult
from kneebench.phases.plan import PlannedPoint, RunPlan
from kneebench.phases.theory import Theory
from kneebench.store import repo


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


def client_info() -> dict[str, str]:
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "host_hint": "local",
    }


def execute_run(
    conn: Connection,
    inputs: RunInputs,
    plan: RunPlan,
    loadgen: LoadGenerator,
    data_dir: Path,
    on_event: Callable[[PointEvent], None] | None = None,
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

    status = "done"
    try:
        for i, planned in enumerate(plan.points):
            if on_event:
                on_event(PointEvent(i, len(plan.points), planned))
            point_id = repo.create_point(conn, run_id, i, planned.phase, planned.spec)
            artifact_dir = data_dir / "runs" / str(run_id) / f"{i:03d}-{planned.phase}"
            result = loadgen.run_point(inputs.endpoint, planned.spec, artifact_dir)
            parsed = parse_point(result.artifact_dir) if result.ok else None
            overhead = point_overhead_s(result.wall_time_s, result.artifact_dir)
            repo.finish_point(conn, point_id, result, parsed, overhead)
            if on_event:
                on_event(PointEvent(i, len(plan.points), planned, result))
            if not result.ok:
                status = "failed"
                break
    except KeyboardInterrupt:
        status = "cancelled"
        raise
    finally:
        repo.finish_run(conn, run_id, status)
    return run_id

"""Read and write kneebench records. Plain functions over a sqlite3 connection."""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection, Row
from typing import Any

from kneebench.detect.server import ModelMetadata, ServedModel, ServerSnapshot
from kneebench.domain import ServerSetup, Workload
from kneebench.loadgen.aiperf.parse import (
    RECORDS_FILE,
    SERVER_METRICS_FILE,
    SUMMARY_FILE,
    TIMESLICES_FILE,
    ParsedPoint,
    stream_summary,
    turn_summary,
)
from kneebench.loadgen.base import LoadMode, PointResult, PointSpec

# Raw artifacts copied into SQLite so reports survive a deleted artifacts directory.
STORED_ARTIFACTS = (
    SUMMARY_FILE,
    RECORDS_FILE,
    TIMESLICES_FILE,
    SERVER_METRICS_FILE,
    "phase_manifest.json",
    "aiperf_config.yaml",
    "kneebench_aiperf_stdout.log",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def upsert_endpoint(conn: Connection, base_url: str, name: str | None = None) -> int:
    row = conn.execute("SELECT id FROM endpoints WHERE base_url = ?", (base_url,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO endpoints (name, base_url, created_at) VALUES (?, ?, ?)",
        (name or base_url, base_url, now()),
    )
    return int(cur.lastrowid or 0)


def insert_snapshot(
    conn: Connection,
    endpoint_id: int,
    snap: ServerSnapshot,
    served: ServedModel | None,
    metadata: ModelMetadata | None,
) -> int:
    cur = conn.execute(
        """INSERT INTO snapshots (endpoint_id, taken_at, vllm_version, served_model,
           max_model_len, hf_model_id, models_json, cache_config_json, hf_config_json,
           safetensors_json, metrics_text_z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            endpoint_id,
            now(),
            snap.vllm_version,
            served.id if served else None,
            served.max_model_len if served else None,
            metadata.hf_id if metadata else (served.hf_id if served else None),
            _dumps([asdict(m) for m in snap.models]),
            _dumps(asdict(snap.cache)) if snap.cache else None,
            _dumps(metadata.config) if metadata else None,
            _dumps(metadata.safetensors_params)
            if metadata and metadata.safetensors_params
            else None,
            zlib.compress(snap.metrics_text.encode()) if snap.metrics_text else None,
        ),
    )
    return int(cur.lastrowid or 0)


def insert_setup(conn: Connection, endpoint_id: int, snapshot_id: int, setup: ServerSetup) -> int:
    cur = conn.execute(
        """INSERT INTO setups (endpoint_id, snapshot_id, gpu_json, gpu_count, max_num_seqs,
           enforce_eager, source_json) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            endpoint_id,
            snapshot_id,
            _dumps(asdict(setup.gpu)),
            setup.gpu_count,
            setup.max_num_seqs,
            None if setup.enforce_eager is None else int(setup.enforce_eager),
            _dumps(setup.sources),
        ),
    )
    return int(cur.lastrowid or 0)


def insert_workload(conn: Connection, workload: Workload) -> int:
    cur = conn.execute(
        """INSERT INTO workloads (product_type, shapes_json, slo_json, time_budget_s,
           gpu_hourly_usd) VALUES (?, ?, ?, ?, ?)""",
        (
            workload.product.value,
            _dumps([asdict(s) for s in workload.shapes]),
            _dumps(asdict(workload.slo)),
            workload.time_budget_s,
            workload.gpu_hourly_usd,
        ),
    )
    return int(cur.lastrowid or 0)


def create_run(
    conn: Connection,
    *,
    endpoint_id: int,
    setup_id: int,
    workload_id: int,
    kneebench_version: str,
    aiperf_version: str,
    vllm_version: str | None,
    client: dict[str, Any],
    network: dict[str, Any] | None,
    theory: dict[str, Any] | None,
    label: str | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO runs (endpoint_id, setup_id, workload_id, label, status, started_at,
           kneebench_version, aiperf_version, vllm_version, client_json, network_json,
           theory_json) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)""",
        (
            endpoint_id,
            setup_id,
            workload_id,
            label,
            now(),
            kneebench_version,
            aiperf_version,
            vllm_version,
            _dumps(client),
            _dumps(network) if network else None,
            _dumps(theory) if theory else None,
        ),
    )
    return int(cur.lastrowid or 0)


def finish_run(conn: Connection, run_id: int, status: str, error: str | None = None) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
        (status, now(), error, run_id),
    )


def set_planned_points(conn: Connection, run_id: int, planned: int) -> None:
    conn.execute("UPDATE runs SET planned_points = ? WHERE id = ?", (planned, run_id))


def mark_interrupted(conn: Connection, live_run_ids: set[int]) -> list[int]:
    """Runs still marked running whose process is gone (killed, crashed, machine slept)."""
    stale = [
        int(r["id"])
        for r in conn.execute("SELECT id FROM runs WHERE status = 'running'")
        if int(r["id"]) not in live_run_ids
    ]
    for run_id in stale:
        conn.execute(
            "UPDATE points SET status = 'interrupted' WHERE run_id = ? AND status = 'running'",
            (run_id,),
        )
        finish_run(
            conn,
            run_id,
            "interrupted",
            "The process running this benchmark stopped before it finished.",
        )
    return stale


def progress(conn: Connection, run_id: int) -> dict[str, Any]:
    """How far a run is, from the database alone (works for CLI and UI runs alike)."""
    run = get_run(conn, run_id)
    if run is None:
        raise KeyError(f"no run with id {run_id}")
    rows = list(
        conn.execute(
            "SELECT seq, phase, concurrency, status FROM points WHERE run_id = ? ORDER BY seq",
            (run_id,),
        )
    )
    current = next((r for r in rows if r["status"] == "running"), None)
    last_seq = rows[-1]["seq"] if rows else -1
    finished_points = sum(1 for r in rows if r["status"] != "running")
    # Points skipped by an early stop leave gaps in seq; count them as done too.
    done = max(finished_points, last_seq + (0 if current else 1))
    return {
        "status": run["status"],
        "error": run["error"],
        "total": run["planned_points"] or len(rows),
        "done": done,
        "current": f"{current['phase']}, {current['concurrency']} in flight" if current else None,
    }


def delete_run(conn: Connection, run_id: int) -> None:
    """Delete a run and everything that only it uses. Artifacts on disk are the caller's job."""
    run = get_run(conn, run_id)
    if run is None:
        raise KeyError(f"no run with id {run_id}")
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM points WHERE run_id = ?", (run_id,))  # cascades to children
        conn.execute("DELETE FROM findings WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        setup = conn.execute(
            "SELECT snapshot_id FROM setups WHERE id = ?", (run["setup_id"],)
        ).fetchone()
        for table, key, value in (
            ("workloads", "workload_id", run["workload_id"]),
            ("setups", "setup_id", run["setup_id"]),
        ):
            if not conn.execute(f"SELECT 1 FROM runs WHERE {key} = ?", (value,)).fetchone():
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (value,))
        if (
            setup
            and not conn.execute(
                "SELECT 1 FROM setups WHERE snapshot_id = ?", (setup["snapshot_id"],)
            ).fetchone()
        ):
            conn.execute("DELETE FROM snapshots WHERE id = ?", (setup["snapshot_id"],))
        if (
            not conn.execute(
                "SELECT 1 FROM runs WHERE endpoint_id = ?", (run["endpoint_id"],)
            ).fetchone()
            and not conn.execute(
                "SELECT 1 FROM snapshots WHERE endpoint_id = ?", (run["endpoint_id"],)
            ).fetchone()
            and not conn.execute(
                "SELECT 1 FROM experiments WHERE endpoint_id = ?", (run["endpoint_id"],)
            ).fetchone()
        ):
            conn.execute("DELETE FROM endpoints WHERE id = ?", (run["endpoint_id"],))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def create_point(conn: Connection, run_id: int, seq: int, phase: str, spec: PointSpec) -> int:
    cur = conn.execute(
        """INSERT INTO points (run_id, seq, phase, spec_json, isl, osl, mode, concurrency,
           request_rate, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
        (
            run_id,
            seq,
            phase,
            _dumps(asdict(spec)),
            spec.isl,
            spec.osl,
            spec.mode.value,
            spec.concurrency,
            spec.request_rate,
            now(),
        ),
    )
    return int(cur.lastrowid or 0)


def finish_point(
    conn: Connection,
    point_id: int,
    result: PointResult,
    parsed: ParsedPoint | None,
    overhead_s: float | None,
    stop_reason: str | None = None,
) -> None:
    conn.execute("BEGIN")
    try:
        conn.execute(
            """UPDATE points SET status = ?, stop_reason = ?, exit_code = ?, argv_json = ?,
               artifact_dir = ?, wall_time_s = ?, overhead_s = ?, finished_at = ?
               WHERE id = ?""",
            (
                "done" if result.ok else "failed",
                stop_reason,
                result.exit_code,
                _dumps(list(result.argv_redacted)),
                str(result.artifact_dir),
                result.wall_time_s,
                overhead_s,
                now(),
                point_id,
            ),
        )
        if parsed is not None:
            _store_parsed(conn, point_id, parsed)
            series = {}
            if result.spec.mode is LoadMode.SCHEDULE:
                series["streams"] = stream_summary(parsed.profiling_requests())
            if result.spec.turns and result.spec.turns > 1:
                series["turns"] = turn_summary(parsed.profiling_requests())
            for name, summary in series.items():
                if summary is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO point_series VALUES (?, ?, ?)",
                        (point_id, name, _dumps(summary)),
                    )
        _store_artifacts(conn, point_id, result.artifact_dir)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def add_series(conn: Connection, point_id: int, name: str, data: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO point_series VALUES (?, ?, ?)", (point_id, name, _dumps(data))
    )


def _store_parsed(conn: Connection, point_id: int, parsed: ParsedPoint) -> None:
    conn.executemany(
        "INSERT INTO point_metrics (point_id, metric, stat, value, unit) VALUES (?, ?, ?, ?, ?)",
        [
            (point_id, metric, stat, value, parsed.units.get(metric))
            for metric, stats in parsed.aggregates.items()
            for stat, value in stats.items()
        ],
    )
    conn.executemany(
        """INSERT INTO requests (point_id, idx, phase, start_ns, end_ns, ttft_ms, tpot_ms,
           e2e_ms, isl, osl, error, error_code, connect_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                point_id,
                i,
                r.phase,
                r.start_ns,
                r.end_ns,
                r.ttft_ms,
                r.tpot_ms,
                r.e2e_ms,
                r.isl,
                r.osl,
                r.error,
                r.error_code,
                r.connect_ms,
            )
            for i, r in enumerate(parsed.requests)
        ],
    )
    conn.executemany(
        """INSERT INTO timeslices (point_id, slice_idx, start_ns, end_ns, metric, stat, value)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (point_id, i, ts.start_ns, ts.end_ns, metric, stat, value)
            for i, ts in enumerate(parsed.timeslices)
            for metric, stats in ts.metrics.items()
            for stat, value in stats.items()
        ],
    )
    conn.executemany(
        """INSERT INTO server_metrics (point_id, metric, labels_json, kind, stat, value)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (point_id, s.metric, _dumps(s.labels), s.kind, stat, value)
            for series in parsed.server.values()
            for s in series
            for stat, value in s.stats.items()
        ],
    )
    conn.executemany(
        "INSERT INTO warnings (point_id, source, code, message) VALUES (?, 'aiperf', ?, ?)",
        [(point_id, w.title, w.body) for w in parsed.warnings],
    )


def _store_artifacts(conn: Connection, point_id: int, artifact_dir: Path) -> None:
    for name in STORED_ARTIFACTS:
        path = artifact_dir / name
        if not path.exists():
            continue
        data = path.read_bytes()
        conn.execute(
            """INSERT OR REPLACE INTO artifacts (point_id, name, path, sha256, size, content_z)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                point_id,
                name,
                str(path),
                hashlib.sha256(data).hexdigest(),
                len(data),
                zlib.compress(data),
            ),
        )


def set_stop_reason(conn: Connection, point_id: int, reason: str) -> None:
    conn.execute("UPDATE points SET stop_reason = ? WHERE id = ?", (reason, point_id))


def save_findings(
    conn: Connection, run_id: int, findings: list[dict[str, Any]], rules_version: str
) -> None:
    conn.execute("DELETE FROM findings WHERE run_id = ?", (run_id,))
    conn.executemany(
        """INSERT INTO findings (run_id, rule_id, status, severity, data_json, rules_version)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (run_id, f["rule_id"], f["status"], f["severity"], _dumps(f["data"]), rules_version)
            for f in findings
        ],
    )


def correct_setup(conn: Connection, run_id: int, setup: ServerSetup, theory: dict[str, Any]) -> int:
    """Replace a run's hardware entry (e.g. the wrong GPU) and its derived theory."""
    run = get_run(conn, run_id)
    if run is None:
        raise KeyError(f"no run with id {run_id}")
    old = conn.execute("SELECT * FROM setups WHERE id = ?", (run["setup_id"],)).fetchone()
    setup_id = insert_setup(conn, old["endpoint_id"], old["snapshot_id"], setup)
    conn.execute(
        "UPDATE runs SET setup_id = ?, theory_json = ? WHERE id = ?",
        (setup_id, _dumps(theory), run_id),
    )
    return setup_id


# ---- reads ----


def get_run(conn: Connection, run_id: int) -> Row | None:
    row: Row | None = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return row


def list_runs(conn: Connection, limit: int = 50) -> list[Row]:
    return list(
        conn.execute(
            """SELECT runs.*, endpoints.base_url, snapshots.served_model FROM runs
           JOIN endpoints ON endpoints.id = runs.endpoint_id
           JOIN setups ON setups.id = runs.setup_id
           JOIN snapshots ON snapshots.id = setups.snapshot_id
           ORDER BY runs.id DESC LIMIT ?""",
            (limit,),
        )
    )


def point_dict(conn: Connection, point_id: int) -> dict[str, Any]:
    """One point with its metrics, server metrics, warnings and profiling requests."""
    p = dict(conn.execute("SELECT * FROM points WHERE id = ?", (point_id,)).fetchone())
    p["metrics"] = point_metrics(conn, point_id)
    p["server"] = point_server_metrics(conn, point_id)
    p["warnings"] = [
        dict(w)
        for w in conn.execute(
            "SELECT source, code, message FROM warnings WHERE point_id = ?", (point_id,)
        )
    ]
    p["series"] = {
        row["name"]: json.loads(row["data_json"])
        for row in conn.execute(
            "SELECT name, data_json FROM point_series WHERE point_id = ?", (point_id,)
        )
    }
    p["requests"] = [
        dict(r)
        for r in conn.execute(
            """SELECT start_ns, end_ns, ttft_ms, tpot_ms, e2e_ms, isl, osl, error,
               error_code, connect_ms FROM requests WHERE point_id = ? AND phase = 'profiling'
               ORDER BY idx""",
            (point_id,),
        )
    ]
    return p


def run_bundle(conn: Connection, run_id: int) -> dict[str, Any]:
    """Everything needed to analyze a run, as plain dicts."""
    run = get_run(conn, run_id)
    if run is None:
        raise KeyError(f"no run with id {run_id}")
    setup = conn.execute("SELECT * FROM setups WHERE id = ?", (run["setup_id"],)).fetchone()
    snapshot = conn.execute(
        "SELECT * FROM snapshots WHERE id = ?", (setup["snapshot_id"],)
    ).fetchone()
    workload = conn.execute(
        "SELECT * FROM workloads WHERE id = ?", (run["workload_id"],)
    ).fetchone()
    endpoint = conn.execute(
        "SELECT * FROM endpoints WHERE id = ?", (run["endpoint_id"],)
    ).fetchone()
    points = [
        point_dict(conn, row["id"])
        for row in conn.execute("SELECT id FROM points WHERE run_id = ? ORDER BY seq", (run_id,))
    ]
    return {
        "run": dict(run),
        "setup": dict(setup),
        "snapshot": dict(snapshot),
        "workload": dict(workload),
        "endpoint": dict(endpoint),
        "points": points,
    }


def point_metrics(conn: Connection, point_id: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in conn.execute(
        "SELECT metric, stat, value FROM point_metrics WHERE point_id = ?", (point_id,)
    ):
        out.setdefault(row["metric"], {})[row["stat"]] = row["value"]
    return out


def point_server_metrics(conn: Connection, point_id: int) -> dict[str, dict[str, float]]:
    """Server metric stats summed across label series (engines, replicas)."""
    out: dict[str, dict[str, float]] = {}
    for row in conn.execute(
        """SELECT metric, stat, kind, value FROM server_metrics WHERE point_id = ?""",
        (point_id,),
    ):
        stats = out.setdefault(row["metric"], {})
        if row["kind"] == "counter" or row["stat"] in ("total", "rate"):
            stats[row["stat"]] = stats.get(row["stat"], 0.0) + row["value"]
        else:
            stats[row["stat"]] = max(stats.get(row["stat"], row["value"]), row["value"])
    return out


def point_requests(conn: Connection, point_id: int, phase: str = "profiling") -> list[Row]:
    return list(
        conn.execute(
            "SELECT * FROM requests WHERE point_id = ? AND phase = ? ORDER BY idx",
            (point_id, phase),
        )
    )


# ---- guided experiments ----

EXPERIMENT_FIELDS = (
    "endpoint_id",
    "base_run_id",
    "rule_id",
    "title",
    "reason",
    "flag",
    "setting",
    "value",
    "expected",
    "risk",
)


def create_experiment(conn: Connection, **fields: Any) -> int:
    values = {k: fields.get(k) for k in EXPERIMENT_FIELDS}
    stamp = now()
    cur = conn.execute(
        f"""INSERT INTO experiments ({", ".join(values)}, status, created_at, updated_at)
           VALUES ({", ".join("?" * len(values))}, 'proposed', ?, ?)""",
        (*values.values(), stamp, stamp),
    )
    return int(cur.lastrowid or 0)


def get_experiment(conn: Connection, experiment_id: int) -> Row | None:
    row: Row | None = conn.execute(
        """SELECT experiments.*, endpoints.base_url, endpoints.deploy_style FROM experiments
           JOIN endpoints ON endpoints.id = experiments.endpoint_id WHERE experiments.id = ?""",
        (experiment_id,),
    ).fetchone()
    return row


def list_experiments(conn: Connection) -> list[Row]:
    return list(
        conn.execute(
            """SELECT experiments.*, endpoints.base_url FROM experiments
               JOIN endpoints ON endpoints.id = experiments.endpoint_id
               ORDER BY experiments.id DESC"""
        )
    )


def update_experiment(conn: Connection, experiment_id: int, **fields: Any) -> None:
    allowed = {"status", "check_json", "outcome", "verify_run_id"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    assignments = ", ".join(f"{k} = ?" for k in sets)
    conn.execute(
        f"UPDATE experiments SET {assignments}, updated_at = ? WHERE id = ?",
        (*sets.values(), now(), experiment_id),
    )


def experiment_for_verify_run(conn: Connection, run_id: int) -> Row | None:
    row: Row | None = conn.execute(
        "SELECT * FROM experiments WHERE verify_run_id = ?", (run_id,)
    ).fetchone()
    return row


def set_deploy_style(conn: Connection, endpoint_id: int, style: str) -> None:
    conn.execute("UPDATE endpoints SET deploy_style = ? WHERE id = ?", (style, endpoint_id))

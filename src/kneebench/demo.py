"""The demo run: a real benchmark shipped with kneebench, so the UI isn't empty on first start.

The run is stored as a sanitized export of its database rows (not a database file), so it
imports into any later schema: columns the schema no longer has are dropped, new ones get
their defaults. Reports are always regenerated from the rows, so the demo picks up every
rule and wording change. Refresh it with `scripts/make_demo.py`.
"""

from __future__ import annotations

import gzip
import json
from importlib.resources import files
from pathlib import Path
from sqlite3 import Connection
from typing import Any

FORMAT = 1
DEMO_LABEL = "demo"
MARKER = "demo.json"  # in the data directory: which demo version was offered, and its run id

# Tables in insert order, each with the columns that point at an earlier table.
TABLES: tuple[tuple[str, dict[str, str]], ...] = (
    ("endpoints", {}),
    ("snapshots", {"endpoint_id": "endpoints"}),
    ("setups", {"endpoint_id": "endpoints", "snapshot_id": "snapshots"}),
    ("workloads", {}),
    ("runs", {"endpoint_id": "endpoints", "setup_id": "setups", "workload_id": "workloads"}),
    ("points", {"run_id": "runs"}),
    ("point_metrics", {"point_id": "points"}),
    ("server_metrics", {"point_id": "points"}),
    ("requests", {"point_id": "points"}),
    ("warnings", {"point_id": "points"}),
)


def bundled() -> dict[str, Any] | None:
    path = files("kneebench.content").joinpath("demo-run.json.gz")
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(gzip.decompress(path.read_bytes()))
    return data


def export_run(conn: Connection, run_id: int) -> dict[str, Any]:
    """Every row the analysis needs for one run. Timeslices and raw artifacts are left out."""
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(f"no run with id {run_id}")
    setup = conn.execute("SELECT * FROM setups WHERE id = ?", (run["setup_id"],)).fetchone()
    ids: dict[str, list[int]] = {
        "endpoints": [run["endpoint_id"]],
        "snapshots": [setup["snapshot_id"]],
        "setups": [run["setup_id"]],
        "workloads": [run["workload_id"]],
        "runs": [run_id],
    }
    ids["points"] = [
        r["id"] for r in conn.execute("SELECT id FROM points WHERE run_id = ?", (run_id,))
    ]
    tables: dict[str, list[dict[str, Any]]] = {}
    for table, _ in TABLES:
        if table in ids:
            rows = [
                conn.execute(f"SELECT * FROM {table} WHERE id = ?", (i,)).fetchone()
                for i in ids[table]
            ]
        else:
            marks = ",".join("?" * len(ids["points"]))
            rows = list(
                conn.execute(f"SELECT * FROM {table} WHERE point_id IN ({marks})", ids["points"])
            )
        tables[table] = [
            {k: v for k, v in dict(r).items() if not isinstance(v, bytes)} for r in rows
        ]
    return {"format": FORMAT, "tables": tables}


def import_run(conn: Connection, bundle: dict[str, Any], label: str = DEMO_LABEL) -> int:
    """Insert an exported run under new ids; returns the new run id."""
    if bundle.get("format") != FORMAT:
        raise ValueError(f"unsupported demo format {bundle.get('format')!r}")
    new_ids: dict[str, dict[int, int]] = {}
    referenced = {parent for _, refs in TABLES for parent in refs.values()}
    conn.execute("BEGIN")  # the connection autocommits; one transaction, not 55k
    try:
        for table, refs in TABLES:
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            mapping = new_ids.setdefault(table, {})
            batches: dict[tuple[str, ...], list[list[Any]]] = {}
            for row in bundle["tables"].get(table, []):
                if table == "endpoints":
                    same = conn.execute(
                        "SELECT id FROM endpoints WHERE base_url = ?", (row["base_url"],)
                    ).fetchone()
                    if same is not None:
                        mapping[row["id"]] = same["id"]
                        continue
                values = {k: v for k, v in row.items() if k in columns and k != "id"}
                for col, parent in refs.items():
                    values[col] = new_ids[parent][row[col]]
                if table == "runs":
                    values["label"] = label
                sql = _insert(table, tuple(values))
                if table in referenced:
                    cur = conn.execute(sql, list(values.values()))
                    mapping[row["id"]] = int(cur.lastrowid or 0)
                else:  # nothing points at these rows, so they go in as one batch
                    batches.setdefault(tuple(values), []).append(list(values.values()))
            for cols, rows in batches.items():
                conn.executemany(_insert(table, cols), rows)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return next(iter(new_ids["runs"].values()))


def _insert(table: str, columns: tuple[str, ...]) -> str:
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"


def _marker(home: Path) -> dict[str, Any]:
    path = home / MARKER
    return json.loads(path.read_text()) if path.exists() else {}


def demo_run_id(home: Path) -> int | None:
    run_id = _marker(home).get("run_id")
    return int(run_id) if run_id is not None else None


def seed(conn: Connection, home: Path) -> int | None:
    """Offer the bundled demo once per demo version.

    A first start imports it. A newer bundled demo replaces the old one, but only if the
    user kept it: a deleted demo stays deleted until `kneebench demo` brings it back.
    """
    bundle = bundled()
    if bundle is None:
        return None
    version = bundle.get("version")
    marker = _marker(home)
    if marker.get("version") == version:
        return None
    old = demo_run_id(home)
    kept = old is not None and _exists(conn, old)
    if marker and not kept:
        _write_marker(home, version, None)
        return None
    if old is not None and kept:
        from kneebench.store import repo

        repo.delete_run(conn, old)
    run_id = import_run(conn, bundle)
    _write_marker(home, version, run_id)
    return run_id


def restore(conn: Connection, home: Path) -> int:
    """Import the demo again (after it was deleted), replacing any existing copy."""
    bundle = bundled()
    if bundle is None:
        raise FileNotFoundError("this kneebench build has no demo run")
    old = demo_run_id(home)
    if old is not None and _exists(conn, old):
        from kneebench.store import repo

        repo.delete_run(conn, old)
    run_id = import_run(conn, bundle)
    _write_marker(home, bundle.get("version"), run_id)
    return run_id


def _exists(conn: Connection, run_id: int) -> bool:
    return conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is not None


def _write_marker(home: Path, version: Any, run_id: int | None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / MARKER).write_text(json.dumps({"version": version, "run_id": run_id}))

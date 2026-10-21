"""Quality runs in SQLite: record a pass, load it back, find an experiment's passes."""

from __future__ import annotations

from sqlite3 import Connection, Row
from typing import Any

from kneebench.quality.ask import Answer
from kneebench.quality.evalset import EvalSet, parse
from kneebench.store.repo import now

ROLES = ("before", "floor", "after")


def save_set(conn: Connection, evalset: EvalSet) -> int:
    row = conn.execute("SELECT id FROM quality_sets WHERE digest = ?", (evalset.digest,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO quality_sets (name, digest, text, items, created_at) VALUES (?, ?, ?, ?, ?)",
        (evalset.name, evalset.digest, evalset.text, len(evalset.items), now()),
    )
    return int(cur.lastrowid or 0)


def load_set(conn: Connection, set_id: int) -> EvalSet:
    row = conn.execute("SELECT name, text FROM quality_sets WHERE id = ?", (set_id,)).fetchone()
    if row is None:
        raise KeyError(f"no eval set {set_id}")
    return parse(row["name"], row["text"])


def start_run(
    conn: Connection,
    set_id: int,
    endpoint_id: int,
    role: str,
    model: str,
    concurrency: int,
    experiment_id: int | None = None,
    label: str | None = None,
) -> int:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    cur = conn.execute(
        """INSERT INTO quality_runs (set_id, endpoint_id, experiment_id, role, model, label,
           concurrency, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
        (set_id, endpoint_id, experiment_id, role, model, label, concurrency, now()),
    )
    return int(cur.lastrowid or 0)


def finish_run(
    conn: Connection, run_id: int, answers: list[Answer] | None, error: str | None = None
) -> None:
    conn.execute("BEGIN")
    try:
        if answers:
            conn.executemany(
                "INSERT INTO quality_answers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        a.item_id,
                        a.category,
                        a.text,
                        int(a.correct),
                        a.finish_reason,
                        a.output_tokens,
                        a.error,
                    )
                    for a in answers
                ],
            )
        conn.execute(
            "UPDATE quality_runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            ("failed" if error else "done", error, now(), run_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def get_run(conn: Connection, run_id: int) -> Row | None:
    row: Row | None = conn.execute(
        """SELECT quality_runs.*, quality_sets.name AS set_name, quality_sets.items AS set_items,
           endpoints.base_url FROM quality_runs
           JOIN quality_sets ON quality_sets.id = quality_runs.set_id
           JOIN endpoints ON endpoints.id = quality_runs.endpoint_id
           WHERE quality_runs.id = ?""",
        (run_id,),
    ).fetchone()
    return row


def answers(conn: Connection, run_id: int) -> list[Answer]:
    return [
        Answer(
            r["item_id"],
            r["category"],
            r["text"],
            bool(r["correct"]),
            r["finish_reason"],
            r["output_tokens"],
            r["error"],
        )
        for r in conn.execute(
            "SELECT * FROM quality_answers WHERE run_id = ? ORDER BY rowid", (run_id,)
        )
    ]


def for_experiment(conn: Connection, experiment_id: int) -> dict[str, Any]:
    """The latest finished pass per role."""
    rows = conn.execute(
        """SELECT id, role FROM quality_runs WHERE experiment_id = ? AND status = 'done'
           ORDER BY id""",
        (experiment_id,),
    )
    return {r["role"]: int(r["id"]) for r in rows}


def list_runs(conn: Connection) -> list[Row]:
    return list(
        conn.execute(
            """SELECT quality_runs.*, quality_sets.name AS set_name, endpoints.base_url,
               (SELECT AVG(correct) FROM quality_answers WHERE run_id = quality_runs.id)
                 AS accuracy
               FROM quality_runs JOIN quality_sets ON quality_sets.id = quality_runs.set_id
               JOIN endpoints ON endpoints.id = quality_runs.endpoint_id
               ORDER BY quality_runs.id DESC"""
        )
    )

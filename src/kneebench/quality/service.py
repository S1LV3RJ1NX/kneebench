"""Record passes over an eval set and compare them."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any

import httpx

from kneebench.loadgen.base import Endpoint
from kneebench.quality import store
from kneebench.quality.ask import CONCURRENCY, Answer, ask_all
from kneebench.quality.evalset import EvalSet, Item
from kneebench.quality.grade import normalize
from kneebench.quality.stats import Comparison, compare
from kneebench.store import repo

MAX_EXAMPLES = 60


def record(
    conn: Connection,
    endpoint: Endpoint,
    evalset: EvalSet,
    role: str,
    experiment_id: int | None = None,
    label: str | None = None,
    concurrency: int = CONCURRENCY,
    on_progress: Callable[[int, int], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    set_id = store.save_set(conn, evalset)
    endpoint_id = repo.upsert_endpoint(conn, endpoint.base_url)
    run_id = store.start_run(
        conn, set_id, endpoint_id, role, endpoint.model, concurrency, experiment_id, label
    )
    try:
        answers = ask_all(endpoint, evalset, concurrency, on_progress, transport)
    except Exception as exc:
        store.finish_run(conn, run_id, None, error=f"{type(exc).__name__}: {exc}")
        raise
    failed = [a for a in answers if a.error]
    error = None
    if len(failed) == len(answers):
        error = f"every question failed ({failed[0].error})"
    store.finish_run(conn, run_id, answers, error=error)
    return run_id


def record_before(
    conn: Connection,
    endpoint: Endpoint,
    evalset: EvalSet,
    experiment_id: int,
    rule: dict[str, float],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Write the rule down, then ask twice: the second pass is the noise floor."""
    conn.execute(
        "UPDATE experiments SET quality_rule_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(rule), repo.now(), experiment_id),
    )
    before = record(conn, endpoint, evalset, "before", experiment_id, on_progress=on_progress)
    floor = record(conn, endpoint, evalset, "floor", experiment_id, on_progress=on_progress)
    return before, floor


@dataclass(frozen=True)
class QualityView:
    comparison: Comparison
    evalset: EvalSet
    before_id: int
    after_id: int
    floor_id: int | None
    examples: list[dict[str, Any]]


def compare_runs(
    conn: Connection,
    before_id: int,
    after_id: int,
    floor_id: int | None = None,
    rule: dict[str, float] | None = None,
) -> QualityView:
    runs = {k: store.get_run(conn, v) for k, v in (("before", before_id), ("after", after_id))}
    if floor_id:
        runs["floor"] = store.get_run(conn, floor_id)
    missing = [k for k, r in runs.items() if r is None]
    if missing:
        raise KeyError(f"no quality run for {', '.join(missing)}")
    sets = {r["set_id"] for r in runs.values() if r is not None}
    if len(sets) != 1:
        raise ValueError("the passes asked different question sets; compare passes of one set")
    evalset = store.load_set(conn, sets.pop())
    before, after = store.answers(conn, before_id), store.answers(conn, after_id)
    floor = store.answers(conn, floor_id) if floor_id else None
    result = compare(before, after, floor, rule)
    return QualityView(
        comparison=result,
        evalset=evalset,
        before_id=before_id,
        after_id=after_id,
        floor_id=floor_id,
        examples=examples(evalset, result.pairs),
    )


def summary(view: QualityView) -> list[str]:
    """The result in sentences, verdict first (lesson 54 section 7)."""
    c = view.comparison
    o, f, rule = c.overall, c.floor, c.rule
    budget = rule["max_drop_points"]
    pts = 100 * o.diff
    sentences = {
        "pass": f"Pass: any drop is within your {budget:g}-point budget at 95% confidence.",
        "fail": "Fail: " + _why_failed(view),
        "inconclusive": (
            f"Inconclusive: {o.n} questions can't show the drop is under {budget:g} point"
            f"{'' if budget == 1 else 's'} (the interval is ±{100 * o.ci:.1f}). Ask more "
            "questions, or decide whether a bigger drop is acceptable."
        ),
    }
    out = [sentences[c.verdict.overall]]
    out.append(
        f"Right before: {o.acc_a:.1%}, after: {o.acc_b:.1%}. Question by question that is "
        f"{pts:+.1f} ± {100 * o.ci:.1f} points (p = {o.p_value:.2f}); scored separately the "
        f"interval would be ±{100 * o.unpaired_ci:.1f}."
    )
    floor = (
        f" Asking the same model twice changed {f.changed_rate:.1%} and flipped "
        f"{f.flip_rate:.1%}: that is the noise floor."
        if f
        else ""
    )
    out.append(
        f"{o.lost} went right to wrong and {o.gained} wrong to right ({o.flip_rate:.1%} flipped); "
        f"{o.changed_rate:.1%} of answers changed at all.{floor}"
    )
    if c.errors_before or c.errors_after:
        out.append(
            f"{c.errors_before + c.errors_after} answers failed to arrive and were left out."
        )
    return out


def _why_failed(view: QualityView) -> str:
    c = view.comparison
    v, o, f = c.verdict, c.overall, c.floor
    reasons = []
    if v.accuracy == "fail":
        reasons.append(
            f"accuracy dropped {-100 * o.diff:.1f} ± {100 * o.ci:.1f} points, past your "
            f"{c.rule['max_drop_points']:g}-point budget"
        )
    if v.flips == "fail" and f is not None:
        reasons.append(
            f"{o.flip_rate:.1%} of answers flipped against a {f.flip_rate:.1%} noise floor"
        )
    if v.categories:
        reasons.append(f"significantly worse on {', '.join(v.categories)}")
    return "; ".join(reasons) + "."


def examples(evalset: EvalSet, pairs: list[tuple[Answer, Answer]]) -> list[dict[str, Any]]:
    """Changed answers to read, regressions first (lesson 54 section 5: read them)."""
    items = {i.id: i for i in evalset.items}
    out = []
    for a, b in pairs:
        kind = (
            "lost"
            if a.correct and not b.correct
            else "gained"
            if b.correct and not a.correct
            else None
        )
        if kind is None and normalize(a.text or "") == normalize(b.text or ""):
            continue
        item = items.get(a.item_id)
        out.append(
            {
                "id": a.item_id,
                "category": a.category,
                "kind": kind or "changed",
                "question": _question(item) if item else "",
                "expected": item.answer if item else "",
                "before": a.text,
                "after": b.text,
            }
        )
    order = {"lost": 0, "gained": 1, "changed": 2}
    out.sort(key=lambda e: order[str(e["kind"])])
    return out[:MAX_EXAMPLES]


def _question(item: Item) -> str:
    users = [m["content"] for m in item.messages if m.get("role") == "user"]
    return users[-1] if users else item.messages[-1]["content"]

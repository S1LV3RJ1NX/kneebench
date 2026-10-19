"""Build a report for a stored run: stored data in, rendered report out. No network."""

from __future__ import annotations

from sqlite3 import Connection

from kneebench.analysis.context import context_from_bundle
from kneebench.analysis.engine import RULES_VERSION, evaluate
from kneebench.narrate import Narrator, PassthroughNarrator
from kneebench.report.composer import Report, compose
from kneebench.store import repo


def build_report(
    conn: Connection,
    run_id: int,
    narrator: Narrator | None = None,
    now: str | None = None,
    style: str = "md",
) -> Report:
    ctx = context_from_bundle(repo.run_bundle(conn, run_id))
    findings = evaluate(ctx)
    repo.save_findings(conn, run_id, [f.to_json() for f in findings], RULES_VERSION)
    return (narrator or PassthroughNarrator()).rewrite(compose(ctx, findings, now=now, style=style))

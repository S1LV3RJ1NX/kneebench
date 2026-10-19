"""Build a report for a stored run: stored data in, rendered report out. No network."""

from __future__ import annotations

from dataclasses import replace
from sqlite3 import Connection

from kneebench.analysis.context import context_from_bundle
from kneebench.analysis.engine import RULES_VERSION, evaluate
from kneebench.domain import Slo
from kneebench.narrate import Narrator, PassthroughNarrator
from kneebench.report.composer import Report, compose
from kneebench.store import repo


def build_report(
    conn: Connection,
    run_id: int,
    narrator: Narrator | None = None,
    now: str | None = None,
    style: str = "md",
    slo: Slo | None = None,
) -> Report:
    """With `slo`, judge the stored measurements against other targets (a what-if)."""
    ctx = context_from_bundle(repo.run_bundle(conn, run_id))
    what_if = None
    if slo is not None and slo != ctx.workload.slo:
        what_if = {"measured_with": ctx.workload.slo, "judged_with": slo}
        ctx = replace(ctx, workload=replace(ctx.workload, slo=slo))
    findings = evaluate(ctx)
    if what_if is None:
        repo.save_findings(conn, run_id, [f.to_json() for f in findings], RULES_VERSION)
    report = compose(ctx, findings, now=now, style=style, what_if=what_if)
    return (narrator or PassthroughNarrator()).rewrite(report)

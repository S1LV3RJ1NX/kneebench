"""Before-and-after comparison of two stored runs. No network."""

from __future__ import annotations

from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Any

import plotly.graph_objects as go

from kneebench.analysis.compare import (
    SETUP_LABELS,
    WORKLOAD_LABELS,
    CompareContext,
    compare,
    metric_rows,
)
from kneebench.analysis.context import context_from_bundle
from kneebench.content.render import (
    RenderedFinding,
    TermTracker,
    make_env,
    num,
    render_finding,
    times,
)
from kneebench.report.charts import ACCENT, SECOND, _fig
from kneebench.report.composer import common_vars
from kneebench.store import repo


@dataclass
class Comparison:
    a_id: int
    b_id: int
    a_label: str | None
    b_label: str | None
    summary: list[str]
    findings: list[RenderedFinding]
    setup: list[tuple[str, str, str, bool]]  # label, before, after, changed
    metrics: list[tuple[str, str, str, str, str]]  # label, before, after, change, verdict
    figures: list[dict[str, Any]] = field(default_factory=list)


def _text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return num(value, 0 if value >= 100 else 1)
    if isinstance(value, int):
        return num(value)
    return str(value)


def build_comparison(
    conn: Connection,
    a_id: int,
    b_id: int,
    style: str = "md",
    declared: dict[str, Any] | None = None,
) -> Comparison:
    labels = []
    for rid in (a_id, b_id):
        row = repo.get_run(conn, rid)
        if row is None:
            raise KeyError(f"run {rid} not found")
        labels.append(row["label"])
    label_a, label_b = labels
    ctx_a = context_from_bundle(repo.run_bundle(conn, a_id))
    ctx_b = context_from_bundle(repo.run_bundle(conn, b_id))
    cc, findings = compare(ctx_a, ctx_b, label_a, label_b, declared)
    env = make_env(TermTracker(style))
    common = common_vars(ctx_b)
    rendered = [render_finding(f, common, env) for f in findings]
    return Comparison(
        a_id=a_id,
        b_id=b_id,
        a_label=label_a,
        b_label=label_b,
        summary=[r.status_line for r in rendered],
        findings=rendered,
        setup=_setup_rows(cc),
        metrics=[
            (
                r["title"],
                _text(r["a"]),
                _text(r["b"]),
                times(r["ratio"]) if r["ratio"] else "-",
                r["verdict"],
            )
            for r in metric_rows(cc)
        ],
        figures=_figures(cc),
    )


def _setup_rows(cc: CompareContext) -> list[tuple[str, str, str, bool]]:
    rows = []
    for source, labels in ((("setup", SETUP_LABELS)), ("workload", WORKLOAD_LABELS)):
        a, b = getattr(cc.a, source), getattr(cc.b, source)
        for key, label in labels.items():
            title = label[0].upper() + label[1:]
            rows.append((title, _text(a.get(key)), _text(b.get(key)), a.get(key) != b.get(key)))
    return rows


def _figures(cc: CompareContext) -> list[dict[str, Any]]:
    if not cc.a.curve and not cc.b.curve:
        return []
    styles = (
        (cc.a, f"Before (run {cc.a.run_id})", SECOND, "dot"),
        (cc.b, f"After (run {cc.b.run_id})", ACCENT, "solid"),
    )
    curve = go.Figure()
    tput = go.Figure()
    for run, name, color, dash in styles:
        pts = run.curve
        curve.add_scatter(
            x=[p["tok_s"] for p in pts],
            y=[p.get("tpot_tail") for p in pts],
            text=[f"{p['concurrency']} users" for p in pts],
            mode="lines+markers",
            name=name,
            line={"color": color, "dash": dash, "width": 2.5},
            hovertemplate="%{text}: %{x:,.0f} tok/s, %{y:.1f} ms<extra></extra>",
        )
        tput.add_scatter(
            x=[p["concurrency"] for p in pts],
            y=[p["tok_s"] for p in pts],
            mode="lines+markers",
            name=name,
            line={"color": color, "dash": dash, "width": 2.5},
            hovertemplate="%{x} users: %{y:,.0f} tok/s<extra></extra>",
        )
    return [
        _fig(
            curve,
            "Latency against throughput: down and to the right is better",
            xaxis={"title": "Output tokens per second"},
            yaxis={"title": "Tail time per output token (ms)", "rangemode": "tozero"},
        ),
        _fig(
            tput,
            "Throughput as users are added",
            xaxis={"title": "Users in flight", "type": "log"},
            yaxis={"title": "Output tokens per second", "rangemode": "tozero"},
        ),
    ]


def render_comparison_md(c: Comparison) -> str:
    def name(rid: int, label: str | None) -> str:
        return f"run {rid}" + (f" ({label})" if label else "")

    out = [f"# kneebench comparison: {name(c.a_id, c.a_label)} vs {name(c.b_id, c.b_label)}", ""]
    out += [" ".join(c.summary), "", "## Setup", "", "| | Before | After |", "|---|---|---|"]
    out += [f"| {'**' + lbl + '**' if ch else lbl} | {a} | {b} |" for lbl, a, b, ch in c.setup]
    out += [
        "",
        "## Results",
        "",
        "| Measure | Before | After | Change | Verdict |",
        "|---|---|---|---|---|",
    ]
    out += [f"| {lbl} | {a} | {b} | {ch} | {v} |" for lbl, a, b, ch, v in c.metrics]
    for f in c.findings:
        out += ["", f"## {f.title}", "", f"**{f.status_line}**"]
        for head, text in (("What it means", f.meaning), ("Why it happens", f.why)):
            if text:
                out += ["", f"**{head}.** {text}"]
        if f.learn_more_url:
            out += ["", f"[Learn more: {f.learn_more_label}]({f.learn_more_url})"]
    return "\n".join(out) + "\n"

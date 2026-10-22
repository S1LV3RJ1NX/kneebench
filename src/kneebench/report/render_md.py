"""Markdown report: readable by people, and by LLMs for follow-up questions."""

from __future__ import annotations

from kneebench.content.render import RenderedFinding, RenderedTable
from kneebench.report.composer import Report

_DOT = {
    "ok": "[good]",
    "warn": "[watch]",
    "bad": "[problem]",
    "none": "[not measured]",
    "info": "[context]",
    "skip": "[not checked]",
}


def _table(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _finding(f: RenderedFinding, level: str = "###") -> list[str]:
    out = [f"{level} {_DOT[f.status]} {f.title} ({f.rule_id})", "", f"**{f.status_line}**", ""]
    if f.measured:
        out += [f"**What we measured.** {f.measured}", ""]
    if isinstance(f.table, RenderedTable):
        out += [*_table(f.table.columns, f.table.rows), ""]
    if f.meaning:
        meaning = f.meaning + (f" {f.analogy}" if f.analogy else "")
        out += [f"**What it means.** {meaning}", ""]
    if f.why:
        why = f.why
        if f.learn_more_url:
            why += f" [Learn more: {f.learn_more_label}]({f.learn_more_url})"
        out += [f"**Why it happens.** {why}", ""]
    for a in f.actions:
        line = f"**What to do.** {a.title}."
        if a.flag:
            line += f" Try `{a.flag}`."
        if a.expected:
            line += f" Expected: {a.expected}"
        if a.verify:
            line += f" Verify: {a.verify}"
        out += [line, ""]
    return out


def render_markdown(report: Report) -> str:
    s = report.sections
    out = [f"# {report.title}", ""]
    summary = report.summary
    out += [
        f"Run {report.run_id} · {summary['product']} workload · vLLM {summary['vllm_version']} · "
        f"generated {report.generated_at}",
        "",
        *([f"> {report.what_if}", ""] if report.what_if else []),
        *(
            [f"## {s['setup']}", "", *(f"- **{k}:** {v}" for k, v in report.setup), ""]
            if report.setup
            else []
        ),
        f"## {s['verdict']}",
        "",
        *(f"- {line}" for line in report.verdict),
        "",
        f"## {s['scorecard']}",
        "",
    ]
    out += _table(
        ("", "Area", "Result"), tuple((_DOT[t.status], t.label, t.text) for t in report.scorecard)
    )
    out.append("")

    if report.problems:
        out += [f"## {s['problems']}", ""]
        for f in report.problems:
            out += _finding(f)
    if report.next_steps:
        out += [f"## {s['next']}", ""]
        for kind in ("server", "test"):
            group = [st for st in report.next_steps if st.action.kind == kind]
            if not group:
                continue
            out += [f"### {s['next_' + kind]}", "", s[f"next_{kind}_intro"], ""]
            for step in group:
                a = step.action
                line = f"{step.rank}. **{a.title}** ({step.rule_id}). Why: {step.reason}"
                if a.expected:
                    line += f" Expected: {a.expected}"
                if a.risk:
                    line += f" Risk: {a.risk}"
                out.append(line)
                if a.flag:
                    out += ["", "   ```", f"   {a.flag}", "   ```"]
            out.append("")
    if report.capacity:
        out += [f"## {s['capacity']}", ""]
        for f in report.capacity:
            out += _finding(f)
    if report.shapes:
        out += [f"## {s['shapes']}", ""]
        for f in report.shapes:
            out += _finding(f)
    if report.reuse:
        out += [f"## {s['reuse']}", ""]
        for f in report.reuse:
            out += _finding(f)
    if report.context:
        out += [f"## {s['context']}", ""]
        for f in report.context:
            out += _finding(f)
    if report.passed:
        out += [f"## {s['passed']}", ""]
        out += [
            f"- {_DOT['ok']} **{f.title}** ({f.rule_id}): {f.status_line}" for f in report.passed
        ]
        out.append("")
    if report.skipped:
        out += [f"## {s['skipped']}", ""]
        out += [f"- **{f.title}** ({f.rule_id}): {f.status_line}" for f in report.skipped]
        out.append("")
    out += [f"## {s['caveats']}", ""] + [f"- {c}" for c in report.caveats] + [""]
    if report.glossary:
        out += [f"## {s['glossary']}", ""]
        out += [f"- **{term}**: {definition}" for term, definition in report.glossary] + [""]
    out += [f"## {s['appendix']}", ""]
    for t in report.appendix:
        out += [f"### {t.title}", "", *_table(t.columns, t.rows), ""]
    out += [
        "",
        "_Versions: "
        + ", ".join(
            f"{k} {summary[k]}" for k in ("kneebench_version", "aiperf_version", "rules_version")
        )
        + "._",
        "",
    ]
    return "\n".join(out)

"""Render rule content (YAML + Jinja) with a run's own numbers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from typing import Any

import jinja2
import yaml
from markupsafe import Markup, escape

from kneebench import content
from kneebench.analysis.engine import REGISTRY, Finding

UNKNOWN = "n/a"


# ---- number formatting ----


def ms(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN
    if value >= 1000:
        s = value / 1000
        text = f"{s:.2f}" if s < 10 else f"{s:.1f}" if s < 100 else f"{s:,.0f}"
        return f"{_trim(text)} s"
    if value >= 100:
        return f"{value:.0f} ms"
    if value >= 1:
        return f"{_trim(f'{value:.1f}')} ms"
    return f"{value:.2f} ms"


def _trim(text: str) -> str:
    """50.0 reads as 50; 1.50 as 1.5."""
    return text.rstrip("0").rstrip(".") if "." in text else text


def pct(value: float | None) -> str:
    if value is None:
        return UNKNOWN
    p = value * 100
    return f"{p:.1f}%" if abs(p) < 10 and p != 0 else f"{p:.0f}%"


def num(value: float | None, decimals: int | None = None) -> str:
    if value is None:
        return UNKNOWN
    if decimals is None:
        decimals = 0 if float(value).is_integer() else 1
    return f"{value:,.{decimals}f}"


def gib(value: float | None) -> str:
    return UNKNOWN if value is None else f"{value:.1f} GiB"


def kib(value_bytes: float | None) -> str:
    return UNKNOWN if value_bytes is None else f"{value_bytes / 1024:,.0f} KiB"


def times(value: float | None) -> str:
    if value is None:
        return UNKNOWN
    return f"{value:.1f}×" if value < 10 else f"{value:,.0f}×"


def _plain(value: Any) -> str:
    return UNKNOWN if value is None else str(value)


def num0(value: float | None) -> str:
    return num(value, 0)


FORMATTERS: dict[str, Callable[..., str]] = {
    "ms": ms,
    "pct": pct,
    "num": num,
    "num0": num0,
    "gib": gib,
    "kib": kib,
    "times": times,
}


# ---- glossary ----


@cache
def glossary() -> dict[str, str]:
    terms: dict[str, str] = content.load("glossary.yaml")["terms"]
    return terms


class TermTracker:
    """Defines a term inline on first use in a report; later uses stay short."""

    def __init__(self, style: str = "md") -> None:
        self.style = style
        self.used: dict[str, str] = {}

    def __call__(self, name: str) -> str:
        key = next((k for k in glossary() if k.lower() == name.lower()), None)
        if key is None:
            raise KeyError(f"term {name!r} is not in glossary.yaml")
        definition = glossary()[key]
        first = key not in self.used
        self.used.setdefault(key, definition)
        if self.style == "html":
            html = f'<abbr title="{escape(definition)}">{escape(name)}</abbr>'
            if first:
                html += f' <span class="def">({escape(definition)})</span>'
            return Markup(html)
        return f"{name} ({definition})" if first else name


# ---- notes ----


def note_link(note: str, anchor: str | None = None) -> tuple[str, str]:
    notes = content.load("notes.yaml")
    entry = notes["notes"][note]
    url = f"{notes['base_url']}{entry['slug']}.md" + (f"#{anchor}" if anchor else "")
    return url, f"Lesson {note}: {entry['title']}"


# ---- rule content ----


@cache
def rule_content(rule_id: str) -> dict[str, Any]:
    path = files("kneebench.content").joinpath("rules", f"{rule_id}.yaml")
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


@dataclass(frozen=True)
class RenderedAction:
    title: str
    flag: str | None
    expected: str | None
    risk: str | None
    verify: str | None


@dataclass(frozen=True)
class RenderedTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RenderedFinding:
    rule_id: str
    status: str
    severity: int
    tags: tuple[str, ...]
    source: str
    title: str
    status_line: str
    measured: str | None = None
    meaning: str | None = None
    analogy: str | None = None
    why: str | None = None
    learn_more_url: str | None = None
    learn_more_label: str | None = None
    actions: tuple[RenderedAction, ...] = field(default=())
    table: RenderedTable | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _cell(row: dict[str, Any], column: dict[str, Any]) -> str:
    fmt: Callable[..., str] = FORMATTERS.get(str(column.get("fmt", "")), _plain)
    return fmt(row.get(str(column["key"])))


def make_env(tracker: TermTracker) -> jinja2.Environment:
    # HTML output escapes every value: model names and errors come from the server.
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=tracker.style == "html")
    env.globals.update(FORMATTERS)
    env.globals["term"] = tracker
    return env


def _pick(value: Any, status: str) -> Any:
    return value.get(status) if isinstance(value, dict) else value


def _render(env: jinja2.Environment, template: str | None, variables: dict[str, Any]) -> str | None:
    if template is None:
        return None
    text = env.from_string(str(template)).render(**variables)
    return " ".join(text.split())


def render_finding(
    finding: Finding,
    common: dict[str, Any],
    env: jinja2.Environment,
) -> RenderedFinding:
    spec = rule_content(finding.rule_id)
    status = finding.status.value
    clash = set(finding.data) & (set(FORMATTERS) | {"term"})
    if clash:
        raise ValueError(f"{finding.rule_id} data keys shadow template helpers: {sorted(clash)}")
    variables = {**common, **finding.data}
    if status == "skip" and isinstance(spec.get("skip_reasons"), dict):
        line_tpl = spec["skip_reasons"].get(
            finding.data.get("reason"), _pick(spec["status_line"], status)
        )
    else:
        line_tpl = _pick(spec["status_line"], status)
    status_line = _render(env, line_tpl, variables) or spec["title"]

    detailed = status in ("warn", "bad", "info")
    measured = meaning = analogy = why = None
    if detailed:
        measured = _render(env, _pick(spec.get("measured"), status), variables)
        meaning = _render(env, _pick(spec.get("meaning"), status), variables)
        analogy = _render(env, spec.get("analogy"), variables) if status != "info" else None
        why = _render(env, spec.get("why"), variables)

    actions = tuple(
        RenderedAction(
            title=_render(env, a["title"], variables) or "",
            flag=_render(env, a.get("flag"), variables),
            expected=_render(env, a.get("expected"), variables),
            risk=_render(env, a.get("risk"), variables),
            verify=_render(env, a.get("verify"), variables),
        )
        for a in spec.get("actions", [])
        if status in a.get("when", [])
    )

    url = label = None
    if spec.get("learn_more"):
        lm = spec["learn_more"]
        url, label = note_link(str(lm["note"]), lm.get("anchor"))
        if lm.get("label"):
            label = f"{label}, {lm['label']}"

    table = None
    if detailed and spec.get("table") and isinstance(finding.data.get(spec["table"]["rows"]), list):
        cols = spec["table"]["columns"]
        rows = tuple(
            tuple(_cell(row, c) for c in cols) for row in finding.data[spec["table"]["rows"]]
        )
        table = RenderedTable(tuple(c["label"] for c in cols), rows)

    rule = REGISTRY[finding.rule_id]
    return RenderedFinding(
        rule_id=finding.rule_id,
        status=status,
        severity=finding.severity,
        tags=tuple(sorted(finding.tags)),
        source=rule.source,
        title=spec["title"],
        status_line=status_line,
        measured=measured,
        meaning=meaning,
        analogy=analogy,
        why=why,
        learn_more_url=url,
        learn_more_label=label,
        actions=actions,
        table=table,
        data=finding.data,
    )

"""Plotly figures built from finding data. Every chart's numbers also appear in a table."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from kneebench.report.composer import Report

ACCENT = "#d97757"  # clay
INK = "#23211c"
MUTED = "#9a958a"
OK = "#2e7d32"
BAD = "#c0392b"
SECOND = "#6f6b61"
_GRID = "#ece7db"
_LAYOUT: dict[str, Any] = {
    "template": "simple_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 60, "r": 20, "t": 40, "b": 52},
    "height": 320,
    "font": {"family": "Geist, Inter, system-ui, sans-serif", "size": 12, "color": "#6f6b61"},
    "legend": {"orientation": "h", "y": -0.28},
    "xaxis": {"gridcolor": _GRID, "linecolor": "#e4dfd3", "zeroline": False},
    "yaxis": {"gridcolor": _GRID, "linecolor": "#e4dfd3", "zeroline": False},
}


def _finding(report: Report, rule_id: str) -> dict[str, Any] | None:
    return next((f for f in report.findings_json if f["rule_id"] == rule_id), None)


def charts(report: Report) -> dict[str, list[dict[str, Any]]]:
    """Figures (as Plotly JSON-able dicts) keyed by the rule they illustrate."""
    out: dict[str, list[dict[str, Any]]] = {}
    c02 = _finding(report, "C02")
    if not c02:
        return out
    data = c02["data"]
    pts = [p for p in data["points"] if p.get("tok_s")]
    if not pts:
        return out
    pct = data["pct_label"]
    c03 = _finding(report, "C03")
    op_c = (c03 or {}).get("data", {}).get("op", {}) or {}
    op_c = op_c.get("concurrency") if isinstance(op_c, dict) else None
    out["C02"] = [_curve(pts, pct, data.get("slo_tpot_ms"), op_c), _throughput(pts, op_c)]
    out["W02"] = [_latency(pts, pct, data.get("slo_tpot_ms"), data.get("slo_ttft_ms"))]
    if any(p.get("kv_max") is not None for p in pts):
        out["W04"] = [_memory(pts)]
    return out


def _fig(fig: go.Figure, title: str, **layout: Any) -> dict[str, Any]:
    merged = {**_LAYOUT}
    for key, value in layout.items():
        merged[key] = (
            {**merged[key], **value} if isinstance(value, dict) and key in merged else value
        )
    fig.update_layout(title={"text": title, "x": 0, "font": {"size": 14, "color": INK}}, **merged)
    result: dict[str, Any] = fig.to_plotly_json()
    return result


def _curve(
    pts: list[dict[str, Any]], pct: int, slo: float | None, op: int | None
) -> dict[str, Any]:
    fig = go.Figure()
    fig.add_scatter(
        x=[p["tok_s"] for p in pts],
        y=[p["tpot_tail"] for p in pts],
        mode="lines+markers+text",
        text=[str(p["concurrency"]) for p in pts],
        textposition="top left",
        marker={
            "color": [MUTED if op and p["concurrency"] > op else ACCENT for p in pts],
            "size": 9,
        },
        line={"color": MUTED},
        name="users in flight",
        hovertemplate="%{text} users<br>%{x:,.0f} tok/s<br>%{y:.1f} ms<extra></extra>",
    )
    if slo:
        fig.add_hline(
            y=slo,
            line_dash="dash",
            line_color=BAD,
            annotation_text=f"target {slo:g} ms",
            annotation_position="top left",
        )
    return _fig(
        fig,
        f"Throughput vs P{pct} time per token (labels: users in flight)",
        xaxis_title="Output tokens per second (all users)",
        yaxis_title=f"P{pct} time per token (ms)",
        showlegend=False,
    )


def _throughput(pts: list[dict[str, Any]], op: int | None) -> dict[str, Any]:
    fig = go.Figure()
    fig.add_scatter(
        x=[p["concurrency"] for p in pts],
        y=[p["tok_s"] for p in pts],
        mode="lines+markers",
        line={"color": ACCENT},
        name="tokens per second",
    )
    best = next((p for p in pts if p["concurrency"] == op), None)
    if best:
        # A marker, not a vline: shapes on log axes need log-space coordinates.
        fig.add_scatter(
            x=[best["concurrency"]],
            y=[best["tok_s"]],
            mode="markers+text",
            marker={"color": OK, "size": 13, "symbol": "star"},
            text=["best point within targets"],
            textposition="middle right",
        )
    return _fig(
        fig,
        "Throughput as users are added",
        xaxis_type="log",
        xaxis_title="Users in flight (log scale)",
        yaxis_title="Output tokens per second",
        showlegend=False,
    )


def _latency(
    pts: list[dict[str, Any]], pct: int, tpot: float | None, ttft: float | None
) -> dict[str, Any]:
    fig = go.Figure()
    xs = [p["concurrency"] for p in pts]
    fig.add_scatter(
        x=xs,
        y=[p["tpot_tail"] for p in pts],
        name=f"P{pct} time per token (ms)",
        mode="lines+markers",
        line={"color": ACCENT},
    )
    fig.add_scatter(
        x=xs,
        y=[p["ttft_tail"] for p in pts],
        name=f"P{pct} time to first token (ms)",
        mode="lines+markers",
        line={"color": SECOND},
        yaxis="y2",
    )
    if tpot:
        fig.add_hline(y=tpot, line_dash="dash", line_color=ACCENT, opacity=0.6)
    if ttft:
        fig.add_hline(y=ttft, line_dash="dash", line_color=SECOND, opacity=0.6, yref="y2")
    return _fig(
        fig,
        "Latency tails as users are added",
        xaxis_type="log",
        xaxis_title="Users in flight (log scale)",
        yaxis={"title": "Time per token (ms)", "rangemode": "tozero"},
        yaxis2={
            "title": "Time to first token, server side (ms)",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
        },
    )


def _memory(pts: list[dict[str, Any]]) -> dict[str, Any]:
    fig = go.Figure()
    xs = [p["concurrency"] for p in pts]
    fig.add_bar(
        x=xs,
        y=[(p.get("kv_max") or 0) * 100 for p in pts],
        name="KV cache peak (%)",
        marker_color=ACCENT,
        opacity=0.7,
    )
    fig.add_scatter(
        x=xs,
        y=[p.get("preemptions") or 0 for p in pts],
        name="preemptions",
        mode="lines+markers",
        line={"color": BAD},
        yaxis="y2",
    )
    return _fig(
        fig,
        "Memory as users are added",
        xaxis_type="log",
        xaxis_title="Users in flight (log scale)",
        yaxis_title="KV cache peak (%)",
        yaxis={"range": [0, 105]},
        yaxis2={"title": "Preemptions", "overlaying": "y", "side": "right", "rangemode": "tozero"},
    )

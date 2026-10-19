"""Plotly figures built from finding data. Every chart's numbers also appear in a table."""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go

from kneebench.report.composer import Report

ACCENT = "#d97757"  # clay
INK = "#23211c"
MUTED = "#b5afa3"
OK = "#2e7d32"
BAD = "#c0392b"
SECOND = "#6f6b61"
_GRID = "#ece7db"
_LAYOUT: dict[str, Any] = {
    "template": "simple_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 60, "r": 24, "t": 44, "b": 52},
    "height": 320,
    "font": {"family": "Geist, Inter, system-ui, sans-serif", "size": 12, "color": "#6f6b61"},
    "legend": {"orientation": "h", "y": -0.28},
    "xaxis": {"gridcolor": _GRID, "linecolor": "#e4dfd3", "zeroline": False},
    "yaxis": {"gridcolor": _GRID, "linecolor": "#e4dfd3", "zeroline": False},
    "hoverlabel": {"font": {"family": "Geist, Inter, sans-serif"}},
}
WALL_LABELS = {
    "slo_tpot": "Per-token target",
    "slo_ttft": "First-token target",
    "saturation": "Saturation",
    "memory": "KV cache memory",
    "max_num_seqs": "max_num_seqs",
}


def _data(report: Report, rule_id: str) -> dict[str, Any] | None:
    f = next((f for f in report.findings_json if f["rule_id"] == rule_id), None)
    return f["data"] if f else None


def charts(report: Report) -> dict[str, list[dict[str, Any]]]:
    """Figures (Plotly JSON) keyed by the rule they illustrate."""
    out: dict[str, list[dict[str, Any]]] = {}
    builders = (
        ("B01", _baseline_vs_floor),
        ("B04", _baseline_spread),
        ("H03", _network_share),
        ("T03", _fits),
        ("W03", _walls),
    )
    for rule_id, build in builders:
        data = _data(report, rule_id)
        fig = build(data) if data else None
        if fig:
            out[rule_id] = [fig]

    r01 = _data(report, "R01")
    if r01 and r01.get("rows"):
        out["R01"] = [_rate_ttft(r01)]
    r02 = _data(report, "R02")
    if r02 and any(r.get("trend") for r in r02.get("rows", [])):
        out["R02"] = [_rate_trend(r02)]
    r04 = _data(report, "R04")
    if r04 and r04.get("pairs"):
        out["R04"] = [_bursts(r04)]

    c02 = _data(report, "C02")
    pts = [p for p in (c02 or {}).get("points", []) if p.get("tok_s")]
    if c02 and pts:
        pct = c02["pct_label"]
        c03 = _data(report, "C03") or {}
        op = c03.get("op") if isinstance(c03.get("op"), dict) else None
        op_c = op.get("concurrency") if op else None
        out["C02"] = [_curve(pts, pct, c02.get("slo_tpot_ms"), op_c), _throughput(pts, op_c)]
        out["W02"] = [_latency(pts, pct, c02.get("slo_tpot_ms"), c02.get("slo_ttft_ms"))]
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


def _fmt_ms(v: float) -> str:
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms"


# ---- single request ----


def _baseline_vs_floor(d: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        ("Time per token", d.get("tpot_ms"), d.get("floor_ms")),
        ("End to end", d.get("e2e_ms"), d.get("e2e_floor_ms")),
        ("First token (server side)", d.get("server_ttft_ms"), d.get("ttft_floor_ms")),
    ]
    rows = [(name, float(v), float(f)) for name, v, f in candidates if v and f]
    if not rows:
        return None
    labels = [r[0] for r in rows]
    ratios = [r[1] / r[2] for r in rows]
    fig = go.Figure()
    fig.add_bar(
        y=labels,
        x=ratios,
        orientation="h",
        marker_color=ACCENT,
        width=0.5,
        text=[f"{_fmt_ms(r[1])} vs {_fmt_ms(r[2])} floor" for r in rows],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}× the floor<extra></extra>",
    )
    if labels[0] == "Time per token":
        # Note 24: 60-80% of bandwidth is healthy, i.e. 1.25x to 1.67x the per-token floor.
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=1.25,
            x1=1.67,
            y0=-0.4,
            y1=0.4,
            fillcolor=OK,
            opacity=0.12,
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=1.46,
            y=-0.42,
            xref="x",
            yref="y",
            text="healthy band",
            showarrow=False,
            yanchor="bottom",
            font={"size": 11, "color": OK},
        )
    fig.add_vline(
        x=1,
        line_color=INK,
        line_width=1.5,
        annotation_text="physics floor",
        annotation_position="bottom right",
    )
    top = max([*ratios, 2.0]) * 1.45
    return _fig(
        fig,
        "One request against the physics floor (1× = as fast as physics allows)",
        xaxis={"title": "Times the floor", "range": [0, top]},
        yaxis={"autorange": "reversed"},
        height=60 + 70 * len(rows),
        showlegend=False,
    )


def _baseline_spread(d: dict[str, Any]) -> dict[str, Any] | None:
    samples = d.get("samples") or []
    if len(samples) < 3:
        return None
    fig = go.Figure()
    fig.add_box(
        x=samples,
        boxpoints="all",
        jitter=0.5,
        pointpos=0,
        marker={"color": ACCENT, "size": 6, "opacity": 0.7},
        line={"color": SECOND},
        fillcolor="rgba(0,0,0,0)",
        name="",
        hovertemplate="%{x:.2f} ms<extra></extra>",
    )
    return _fig(
        fig,
        f"Time per token across {len(samples)} identical requests",
        xaxis={"title": "Time per token (ms)"},
        yaxis={"showticklabels": False},
        height=200,
        showlegend=False,
    )


def _network_share(d: dict[str, Any]) -> dict[str, Any] | None:
    rtt, ttft = d.get("http_rtt_ms"), d.get("user_ttft_ms")
    if not rtt or not ttft:
        return None
    network = min(rtt, ttft)
    server = max(ttft - rtt, 0)
    fig = go.Figure()
    fig.add_bar(
        y=["First token"],
        x=[network],
        orientation="h",
        name="Network round trip",
        marker_color=MUTED,
        text=[_fmt_ms(network)],
        textposition="inside",
        width=0.45,
    )
    fig.add_bar(
        y=["First token"],
        x=[server],
        orientation="h",
        name="Server",
        marker_color=ACCENT,
        text=[_fmt_ms(server) if server > ttft * 0.08 else ""],
        textposition="inside",
        width=0.45,
    )
    return _fig(
        fig,
        f"Where the {_fmt_ms(ttft)} to the first token goes, from this location",
        barmode="stack",
        xaxis={"title": "Milliseconds"},
        height=190,
        yaxis={"showticklabels": False},
    )


def _fits(d: dict[str, Any]) -> dict[str, Any] | None:
    shapes = [s for s in d.get("shapes", []) if s.get("fits")]
    if not shapes:
        return None
    fig = go.Figure()
    fig.add_bar(
        y=[f"{s['name']} ({s['isl']:,} in, {s['osl']:,} out)" for s in shapes],
        x=[s["fits"] for s in shapes],
        orientation="h",
        marker_color=ACCENT,
        width=0.55,
        text=[f"{s['fits']:,.0f}" for s in shapes],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x:,.0f} requests<extra></extra>",
    )
    return _fig(
        fig,
        "Requests that fit in the KV cache at their full length",
        xaxis={"title": "Requests in memory at once"},
        yaxis={"autorange": "reversed"},
        height=80 + 52 * len(shapes),
        showlegend=False,
        margin={"l": 210},
    )


def _walls(d: dict[str, Any]) -> dict[str, Any] | None:
    walls: dict[str, float] = d.get("walls") or {}
    if not walls:
        return None
    items = sorted(walls.items(), key=lambda kv: kv[1])
    limiter = d.get("limiter")
    fig = go.Figure()
    fig.add_bar(
        y=[WALL_LABELS.get(k, k) for k, _ in items],
        x=[v for _, v in items],
        orientation="h",
        marker_color=[ACCENT if k == limiter else MUTED for k, _ in items],
        width=0.55,
        text=[f"{v:,.0f}" for _, v in items],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,.0f} users<extra></extra>",
    )
    if d.get("max_tested"):
        # Shapes on a log axis take log10 coordinates.
        fig.add_vline(
            x=math.log10(d["max_tested"]),
            line_dash="dot",
            line_color=SECOND,
            annotation_text="highest level tested",
            annotation_position="top",
        )
    return _fig(
        fig,
        "Each wall's limit; the lowest one is your capacity",
        xaxis={"title": "Users in flight (log scale)", "type": "log"},
        yaxis={"autorange": "reversed"},
        height=90 + 46 * len(items),
        showlegend=False,
        margin={"l": 150},
    )


# ---- concurrency sweep ----


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
            "size": 10,
            "line": {"color": "#fff", "width": 1.5},
        },
        line={"color": MUTED, "width": 2},
        name="users in flight",
        hovertemplate="%{text} users<br>%{x:,.0f} tok/s<br>%{y:.1f} ms<extra></extra>",
    )
    if slo:
        fig.add_hline(
            y=slo,
            line_dash="dash",
            line_color=BAD,
            opacity=0.7,
            annotation_text=f"target {slo:g} ms",
            annotation_position="top left",
        )
    return _fig(
        fig,
        f"Throughput against P{pct} time per token (labels: users in flight)",
        xaxis={"title": "Output tokens per second, all users"},
        yaxis={"title": f"P{pct} time per token (ms)", "rangemode": "tozero"},
        showlegend=False,
    )


def _throughput(pts: list[dict[str, Any]], op: int | None) -> dict[str, Any]:
    fig = go.Figure()
    fig.add_scatter(
        x=[p["concurrency"] for p in pts],
        y=[p["tok_s"] for p in pts],
        mode="lines+markers",
        line={"color": ACCENT, "width": 2.5},
        marker={"size": 8},
        fill="tozeroy",
        fillcolor="rgba(217,119,87,0.08)",
        name="tokens per second",
        hovertemplate="%{x} users: %{y:,.0f} tok/s<extra></extra>",
    )
    best = next((p for p in pts if p["concurrency"] == op), None)
    if best:
        # A marker, not a vline: shapes on log axes need log-space coordinates.
        fig.add_scatter(
            x=[best["concurrency"]],
            y=[best["tok_s"]],
            mode="markers+text",
            marker={"color": OK, "size": 14, "symbol": "star"},
            text=["best point within targets"],
            textposition="middle right",
            hoverinfo="skip",
        )
    return _fig(
        fig,
        "Throughput as users are added",
        xaxis={"title": "Users in flight (log scale)", "type": "log"},
        yaxis={"title": "Output tokens per second", "rangemode": "tozero"},
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
        name=f"P{pct} time per token",
        mode="lines+markers",
        line={"color": ACCENT, "width": 2.5},
    )
    fig.add_scatter(
        x=xs,
        y=[p["ttft_tail"] for p in pts],
        name=f"P{pct} time to first token (server side)",
        mode="lines+markers",
        line={"color": SECOND, "width": 2.5, "dash": "dot"},
        yaxis="y2",
    )
    if tpot:
        fig.add_hline(y=tpot, line_dash="dash", line_color=ACCENT, opacity=0.5)
    if ttft:
        fig.add_hline(y=ttft, line_dash="dash", line_color=SECOND, opacity=0.5, yref="y2")
    return _fig(
        fig,
        "Latency tails as users are added (dashed: your targets)",
        xaxis={"title": "Users in flight (log scale)", "type": "log"},
        yaxis={"title": "Time per token (ms)", "rangemode": "tozero"},
        yaxis2={
            "title": "Time to first token (ms)",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
            "showgrid": False,
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
        opacity=0.75,
    )
    fig.add_scatter(
        x=xs,
        y=[p.get("preemptions") or 0 for p in pts],
        name="Preemptions",
        mode="lines+markers",
        line={"color": BAD, "width": 2},
        yaxis="y2",
    )
    return _fig(
        fig,
        "Memory as users are added",
        xaxis={"title": "Users in flight (log scale)", "type": "log"},
        yaxis={"title": "KV cache peak (%)", "range": [0, 105]},
        yaxis2={
            "title": "Preemptions",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
            "showgrid": False,
        },
    )


# ---- request-rate sweep ----


def _rate_ttft(d: dict[str, Any]) -> dict[str, Any]:
    rows = d["rows"]
    pct = d["pct_label"]
    xs = [r["requested"] for r in rows]
    fig = go.Figure()
    fig.add_scatter(
        x=xs,
        y=[r["ttft_tail"] for r in rows],
        mode="lines+markers",
        name=f"P{pct} time to first token",
        line={"color": ACCENT, "width": 2.5},
        marker={"size": 9, "color": [BAD if r["past_capacity"] else ACCENT for r in rows]},
        hovertemplate="%{x:.2f} req/s: %{y:,.0f} ms<extra></extra>",
    )
    fig.add_scatter(
        x=xs,
        y=[r["completed"] for r in rows],
        mode="lines+markers",
        name="Completed req/s",
        line={"color": SECOND, "dash": "dot"},
        yaxis="y2",
        hovertemplate="%{y:.2f} completed req/s<extra></extra>",
    )
    fig.add_scatter(
        x=xs,
        y=xs,
        mode="lines",
        name="Requested req/s",
        line={"color": MUTED, "width": 1},
        yaxis="y2",
        hoverinfo="skip",
    )
    if d.get("slo_ttft_ms"):
        fig.add_hline(
            y=d["slo_ttft_ms"],
            line_dash="dash",
            line_color=BAD,
            opacity=0.6,
            annotation_text=f"target {d['slo_ttft_ms']:g} ms",
            annotation_position="top left",
        )
    return _fig(
        fig,
        "Random arrivals: first-token time as the rate rises (red: queue keeps growing)",
        xaxis={"title": "Requested requests per second"},
        yaxis={"title": f"P{pct} time to first token (ms)", "rangemode": "tozero"},
        yaxis2={
            "title": "Requests per second",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
            "showgrid": False,
        },
    )


def _rate_trend(d: dict[str, Any]) -> dict[str, Any]:
    fig = go.Figure()
    rows = [r for r in d["rows"] if r.get("trend") and r["arrival"] == "poisson"]
    palette = ["#e8c3b3", "#e0a088", "#d97757", "#b5533a", "#7d3324"]
    for i, r in enumerate(rows):
        t = r["trend"]
        fig.add_scatter(
            x=[x for x, _ in t],
            y=[y for _, y in t],
            mode="lines+markers",
            name=f"{r['requested']:.2f} req/s",
            line={"color": palette[min(i, len(palette) - 1)], "width": 2},
            hovertemplate="%{x:.0f} s: %{y:,.0f} ms<extra></extra>",
        )
    return _fig(
        fig,
        "First-token time during each run (rising lines: past capacity)",
        xaxis={"title": "Seconds into the run"},
        yaxis={"title": "Median time to first token (ms)", "rangemode": "tozero"},
    )


def _bursts(d: dict[str, Any]) -> dict[str, Any]:
    pairs = d["pairs"]
    labels = [f"{p['rate']:.2f} req/s" for p in pairs]
    fig = go.Figure()
    fig.add_bar(
        x=labels, y=[p["smooth_ms"] for p in pairs], name="Random (Poisson)", marker_color=MUTED
    )
    fig.add_bar(x=labels, y=[p["bursty_ms"] for p in pairs], name="Bursty", marker_color=ACCENT)
    return _fig(
        fig,
        f"P{d['pct_label']} time to first token: steady vs bursty arrivals",
        barmode="group",
        yaxis={"title": "Milliseconds", "rangemode": "tozero"},
        xaxis={"title": "Average arrival rate"},
    )

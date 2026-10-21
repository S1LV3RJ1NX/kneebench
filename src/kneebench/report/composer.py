"""Compose a report: diagnosis first, evidence second, education alongside, next steps last.

Which sections appear and in what order comes from the findings and the team's product
type (content/profiles.yaml); all words come from the content layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kneebench import __version__, content
from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import RULES_VERSION, Finding, Status
from kneebench.content.render import (
    RenderedAction,
    RenderedFinding,
    TermTracker,
    make_env,
    ms,
    num,
    render_finding,
)

INFO_RULES_IN_CONTEXT = ("T01", "T02", "T03", "T05", "B05")
# Healthy baseline findings still get a card here, so their charts stay one click away.
CHARTED_OK_IN_CONTEXT = ("B01", "B04")
# Findings that describe the load curve and its walls get their own section.
CAPACITY_RULES = (
    "W03",
    "C02",
    "R01",
    "W02",
    "R02",
    "R04",
    "W01",
    "C01",
    "C04",
    "W05",
    "W09",
    "W08",
)
# Note 26: the shape map leads; the rest explain why shapes differ.
SHAPE_RULES = ("S06", "S01", "S02", "S03", "S04", "S07", "S05")
CAVEAT_RULES = ("H06",)
LIMITER_TEXT = {
    "slo_tpot": "per-token latency crosses your target",
    "slo_ttft": "time to first token crosses your target",
    "saturation": "the GPU stops adding throughput",
    "memory": "the KV cache is full",
    "max_num_seqs": "the server's max_num_seqs cap is reached",
}


@dataclass(frozen=True)
class Tile:
    key: str
    label: str
    status: str  # ok, warn, bad, none
    text: str


@dataclass(frozen=True)
class NextStep:
    rank: int
    rule_id: str
    reason: str
    action: RenderedAction


@dataclass(frozen=True)
class AppendixTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass
class Report:
    run_id: int
    title: str
    generated_at: str
    summary: dict[str, Any]
    style: str
    verdict: list[str]
    scorecard: list[Tile]
    problems: list[RenderedFinding]
    capacity: list[RenderedFinding]
    shapes: list[RenderedFinding]
    context: list[RenderedFinding]
    passed: list[RenderedFinding]
    skipped: list[RenderedFinding]
    next_steps: list[NextStep]
    caveats: list[str]
    glossary: list[tuple[str, str]]
    appendix: list[AppendixTable]
    sections: dict[str, str]
    findings_json: list[dict[str, Any]] = field(default_factory=list)
    # Set when the report judges the run against targets other than the ones it ran with.
    what_if: str | None = None


def common_vars(ctx: RunContext) -> dict[str, Any]:
    profiles = content.load("profiles.yaml")["profiles"]
    count = ctx.setup.gpu_count
    return {
        "model": ctx.hf_model_id or ctx.served_model or "the model",
        "served_model": ctx.served_model,
        "gpu_name": ctx.setup.gpu_name,
        "gpu_label": f"{count}× {ctx.setup.gpu_name}" if count > 1 else ctx.setup.gpu_name,
        "gpu_count": count,
        "product_label": profiles[ctx.workload.product.value]["label"],
        "vllm_version": ctx.vllm_version,
        "slo": ctx.workload.slo,
    }


def _weight(finding: Finding, product: str) -> int:
    profiles = content.load("profiles.yaml")
    weights: dict[str, int] = profiles["profiles"][product]["weights"]
    default: int = profiles["default_weight"]
    return max((weights.get(t, default) for t in finding.tags), default=default)


def _next_steps(problems: list[RenderedFinding]) -> list[NextStep]:
    """Server changes first, then test fixes, each in problem order.

    One step per flag (or per title, without a flag): two findings can ask for the same
    setting, and the higher-ranked one wins rather than giving two different values.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    ordered: dict[str, list[tuple[list[str], str, RenderedAction]]] = {"server": [], "test": []}
    for r in problems:
        for action in r.actions:
            key = (action.kind, _setting(action))
            if key in seen:
                seen[key].append(r.rule_id)  # the kept step lists every finding behind it
                continue
            seen[key] = [r.rule_id]
            ordered.setdefault(action.kind, []).append((seen[key], r.status_line, action))
    steps: list[NextStep] = []
    for group in ordered.values():
        for rank, (rule_ids, reason, action) in enumerate(group, start=1):
            steps.append(NextStep(rank, ", ".join(dict.fromkeys(rule_ids)), reason, action))
    return steps


def _setting(action: RenderedAction) -> str:
    """What an action changes: its group, its first --flag, the whole command, or its title."""
    if action.group:
        return action.group
    if not action.flag:
        return action.title
    return next((w for w in action.flag.split() if w.startswith("--")), action.flag)


def _by(findings: list[Finding], rule_id: str) -> Finding | None:
    return next((f for f in findings if f.rule_id == rule_id), None)


def compose(
    ctx: RunContext,
    findings: list[Finding],
    now: str | None = None,
    style: str = "md",
    what_if: dict[str, Any] | None = None,
) -> Report:
    tracker = TermTracker(style)
    env = make_env(tracker)
    common = common_vars(ctx)
    text = content.load("report.yaml")
    product = ctx.workload.product.value

    def render_text(template: str, **extra: Any) -> str:
        return " ".join(env.from_string(template).render(**{**common, **extra}).split())

    verdict = _verdict(ctx, findings, text["verdict"], render_text)
    scorecard = _scorecard(ctx, findings, text["scorecard"], render_text)

    score = {f.rule_id: f.severity * _weight(f, product) for f in findings}
    problems_f = sorted(
        (f for f in findings if f.status in (Status.BAD, Status.WARN)),
        key=lambda f: (-score[f.rule_id], f.rule_id),
    )
    charted_ok = [
        f
        for rid in CHARTED_OK_IN_CONTEXT
        for f in findings
        if f.rule_id == rid and f.status is Status.OK
    ]
    context_f = charted_ok + sorted(
        (f for f in findings if f.status is Status.INFO and f.rule_id in INFO_RULES_IN_CONTEXT),
        key=lambda f: (-_weight(f, product), f.rule_id),
    )
    capacity_f = [
        f
        for rid in CAPACITY_RULES
        for f in findings
        if f.rule_id == rid and f.status in (Status.INFO, Status.OK)
    ]
    shapes_f = sorted(
        (f for f in findings if f.rule_id in SHAPE_RULES and f.status in (Status.INFO, Status.OK)),
        # The map is the overview, so it stays on top whatever the profile.
        key=lambda f: (f.rule_id != "S06", -_weight(f, product), SHAPE_RULES.index(f.rule_id)),
    )
    problem_ids = {f.rule_id for f in problems_f}
    # Render in reading order so each term is defined where it first appears.
    problems = [render_finding(f, common, env) for f in problems_f]
    capacity_r = [render_finding(f, common, env) for f in capacity_f]
    shapes_r = [render_finding(f, common, env) for f in shapes_f]
    context_r = [render_finding(f, common, env) for f in context_f]
    shown = (
        problem_ids
        | {f.rule_id for f in capacity_f}
        | {f.rule_id for f in shapes_f}
        | {f.rule_id for f in charted_ok}
    )
    passed = [
        render_finding(f, common, env)
        for f in findings
        if f.status is Status.OK and f.rule_id not in shown
    ]
    skipped = [render_finding(f, common, env) for f in findings if f.status is Status.SKIP]

    steps = _next_steps(problems)

    caveats = [render_text(c) for c in text["caveats"]["always"]]
    rtt = ctx.http_rtt_ms
    if rtt is not None and rtt >= 20:
        caveats.append(render_text(text["caveats"]["remote_client"], http_rtt_ms=rtt))
    if not ctx.metrics_available:
        caveats.append(render_text(text["caveats"]["no_metrics"]))
    for f in findings:
        if f.rule_id in CAVEAT_RULES and f.status is Status.INFO:
            caveats.append(render_finding(f, common, env).status_line)

    return Report(
        run_id=ctx.run_id,
        title=f"kneebench report: {common['model']} on {common['gpu_label']}",
        generated_at=now or datetime.now(UTC).isoformat(timespec="seconds"),
        style=style,
        summary={
            "model": common["model"],
            "served_model": ctx.served_model,
            "gpu": common["gpu_label"],
            "vllm_version": ctx.vllm_version or "unknown",
            "product": common["product_label"],
            "run_started": ctx.started_at,
            "run_status": ctx.status,
            "run_error": ctx.error,
            "kneebench_version": __version__,
            "aiperf_version": ctx.aiperf_version,
            "rules_version": RULES_VERSION,
            "max_model_len": ctx.max_model_len,
            "slo": {
                "ttft_ms": ctx.workload.slo.ttft_ms,
                "tpot_ms": ctx.workload.slo.tpot_ms,
                "percentile": ctx.workload.slo.percentile,
            },
        },
        verdict=verdict,
        scorecard=scorecard,
        problems=problems,
        capacity=capacity_r,
        shapes=shapes_r,
        context=context_r,
        passed=passed,
        skipped=skipped,
        next_steps=steps,
        caveats=caveats,
        glossary=sorted(tracker.used.items()),
        appendix=_appendix(ctx),
        sections=text["sections"],
        findings_json=[f.to_json() for f in findings],
        what_if=render_text(text["what_if"], **what_if) if what_if else None,
    )


VERDICT_MAX_SENTENCES = 3  # the verdict is two or three plain sentences


def _verdict(
    ctx: RunContext, findings: list[Finding], text: dict[str, str], render: Any
) -> list[str]:
    """The few sentences that matter most, in priority order."""
    t06, t07, b01 = _by(findings, "T06"), _by(findings, "T07"), _by(findings, "B01")
    h03, h04 = _by(findings, "H03"), _by(findings, "H04")
    mismatch = bool((t06 and t06.status is Status.BAD) or (t07 and t07.status is Status.BAD))

    decode = None
    # A health verdict against the wrong GPU would contradict the mismatch sentence.
    if b01 is not None and b01.status is not Status.SKIP and not mismatch:
        key = {
            Status.OK: "decode_healthy",
            Status.WARN: "decode_slow",
            Status.BAD: "decode_very_slow",
        }[b01.status]
        decode = render(text[key], **b01.data)
    network = None
    if h03 is not None and h03.status is Status.BAD and h03.data.get("unmeasurable"):
        network = render(text["network_dominates"], **h03.data)
    short = render(text["short_outputs"]) if h04 is not None and h04.status is Status.BAD else None
    first = render(text["gpu_mismatch"]) if mismatch else None

    if ctx.points_in("concurrency"):
        ranked = [first, _capacity_sentence(ctx, findings, text, render), network, short, decode]
        # The capacity verdict is already two sentences, so it gets one companion at most.
        limit = VERDICT_MAX_SENTENCES - 1
    else:
        ranked = [first, decode, network, short, render(text["baseline_only_next"])]
        limit = VERDICT_MAX_SENTENCES
    return [s for s in ranked if s][:limit]


def _capacity_sentence(
    ctx: RunContext, findings: list[Finding], text: dict[str, str], render: Any
) -> str | None:
    w03, c03 = _by(findings, "W03"), _by(findings, "C03")
    if w03 is None or c03 is None:
        return None
    shape = ctx.theory.shape(ctx.primary_shape) or ctx.theory.shape("baseline")
    assert shape is not None
    slo = ctx.workload.slo
    data = dict(w03.data)
    limiter = str(data["limiter"])
    kv_at_break = _kv_near(ctx, float(data["c_star"]))
    extra = {
        "isl": shape.isl,
        "osl": shape.osl,
        "limiter_text": LIMITER_TEXT[limiter],
        "slo_tpot_ms": slo.tpot_ms,
        "slo_ttft_ms": slo.ttft_ms,
        "pct_label": slo.percentile,
        "kv_at_break": kv_at_break,
        "memory_not_full": limiter != "memory" and kv_at_break is not None and kv_at_break < 0.9,
    }
    if c03.status is Status.BAD:
        key = "capacity_none"
    elif c03.status is Status.WARN:
        key = "capacity_above_range"
    else:
        key = "capacity"
    return str(render(text[key], **{**data, **extra}))


def _kv_near(ctx: RunContext, concurrency: float) -> float | None:
    """KV cache peak at the tested level closest to a concurrency."""
    from kneebench.analysis.sweep import curve

    pts = curve(ctx.points_in("concurrency"))
    if not pts:
        return None
    near = min(pts, key=lambda p: abs(p.concurrency - concurrency))
    return near.kv_usage_max


def _scorecard(
    ctx: RunContext, findings: list[Finding], text: dict[str, Any], render: Any
) -> list[Tile]:
    tiles: list[Tile] = []

    b01, t06, t07 = _by(findings, "B01"), _by(findings, "T06"), _by(findings, "T07")
    t = text["baseline"]
    if (t06 and t06.status is Status.BAD) or (t07 and t07.status is Status.BAD):
        tiles.append(Tile("baseline", t["label"], "bad", t["mismatch"]))
    elif b01 is None or b01.status is Status.SKIP:
        tiles.append(Tile("baseline", t["label"], "none", t["none"]))
    else:
        tiles.append(
            Tile("baseline", t["label"], b01.status.value, render(t[b01.status.value], **b01.data))
        )

    c03, w03, w08 = _by(findings, "C03"), _by(findings, "W03"), _by(findings, "W08")
    cap, head = text["capacity"], text["headroom"]
    if c03 is None or c03.status is Status.SKIP or w03 is None:
        tiles.append(Tile("capacity", cap["label"], "none", cap["none"]))
        tiles.append(Tile("headroom", head["label"], "none", head["none"]))
    else:
        tiles.append(
            Tile(
                "capacity",
                cap["label"],
                c03.status.value,
                render(cap[c03.status.value], **{**c03.data, **w03.data}),
            )
        )
        if w08 is not None and c03.status is not Status.BAD:
            tiles.append(Tile("headroom", head["label"], "ok", render(head["ok"], **w08.data)))
        else:
            tiles.append(Tile("headroom", head["label"], "none", head["none"]))

    m = text["memory"]
    t04, w04, w06 = _by(findings, "T04"), _by(findings, "W04"), _by(findings, "W06")
    if w04 is not None and w04.status is Status.BAD:
        tiles.append(Tile("memory", m["label"], "bad", render(m["preempting"], **w04.data)))
    elif w06 is not None and w06.status in (Status.WARN, Status.BAD):
        tiles.append(Tile("memory", m["label"], "warn", render(m["cap_too_high"], **w06.data)))
    elif w04 is not None and w04.status is Status.OK:
        tiles.append(Tile("memory", m["label"], "ok", render(m["no_preemption"], **w04.data)))
    elif t04 is None or t04.status is Status.SKIP:
        tiles.append(Tile("memory", m["label"], "none", m["none"]))
    else:
        tiles.append(
            Tile("memory", m["label"], t04.status.value, render(m[t04.status.value], **t04.data))
        )

    reliability = [
        f for f in findings if "reliability" in f.tags and f.status in (Status.BAD, Status.WARN)
    ]
    r = text["reliability"]
    if not reliability:
        tiles.append(Tile("reliability", r["label"], "ok", r["ok"]))
    else:
        worst = "bad" if any(f.status is Status.BAD for f in reliability) else "warn"
        tiles.append(
            Tile("reliability", r["label"], worst, render(r[worst], count=len(reliability)))
        )
    return tiles


def _appendix(ctx: RunContext) -> list[AppendixTable]:
    rows = []
    for p in ctx.points:
        rows.append(
            (
                str(p.seq),
                p.phase,
                p.status,
                num(p.isl),
                num(p.osl),
                num(p.concurrency) if p.concurrency else "n/a",
                num(p.m("request_count", "avg"), 0),
                ms(p.m("time_to_first_token", "p50")),
                ms(p.m("time_to_first_token", "p99")),
                ms(p.m("inter_token_latency", "p50")),
                ms(p.m("inter_token_latency", "p99")),
                ms(p.m("request_latency", "p50")),
                num(p.m("output_token_throughput", "avg"), 1),
                num(p.m("request_error_rate", "avg"), 1),
            )
        )
    tables = [
        AppendixTable(
            "Measured points",
            (
                "#",
                "Phase",
                "Status",
                "Prompt tokens",
                "Output tokens",
                "Concurrency",
                "Requests",
                "TTFT P50",
                "TTFT P99",
                "TPOT P50",
                "TPOT P99",
                "E2E P50",
                "Output tok/s",
                "Error %",
            ),
            tuple(rows),
        )
    ]
    net = ctx.network or {}
    if net:
        tables.append(
            AppendixTable(
                "Client distance",
                ("DNS", "TCP connect", "TLS handshake", "HTTP round trip (/health)"),
                (
                    (
                        ms(net.get("dns_ms")),
                        ms(net.get("tcp_connect_ms")),
                        ms(net.get("tls_handshake_ms")),
                        ms(net.get("http_rtt_ms")),
                    ),
                ),
            )
        )
    t = ctx.theory
    tables.append(
        AppendixTable(
            "Setup",
            ("Item", "Value"),
            (
                ("Model", str(t.model_hf_id or ctx.served_model)),
                ("Parameters", num(t.model_params)),
                ("Weight bytes", num(t.model_weight_bytes)),
                (
                    "Weight dtype",
                    t.model_weight_dtype
                    + (
                        f" ({t.model_quantization}, {t.model_quant_scheme})"
                        if t.model_quant_scheme
                        else f" ({t.model_quantization})"
                        if t.model_quantization
                        else ""
                    ),
                ),
                ("KV bytes per token", num(t.kv_bytes_per_token)),
                ("GPU", f"{t.gpu_count}× {t.gpu_name}"),
                ("GPU memory (GiB)", num(t.gpu_memory_gib, 1)),
                ("GPU bandwidth (TB/s)", num(t.gpu_bandwidth_tbps, 2)),
                ("KV cache pool (tokens)", num(t.pool_tokens)),
                ("max_model_len", num(t.max_model_len)),
                (
                    "gpu_memory_utilization",
                    num(t.gpu_memory_utilization, 2) if t.gpu_memory_utilization else "n/a",
                ),
                ("Default max_num_seqs for this vLLM and GPU", num(t.default_max_num_seqs)),
                ("vLLM", str(ctx.vllm_version or "unknown")),
                ("AIPerf", ctx.aiperf_version),
            ),
        )
    )
    return tables

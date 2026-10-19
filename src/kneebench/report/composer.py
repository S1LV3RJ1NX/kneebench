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
# Findings that describe the load curve and its walls get their own section.
CAPACITY_RULES = ("W03", "C02", "W02", "W01", "C01", "C04", "W05", "W09", "W08")
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
    verdict: list[str]
    scorecard: list[Tile]
    problems: list[RenderedFinding]
    capacity: list[RenderedFinding]
    context: list[RenderedFinding]
    passed: list[RenderedFinding]
    skipped: list[RenderedFinding]
    next_steps: list[NextStep]
    caveats: list[str]
    glossary: list[tuple[str, str]]
    appendix: list[AppendixTable]
    sections: dict[str, str]
    findings_json: list[dict[str, Any]] = field(default_factory=list)


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


def _by(findings: list[Finding], rule_id: str) -> Finding | None:
    return next((f for f in findings if f.rule_id == rule_id), None)


def compose(ctx: RunContext, findings: list[Finding], now: str | None = None) -> Report:
    tracker = TermTracker("md")
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
    context_f = sorted(
        (f for f in findings if f.status is Status.INFO and f.rule_id in INFO_RULES_IN_CONTEXT),
        key=lambda f: (-_weight(f, product), f.rule_id),
    )
    capacity_f = [
        f
        for rid in CAPACITY_RULES
        for f in findings
        if f.rule_id == rid and f.status in (Status.INFO, Status.OK)
    ]
    problem_ids = {f.rule_id for f in problems_f}
    # Render in reading order so each term is defined where it first appears.
    problems = [render_finding(f, common, env) for f in problems_f]
    capacity_r = [render_finding(f, common, env) for f in capacity_f]
    context_r = [render_finding(f, common, env) for f in context_f]
    shown = problem_ids | {f.rule_id for f in capacity_f}
    passed = [
        render_finding(f, common, env)
        for f in findings
        if f.status is Status.OK and f.rule_id not in shown
    ]
    skipped = [render_finding(f, common, env) for f in findings if f.status is Status.SKIP]

    steps: list[NextStep] = []
    for r in problems:
        for action in r.actions:
            steps.append(NextStep(len(steps) + 1, r.rule_id, r.status_line, action))

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
        summary={
            "model": common["model"],
            "served_model": ctx.served_model,
            "gpu": common["gpu_label"],
            "vllm_version": ctx.vllm_version or "unknown",
            "product": common["product_label"],
            "run_started": ctx.started_at,
            "run_status": ctx.status,
            "kneebench_version": __version__,
            "aiperf_version": ctx.aiperf_version,
            "rules_version": RULES_VERSION,
            "max_model_len": ctx.max_model_len,
        },
        verdict=verdict,
        scorecard=scorecard,
        problems=problems,
        capacity=capacity_r,
        context=context_r,
        passed=passed,
        skipped=skipped,
        next_steps=steps,
        caveats=caveats,
        glossary=sorted(tracker.used.items()),
        appendix=_appendix(ctx),
        sections=text["sections"],
        findings_json=[f.to_json() for f in findings],
    )


def _verdict(
    ctx: RunContext, findings: list[Finding], text: dict[str, str], render: Any
) -> list[str]:
    sentences: list[str] = []
    t06, t07, b01 = _by(findings, "T06"), _by(findings, "T07"), _by(findings, "B01")
    mismatch = (t06 and t06.status is Status.BAD) or (t07 and t07.status is Status.BAD)
    if mismatch:
        sentences.append(render(text["gpu_mismatch"]))
    capacity = _capacity_sentence(ctx, findings, text, render)
    if capacity:
        sentences.append(capacity)
    # A health verdict against the wrong GPU would contradict the sentence above.
    if b01 is not None and b01.status is not Status.SKIP and not mismatch:
        key = {
            Status.OK: "decode_healthy",
            Status.WARN: "decode_slow",
            Status.BAD: "decode_very_slow",
        }[b01.status]
        sentences.append(render(text[key], **b01.data))
    h03, h04 = _by(findings, "H03"), _by(findings, "H04")
    if h03 is not None and h03.status is Status.BAD and h03.data.get("unmeasurable"):
        sentences.append(render(text["network_dominates"], **h03.data))
    if h04 is not None and h04.status is Status.BAD:
        sentences.append(render(text["short_outputs"]))
    if not ctx.points_in("concurrency"):
        sentences.append(render(text["baseline_only_next"]))
    return sentences


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
                render(cap[c03.status.value], **{**w03.data, **c03.data}),
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
                    + (f" ({t.model_quantization})" if t.model_quantization else ""),
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

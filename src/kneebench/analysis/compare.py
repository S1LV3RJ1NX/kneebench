"""Compare two runs: what changed in the setup, and what that did to the numbers (note 24)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, Rule, evaluate, info, ok, warn

# Changes smaller than this are treated as run-to-run noise.
NOISE = 0.05

# (key, label, unit, better): +1 when higher is better, -1 when lower is better.
METRICS: tuple[tuple[str, str, str, int], ...] = (
    ("capacity_users", "Users within your targets", "", 1),
    ("op_tok_s", "Tokens per second within your targets", "", 1),
    ("peak_tok_s", "Peak tokens per second", "", 1),
    ("rate_rps", "Requests per second with random arrivals", "", 1),
    ("tpot_ms", "Time per output token, one user", "ms", -1),
    ("ttft_ms", "Time to first token, one user, without the network", "ms", -1),
    ("mbu_pct", "Memory bandwidth used, one user", "%", 1),
)


@dataclass(frozen=True)
class RunFacts:
    run_id: int
    label: str | None
    started_at: str
    setup: dict[str, Any]
    workload: dict[str, Any]
    metrics: dict[str, float | None]
    curve: list[dict[str, Any]]
    max_tested: int | None


# Labels read inside sentences ("the GPU and the vLLM version changed"); tables capitalize them.
SETUP_LABELS = {
    "model": "model",
    "gpu": "GPU",
    "vllm_version": "vLLM version",
    "max_model_len": "context window",
    "max_num_seqs": "max_num_seqs",
    "enforce_eager": "enforce_eager",
    "kv_pool_tokens": "KV cache tokens",
}
WORKLOAD_LABELS = {
    "product": "traffic type",
    "shape": "main request size",
    "targets": "latency targets",
}


def facts(ctx: RunContext, label: str | None = None) -> RunFacts:
    found = {f.rule_id: f.data for f in evaluate(ctx)}
    b01, w03, r01, c02 = (found.get(k, {}) for k in ("B01", "W03", "R01", "C02"))
    points = c02.get("points") or []
    shape = ctx.theory.shape(ctx.primary_shape)
    slo = ctx.workload.slo
    count = ctx.setup.gpu_count
    return RunFacts(
        run_id=ctx.run_id,
        label=label,
        started_at=ctx.started_at,
        setup={
            "model": ctx.hf_model_id or ctx.served_model,
            "gpu": f"{count}× {ctx.setup.gpu_name}" if count > 1 else ctx.setup.gpu_name,
            "vllm_version": ctx.vllm_version,
            "max_model_len": ctx.max_model_len,
            "max_num_seqs": ctx.setup.max_num_seqs or "default",
            "enforce_eager": {True: "on", False: "off", None: "unknown"}[ctx.setup.enforce_eager],
            "kv_pool_tokens": ctx.theory.pool_tokens,
        },
        workload={
            "product": ctx.workload.product.value,
            "shape": f"{shape.isl}/{shape.osl}" if shape else None,
            "targets": (
                f"TTFT {slo.ttft_ms or '-'} ms, TPOT {slo.tpot_ms or '-'} ms at P{slo.percentile}"
            ),
        },
        metrics={
            "capacity_users": w03.get("op"),
            "op_tok_s": w03.get("op_tok_s"),
            "peak_tok_s": max((p["tok_s"] for p in points if p.get("tok_s")), default=None),
            "rate_rps": r01.get("max_rps"),
            "tpot_ms": b01.get("tpot_ms"),
            "ttft_ms": b01.get("server_ttft_ms"),
            "mbu_pct": b01["mbu"] * 100 if b01.get("mbu") else None,
        },
        curve=[p for p in points if p.get("tok_s")],
        max_tested=w03.get("max_tested"),
    )


@dataclass(frozen=True)
class CompareContext:
    a: RunFacts  # before
    b: RunFacts  # after


def changes(a: dict[str, Any], b: dict[str, Any], labels: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {"key": k, "label": labels[k], "a": a.get(k), "b": b.get(k)}
        for k in labels
        if a.get(k) != b.get(k)
    ]


def metric_rows(cc: CompareContext) -> list[dict[str, Any]]:
    rows = []
    for key, label, unit, better in METRICS:
        va, vb = cc.a.metrics.get(key), cc.b.metrics.get(key)
        if va is None and vb is None:
            continue
        ratio = vb / va if va and vb is not None else None
        if ratio is None:
            verdict = "unknown"
        elif abs(ratio - 1) < NOISE:
            verdict = "same"
        else:
            verdict = "better" if (ratio > 1) == (better > 0) else "worse"
        rows.append(
            {
                "key": key,
                "label": label,
                "title": f"{label} ({unit})" if unit else label,
                "a": va,
                "b": vb,
                "ratio": ratio,
                "verdict": verdict,
            }
        )
    return rows


CompareFn = Callable[[CompareContext], Finding | None]
COMPARE_REGISTRY: dict[str, Rule] = {}


def compare_rule(rule_id: str, *, tags: set[str], source: str) -> Callable[[CompareFn], CompareFn]:
    def register(fn: CompareFn) -> CompareFn:
        COMPARE_REGISTRY[rule_id] = Rule(rule_id, "X", frozenset(tags), source, fn)  # type: ignore[arg-type]
        return fn

    return register


@compare_rule("X01", tags={"setup"}, source="24#3")
def what_changed(cc: CompareContext) -> Finding:
    setup = changes(cc.a.setup, cc.b.setup, SETUP_LABELS)
    workload = changes(cc.a.workload, cc.b.workload, WORKLOAD_LABELS)
    data = {"setup": setup, "workload": workload, "n_setup": len(setup)}
    if workload:
        return warn("X01", case="different_test", **data)
    if len(setup) > 1:
        return warn("X01", case="several", **data)
    if not setup:
        return info("X01", case="same", **data)
    return ok("X01", case="one", change=setup[0], **data)


@compare_rule("X02", tags={"capacity"}, source="24#6")
def what_it_did(cc: CompareContext) -> Finding | None:
    rows = metric_rows(cc)
    comparable = not changes(cc.a.workload, cc.b.workload, WORKLOAD_LABELS)
    known = [r for r in rows if r["verdict"] != "unknown"]
    if not known:
        return None
    better = [r for r in known if r["verdict"] == "better"]
    worse = [r for r in known if r["verdict"] == "worse"]
    headline = next((r for r in known if r["key"] == "capacity_users"), known[0])
    data = {
        "rows": rows,
        "better": [r["label"] for r in better],
        "worse": [r["label"] for r in worse],
        "headline": headline,
        "noise_pct": NOISE * 100,
        "comparable": comparable,
        # Capacity is measured on a doubling grid; one step is a factor of two.
        "grid_note": headline["key"] == "capacity_users" and headline["verdict"] != "same",
    }
    if not comparable:
        return info("X02", case="not_comparable", **data)
    if worse:
        return warn("X02", case="worse", **data)
    if better:
        return ok("X02", case="better", **data)
    return info("X02", case="same", **data)


def compare(
    a: RunContext, b: RunContext, label_a: str | None = None, label_b: str | None = None
) -> tuple[CompareContext, list[Finding]]:
    cc = CompareContext(facts(a, label_a), facts(b, label_b))
    findings = []
    for r in sorted(COMPARE_REGISTRY.values(), key=lambda r: r.id):
        found = r.fn(cc)  # type: ignore[arg-type]
        if found is not None:
            findings.append(Finding(found.rule_id, found.status, found.data, r.tags))
    return cc, findings

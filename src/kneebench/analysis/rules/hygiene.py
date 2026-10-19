"""H rules: is the test itself trustworthy? (notes 24 to 27, plus client distance)."""

from __future__ import annotations

import statistics

from kneebench.analysis.context import UNRELIABLE_NETWORK_SHARE, RunContext
from kneebench.analysis.engine import Finding, bad, info, ok, rule, skip, warn
from kneebench.analysis.rules.concurrency import sweep
from kneebench.analysis.sweep import GATEWAY_CODES
from kneebench.servers.vllm import version_status

# Note 25 section 6: the client must actually hold the requested load.
CLIENT_CONCURRENCY_MIN = 0.9
# Requests in flight at the client but not at the server point at something in between.
SERVER_CONCURRENCY_MIN = 0.8
# Failed requests that all die after the same time look like a gateway timeout.
TIMEOUT_CLUSTER_MIN_S = 10.0
TIMEOUT_CLUSTER_SPREAD = 0.05

# Share of user-seen TTFT spent on the network before TTFT findings get a caveat.
NETWORK_SHARE_WARN = 0.30
NETWORK_SHARE_BAD = UNRELIABLE_NETWORK_SHARE
# Past this, the network takes longer than the server's whole first-token work.
NETWORK_SHARE_DOMINATES = 0.90
# Note 26 section 5: forced outputs that come back short mean ignore_eos was not honored.
OUTPUT_LENGTH_MIN_RATIO = 0.95
OSL_WARNING = "Output Sequence Length Mismatch"
TOKEN_WARNING = "Token Count Discrepancy"


@rule("H03", tags={"reliability", "network", "ttft"}, source="25#6")
def client_distance(ctx: RunContext) -> Finding | None:
    base = ctx.baseline
    net = ctx.network or {}
    rtt = net.get("http_rtt_ms")
    if base is None:
        return None
    ttft = base.m("time_to_first_token", "p50")
    if rtt is None or not ttft:
        return skip("H03", "no_network_probe")
    share = min(rtt / ttft, 1.0)
    data = {
        "http_rtt_ms": rtt,
        "tcp_connect_ms": net.get("tcp_connect_ms"),
        "tls_handshake_ms": net.get("tls_handshake_ms"),
        "user_ttft_ms": ttft,
        "server_ttft_ms": ctx.server_ttft_ms(base),
        "share": share,
        "unmeasurable": share >= NETWORK_SHARE_BAD,
        "dominates": share >= NETWORK_SHARE_DOMINATES,
        "aiperf_rtt_ms": base.m("network_rtt", "avg"),
    }
    if share >= NETWORK_SHARE_BAD:
        return bad("H03", **data)
    if share >= NETWORK_SHARE_WARN:
        return warn("H03", **data)
    return ok("H03", **data)


@rule("H04", tags={"reliability", "shape"}, source="26#5")
def output_length(ctx: RunContext) -> Finding | None:
    checked = [p for p in ctx.points if p.status == "done"]
    if not checked:
        return None
    short = []
    for p in checked:
        mean_osl = p.m("output_sequence_length", "avg")
        mismatches = p.m("osl_mismatch_count", "avg") or 0
        ratio = (mean_osl / p.osl) if mean_osl else None
        if p.has_warning(OSL_WARNING) or (ratio is not None and ratio < OUTPUT_LENGTH_MIN_RATIO):
            short.append(
                {
                    "phase": p.phase,
                    "requested": p.osl,
                    "mean": mean_osl,
                    "ratio": ratio,
                    "mismatches": mismatches,
                }
            )
    if short:
        return bad("H04", points=short, checked=len(checked))
    return ok("H04", checked=len(checked))


@rule("H05", tags={"reliability", "tokenizer"}, source="26#5")
def token_count_discrepancy(ctx: RunContext) -> Finding | None:
    flagged = [p.phase for p in ctx.points if p.has_warning(TOKEN_WARNING)]
    if not flagged:
        return None
    return warn("H05", phases=flagged, tokenizer=ctx.hf_model_id)


@rule("H07", tags={"reliability", "version"}, source="24#3")
def versions(ctx: RunContext) -> Finding:
    status = version_status(ctx.vllm_version)
    data = {
        "vllm_version": ctx.vllm_version,
        "aiperf_version": ctx.aiperf_version,
        "kneebench_version": ctx.kneebench_version,
        "tested_min": status.tested_min,
        "tested_max_exclusive": status.tested_max_exclusive,
    }
    if ctx.vllm_version is None:
        return warn("H07", unknown=True, **data)
    return ok("H07", **data) if status.in_tested_range else warn("H07", unknown=False, **data)


@rule("H08", tags={"reliability", "memory"}, source="27#1")
def metrics_visibility(ctx: RunContext) -> Finding | None:
    if ctx.metrics_available:
        return None
    return warn("H08")


@rule("H01", tags={"reliability", "throughput"}, source="25#6")
def steady_state(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    if not pts:
        return None
    short = [
        {"concurrency": p.concurrency, "share": p.steady_share}
        for p in pts
        if not p.steady_reliable
    ]
    if short:
        return warn("H01", points=short, checked=len(pts))
    return ok("H01", checked=len(pts))


@rule("H02", tags={"reliability", "network"}, source="25#6")
def weak_client(ctx: RunContext) -> Finding | None:
    pts = [p for p in ctx.points_in("concurrency") if p.concurrency]
    if not pts:
        return None
    short = []
    for p in pts:
        achieved = p.m("effective_concurrency", "avg")
        target = p.concurrency or 0
        if achieved is not None and achieved < CLIENT_CONCURRENCY_MIN * target:
            short.append({"concurrency": target, "achieved": achieved})
    if short:
        return warn("H02", points=short)
    return ok("H02", checked=len(pts))


@rule("H06", tags={"reliability", "ttft", "prefill"}, source="25#6")
def prefix_caching(ctx: RunContext) -> Finding | None:
    if not ctx.theory.prefix_caching:
        return None
    hits = queries = 0.0
    for p in ctx.points:
        hits += p.server_stat("prefix_cache_hits", "total") or 0
        queries += p.server_stat("prefix_cache_queries", "total") or 0
    return info("H06", hit_rate=hits / queries if queries else None)


@rule("H09", tags={"reliability", "network", "capacity"}, source="25#6")
def gateway(ctx: RunContext) -> Finding | None:
    pts = sweep(ctx)
    if not pts:
        return None
    codes: dict[int, int] = {}
    for p in pts:
        for code, n in p.error_codes.items():
            if code in GATEWAY_CODES:
                codes[code] = codes.get(code, 0) + n
    held = [
        {"concurrency": p.concurrency, "server_running": p.running_max}
        for p in pts
        if p.running_max is not None
        and (p.waiting_max or 0) == 0
        and p.running_max
        < SERVER_CONCURRENCY_MIN * min(p.concurrency, ctx.theory.default_max_num_seqs)
    ]
    timeout_s = _timeout_cluster(ctx)
    if codes or held or timeout_s:
        return warn(
            "H09",
            codes=", ".join(f"{k} x{v}" for k, v in sorted(codes.items())),
            held=held,
            timeout_s=timeout_s,
            stopped=any(p.stop_reason == "gateway_errors" for p in ctx.points),
        )
    return ok("H09", checked=len(pts))


def _timeout_cluster(ctx: RunContext) -> float | None:
    failed = [r.e2e_ms for p in ctx.points for r in p.requests if not r.ok and r.e2e_ms]
    if len(failed) < 3:
        return None
    median = statistics.median(failed)
    spread = (max(failed) - min(failed)) / median if median else 1.0
    if median / 1000 >= TIMEOUT_CLUSTER_MIN_S and spread <= TIMEOUT_CLUSTER_SPREAD:
        return median / 1000
    return None

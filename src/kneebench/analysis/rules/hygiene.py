"""H rules: is the test itself trustworthy? (notes 24 to 27, plus client distance)."""

from __future__ import annotations

import statistics

from kneebench.analysis import settings
from kneebench.analysis.context import UNRELIABLE_NETWORK_SHARE, RunContext
from kneebench.analysis.engine import Finding, bad, info, ok, rule, skip, warn
from kneebench.analysis.rules.concurrency import sweep
from kneebench.analysis.sweep import GATEWAY_CODES
from kneebench.phases.plan import CANCEL_PHASE
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
SHORT_OUTPUT_MAX_SHARE = 0.05
# One short answer in a 16-request point is 6%, but not a shortfall that moves the numbers.
SHORT_OUTPUT_MIN_COUNT = 2
OSL_WARNING = "Output Sequence Length Mismatch"
TOKEN_WARNING = "Token Count Discrepancy"


# Token counts that differ by the same few tokens everywhere are the chat template, not a
# tokenizer mismatch (kneebench estimate); a real mismatch is off by a share of the length.
TEMPLATE_OFFSET_SPREAD = 8
TOKEN_RATIO_WARN = 0.05


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
    # Answers kneebench abandons on purpose are short by design.
    checked = [p for p in ctx.points if p.status == "done" and p.phase != CANCEL_PHASE]
    if not checked:
        return None
    short = []
    for p in checked:
        # With ignore_eos the server generates exactly what was asked. AIPerf counts by
        # re-tokenizing the streamed text, which drifts on long answers; prefer the server's.
        mean_osl = p.m("usage_completion_tokens", "avg") or p.m("output_sequence_length", "avg")
        mismatches = p.m("osl_mismatch_count", "avg") or 0
        requests = p.m("request_count", "avg") or 0
        schedule = p.spec.get("schedule") or ()
        # A scheduled point mixes lengths (long prompts with short answers among the streams).
        expected = sum(r[2] for r in schedule) / len(schedule) if schedule else p.osl
        ratio = (mean_osl / expected) if mean_osl else None
        if p.m("usage_completion_tokens", "avg"):
            # AIPerf's per-request mismatches compare its recount with each request's sampled
            # length; with the server's own count, only a shortfall in the mean is real.
            mismatches = 0.0
        share = mismatches / requests if requests else None
        # AIPerf warns on a single short answer; only a real shortfall changes the numbers.
        warned_without_counts = share is None and p.has_warning(OSL_WARNING)
        if (
            warned_without_counts
            or (
                share is not None
                and share > SHORT_OUTPUT_MAX_SHARE
                and mismatches >= SHORT_OUTPUT_MIN_COUNT
            )
            or (ratio is not None and ratio < OUTPUT_LENGTH_MIN_RATIO)
        ):
            short.append(
                {
                    "phase": p.phase,
                    "requested": p.osl,
                    "mean": mean_osl,
                    "ratio": ratio,
                    "mismatches": mismatches,
                    "share": share,
                }
            )
    if short:
        return bad("H04", points=short, checked=len(checked))
    return ok("H04", checked=len(checked))


@rule("H05", tags={"reliability", "tokenizer"}, source="26#5")
def token_count_discrepancy(ctx: RunContext) -> Finding | None:
    flagged = list(dict.fromkeys(p.phase for p in ctx.points if p.has_warning(TOKEN_WARNING)))
    pairs = [
        (client, server)
        for p in ctx.points
        if not (p.spec.get("turns") or 0) > 1
        and (client := p.m("input_sequence_length", "avg"))
        and (server := p.m("usage_prompt_tokens", "avg"))
    ]
    if len(pairs) >= 2:
        offsets = [server - client for client, server in pairs]
        # The server wraps every prompt in the chat template: a fixed number of extra tokens,
        # large next to a short prompt. A wrong tokenizer is off by a share of the length.
        if max(offsets) - min(offsets) <= TEMPLATE_OFFSET_SPREAD:
            return ok("H05", overhead=round(sum(offsets) / len(offsets)), checked=len(pairs))
        ratios = [server / client for client, server in pairs if client > 0]
        off = max(abs(r - 1) for r in ratios)
        if off > TOKEN_RATIO_WARN:
            return warn("H05", phases=flagged or ["all points"], off=off, tokenizer=ctx.hf_model_id)
        return ok("H05", overhead=None, checked=len(pairs))
    if not flagged:
        return None
    return warn("H05", phases=flagged, off=None, tokenizer=ctx.hf_model_id)


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
    seqs = settings.max_num_seqs(ctx).value
    held = [
        {"concurrency": p.concurrency, "server_running": p.running_max}
        for p in pts
        if p.running_max is not None
        and (p.waiting_max or 0) == 0
        and p.running_max < SERVER_CONCURRENCY_MIN * min(p.concurrency, seqs)
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
    failed = [
        r.e2e_ms
        for p in ctx.points
        if p.phase != CANCEL_PHASE
        for r in p.requests
        if not r.ok and r.e2e_ms
    ]
    if len(failed) < 3:
        return None
    median = statistics.median(failed)
    spread = (max(failed) - min(failed)) / median if median else 1.0
    if median / 1000 >= TIMEOUT_CLUSTER_MIN_S and spread <= TIMEOUT_CLUSTER_SPREAD:
        return median / 1000
    return None


@rule("H10", tags={"reliability"}, source="24#5")
def other_traffic(ctx: RunContext) -> Finding | None:
    """Someone else's requests on the server blur every number (note 24 section 5).

    Two signs: requests already running when kneebench detected the server, and more than
    one request running while kneebench sent one at a time.
    """
    start = ctx.activity_at_start
    single = [
        {"phase": p.phase, "running": running}
        for p in ctx.points
        if p.status == "done"
        and p.concurrency == 1
        and (running := p.server_stat("running", "max")) is not None
        and running > 1
    ]
    if start is None and not any(p.server_stat("running", "max") is not None for p in ctx.points):
        return None
    busy = start is not None and (start["running"] or start["waiting"]) > 0
    data = {"start": start, "busy_at_start": busy, "single": single}
    if busy or single:
        return warn("H10", **data)
    return ok("H10", **data)

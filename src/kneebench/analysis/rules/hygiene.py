"""H rules: is the test itself trustworthy? (notes 24 to 27, plus client distance)."""

from __future__ import annotations

from kneebench.analysis.context import UNRELIABLE_NETWORK_SHARE, RunContext
from kneebench.analysis.engine import Finding, bad, ok, rule, skip, warn
from kneebench.servers.vllm import version_status

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

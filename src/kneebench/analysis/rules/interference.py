"""I rules: do long prompts freeze other users' streaming answers (notes 44 and 45)?"""

from __future__ import annotations

from typing import Any

from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, info, ok, rule, warn
from kneebench.analysis.rules.scheduler import _prefill_ms_per_token

# A spike counts as caused by a long prompt when it starts within this long of the arrival.
ALIGN_S = 1.5
# Streams that already pause this much without long prompts (P99 over the median) are too
# noisy, usually from the network or a gateway, to pin pauses on the long prompts.
NOISY_P99_FACTOR = 5.0
# Note 45 section 5: chunks below about 500 tokens waste the GPU; tiles come in 128s.
MIN_CHUNK = 512
CHUNK_STEP = 128


def _streams(ctx: RunContext, phase: str) -> dict[str, Any] | None:
    pts = [p for p in ctx.points_in(phase) if p.series.get("streams")]
    return pts[0].series["streams"] if pts else None


def _aligned(mixed: dict[str, Any], alone: dict[str, Any]) -> int:
    """How many long prompts were followed by a pause longer than the streams ever had alone."""
    threshold = max(mixed["stall_ms"], alone["p999_ms"])
    spikes = [at for at, gap in mixed["series"] if gap > threshold]
    return sum(
        1
        for inj in mixed["injected"]
        if any(inj["at_s"] <= at <= inj["at_s"] + ALIGN_S for at in spikes)
    )


@rule("I01", tags={"tpot", "prefill", "shape", "latency"}, source="44#4")
def long_prompt_interference(ctx: RunContext) -> Finding | None:
    alone = _streams(ctx, "interference_base")
    mixed = _streams(ctx, "interference")
    if alone is None or mixed is None or not mixed["injected"]:
        return None
    target = ctx.workload.slo.tpot_ms
    budget = ctx.theory.default_max_num_batched_tokens
    per_token = _prefill_ms_per_token(ctx)
    # Note 45 section 5: a step carrying a chunk must still finish within the per-token target.
    safe = None
    if target and per_token:
        safe = max(MIN_CHUNK, int(target / per_token) // CHUNK_STEP * CHUNK_STEP)
    ttfts = [i["ttft_ms"] for i in mixed["injected"] if i["ok"] and i["ttft_ms"]]
    aligned = _aligned(mixed, alone)
    noisy = alone["p99_ms"] > NOISY_P99_FACTOR * alone["p50_ms"]
    data = {
        "noisy": noisy,
        "streams": mixed["streams"],
        "injections": len(mixed["injected"]),
        "inject_isl": mixed["injected"][0]["isl"],
        "aligned": aligned,
        "alone": alone,
        "mixed": mixed,
        "rows": [
            {"run": "Streams alone", **_row(alone)},
            {"run": "With long prompts", **_row(mixed)},
        ],
        "amplification": mixed["max_ms"] / mixed["p50_ms"] if mixed["p50_ms"] else None,
        "inject_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
        "tpot_target_ms": target,
        "budget": budget,
        "safe_budget": safe,
        "can_lower": safe is not None and safe < budget,
    }
    if noisy:
        return info("I01", **data)
    caused = aligned * 2 >= len(mixed["injected"])
    if target and mixed["max_ms"] > target and caused:
        return warn("I01", **data)
    if (target and mixed["max_ms"] <= target) or not caused:
        return ok("I01", **data)
    return info("I01", **data)


def _row(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in ("p50_ms", "p99_ms", "p999_ms", "max_ms", "stalls")}

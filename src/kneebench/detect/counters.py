"""Read the server's request and token counters, before and after a probe."""

from __future__ import annotations

from kneebench.detect.endpoint import probe
from kneebench.detect.metrics import find, parse_samples
from kneebench.loadgen.base import Endpoint

GENERATED = "vllm:generation_tokens_total"
FINISHED = "vllm:request_success_total"


def read_counters(endpoint: Endpoint) -> dict[str, float] | None:
    """Generated tokens and finished requests by reason, or None without /metrics."""
    result = probe(endpoint.base_url, "/metrics", endpoint.api_key, expect_json=False)
    if not result.ok or not result.text:
        return None
    try:
        samples = parse_samples(result.text)
    except ValueError:
        return None
    generated = find(samples, GENERATED)
    if not generated:
        return None
    counters = {"generated": sum(s.value for s in generated)}
    for s in find(samples, FINISHED):
        reason = s.labels.get("finished_reason", "other")
        counters[reason] = counters.get(reason, 0.0) + s.value
    return counters

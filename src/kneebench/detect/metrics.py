"""Parse a Prometheus /metrics snapshot from a vLLM server."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

CACHE_CONFIG_METRIC = "vllm:cache_config_info"


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class CacheConfig:
    """What vLLM reports about its KV cache: replaces guessing for the memory wall."""

    gpu_memory_utilization: float | None
    kv_cache_size_tokens: int | None
    kv_cache_max_concurrency: float | None
    block_size: int | None
    enable_prefix_caching: bool | None
    cache_dtype: str
    num_gpu_blocks: int | None


def parse_samples(text: str) -> list[Sample]:
    samples: list[Sample] = []
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            samples.append(Sample(name=s.name, labels=dict(s.labels), value=float(s.value)))
    return samples


def find(samples: list[Sample], *names: str) -> list[Sample]:
    """Samples for the first name present (metric names differ across vLLM versions)."""
    for name in names:
        hits = [s for s in samples if s.name == name]
        if hits:
            return hits
    return []


def activity(text: str | None) -> dict[str, float] | None:
    """Requests running and waiting in the server, summed over engines, from one snapshot."""
    if not text:
        return None
    try:
        samples = parse_samples(text)
    except ValueError:
        return None
    running = find(samples, "vllm:num_requests_running")
    if not running:
        return None
    return {
        "running": sum(s.value for s in running),
        "waiting": sum(s.value for s in find(samples, "vllm:num_requests_waiting")),
    }


def cache_config(samples: list[Sample]) -> CacheConfig | None:
    hits = find(samples, CACHE_CONFIG_METRIC)
    if not hits:
        return None
    labels = hits[0].labels
    block = _int(labels.get("block_size"))
    blocks = _int(labels.get("num_gpu_blocks"))
    tokens = _int(labels.get("kv_cache_size_tokens"))
    if tokens is None and block and blocks:
        tokens = block * blocks
    return CacheConfig(
        gpu_memory_utilization=_float(labels.get("gpu_memory_utilization")),
        kv_cache_size_tokens=tokens,
        kv_cache_max_concurrency=_float(labels.get("kv_cache_max_concurrency")),
        block_size=block,
        enable_prefix_caching=_bool(labels.get("enable_prefix_caching")),
        cache_dtype=labels.get("cache_dtype") or "auto",
        num_gpu_blocks=blocks,
    )


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "", "None") else None
    except ValueError:
        return None


def _int(value: str | None) -> int | None:
    f = _float(value)
    return int(f) if f is not None else None


def _bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() == "true"

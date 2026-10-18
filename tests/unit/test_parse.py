"""Parse recorded AIPerf artifacts and /metrics snapshots."""

from kneebench.detect.metrics import cache_config, find, parse_samples
from kneebench.loadgen.aiperf.parse import parse_point, parse_warnings
from tests.builders import BASELINE_FIXTURE, ENDPOINT_FIXTURE


def test_parse_recorded_baseline() -> None:
    p = parse_point(BASELINE_FIXTURE)
    assert p.aiperf_version == "0.13.0"
    assert p.stat("request_count") == 30
    assert 5 < (p.stat("inter_token_latency", "p50") or 0) < 7
    assert p.stat("osl_mismatch_count") == 0
    assert len(p.requests) == 40
    assert len(p.profiling_requests()) == 30
    assert all(r.ok and r.ttft_ms and r.tpot_ms for r in p.profiling_requests())
    assert p.timeslices and "inter_token_latency" in p.timeslices[0].metrics
    assert p.profiling_window_ns is not None
    assert not p.warnings  # the GPU telemetry panel is informational and dropped


def test_server_metrics_are_profiling_window_deltas() -> None:
    p = parse_point(BASELINE_FIXTURE)
    preempt = p.server_series("vllm:num_preemptions", "vllm:num_preemptions_total")
    assert preempt and preempt[0].kind == "counter"
    assert preempt[0].stats["total"] == 0
    kv = p.server_series("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    assert kv and 0 <= kv[0].stats["max"] < 0.01
    hits = p.server_series("vllm:prefix_cache_hits")[0].stats["total"]
    queries = p.server_series("vllm:prefix_cache_queries")[0].stats["total"]
    assert hits / queries < 0.1  # cache busting keeps prefix hits to the chat template


def test_parse_warning_panels() -> None:
    console = (
        "╭──── Output Sequence Length Mismatch Warning ────╮\n"
        "│  16 of 16 requests (100.0%) differ.            │\n"
        "╰─────────────────────────────────────────────────╯\n"
        "╭──── GPU Telemetry Platform ────╮\n"
        "│  Platform: unknown             │\n"
        "╰────────────────────────────────╯\n"
    )
    warnings = parse_warnings(console)
    assert [w.title for w in warnings] == ["Output Sequence Length Mismatch Warning"]
    assert "16 of 16" in warnings[0].body


def test_cache_config_from_metrics_snapshot() -> None:
    samples = parse_samples((ENDPOINT_FIXTURE / "metrics.prom").read_text())
    c = cache_config(samples)
    assert c is not None
    assert c.kv_cache_size_tokens == 532_768
    assert c.gpu_memory_utilization == 0.9
    assert c.block_size == 16
    assert c.enable_prefix_caching is True
    assert c.cache_dtype == "auto"
    assert c.kv_cache_max_concurrency is not None and 65 < c.kv_cache_max_concurrency < 66
    assert find(samples, "vllm:no_such_metric", "vllm:num_requests_waiting")


def test_cache_config_derives_tokens_from_blocks_on_older_servers() -> None:
    text = 'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000",cache_dtype="fp8"} 1.0\n'
    c = cache_config(parse_samples(text))
    assert c is not None
    assert c.kv_cache_size_tokens == 16_000
    assert c.cache_dtype == "fp8"
    assert cache_config(parse_samples("other_metric 1.0\n")) is None

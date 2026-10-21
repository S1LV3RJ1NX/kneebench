"""Translate a PointSpec into a complete AIPerf YAML config (schema 2.0).

kneebench writes the whole config instead of passing flags: AIPerf 0.13 does not build
datasets or phases from CLI flags once a config file is given, and a file per point is
a reproducible artifact. The API key never appears in the file or in argv: the file
holds a ``${KNEEBENCH_AIPERF_API_KEY}`` placeholder that AIPerf resolves from the child
process environment, and AIPerf redacts it in its own JSON exports.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from kneebench.loadgen.base import Arrival, Endpoint, LoadMode, PointSpec
from kneebench.loadgen.texts import rag_prompts

API_KEY_ENV = "KNEEBENCH_AIPERF_API_KEY"
CONFIG_FILENAME = "aiperf_config.yaml"

# Grace period after a timed phase so in-flight requests finish and are counted.
_RATE_GRACE_S = 60.0
# Upper bound on synthetic prompts for timed phases; enough to avoid prompt reuse.
_MAX_ENTRIES = 20_000


def build_config(endpoint: Endpoint, spec: PointSpec, artifact_dir: Path) -> dict[str, Any]:
    endpoint_cfg: dict[str, Any] = {
        "url": endpoint.base_url,
        "type": endpoint.endpoint_type,
        "streaming": True,
        # ignore_eos forces the requested output length (note 26); greedy for repeatability.
        "extra": {"ignore_eos": True, "temperature": 0},
    }
    if endpoint.api_key:
        endpoint_cfg["apiKey"] = f"${{{API_KEY_ENV}}}"
    if endpoint.extra_headers:
        endpoint_cfg["headers"] = dict(endpoint.extra_headers)

    benchmark: dict[str, Any] = {
        "model": endpoint.model,
        "tokenizer": {"name": endpoint.tokenizer},
        "endpoint": endpoint_cfg,
        "dataset": _dataset(spec),
        "phases": _phases(spec),
        "networkLatency": {"enabled": True},
        "artifacts": _artifacts(spec, artifact_dir),
    }
    if spec.goodput:
        benchmark["slos"] = dict(spec.goodput)

    return {"schemaVersion": "2.0", "randomSeed": spec.random_seed, "benchmark": benchmark}


def build_env(endpoint: Endpoint, base_env: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in base_env.items() if k != API_KEY_ENV}
    if endpoint.api_key:
        env[API_KEY_ENV] = endpoint.api_key
    return env


def _dataset(spec: PointSpec) -> dict[str, Any]:
    if spec.schedule:
        return _schedule_dataset(spec)
    if spec.rag_order:
        return _rag_dataset(spec)
    return _synthetic_dataset(spec)


def _rag_dataset(spec: PointSpec) -> dict[str, Any]:
    count = (spec.request_count or 0) + (spec.warmup_count or 0)
    return {
        "type": "file",
        "format": "single_turn",
        "sampling": "sequential",
        "records": [
            {"text": text, "output_length": spec.osl}
            for text in rag_prompts(spec.rag_order or "sorted", max(count, 1), spec.random_seed)
        ],
    }


def _synthetic_dataset(spec: PointSpec) -> dict[str, Any]:
    prompts: dict[str, Any] = {
        "isl": {"mean": spec.isl, "stddev": spec.isl_stddev},
        "osl": {"mean": spec.osl, "stddev": spec.osl_stddev},
    }
    if spec.cache_bust:
        # Unique prefix per request so prefix caching can't flatter TTFT (note 25).
        prompts["cacheBust"] = {"target": "first_turn_prefix"}
    dataset: dict[str, Any] = {
        "type": "synthetic",
        "entries": _dataset_entries(spec),
        "prompts": prompts,
    }
    if spec.prefix_pool and spec.prefix_len:
        dataset["prefixPrompts"] = {"poolSize": spec.prefix_pool, "length": spec.prefix_len}
    elif spec.system_len:
        dataset["prefixPrompts"] = {"sharedSystemLength": spec.system_len}
    if spec.turns and spec.turns > 1:
        dataset["turns"] = spec.turns
        dataset["turnDelay"] = 0
    return dataset


def _schedule_dataset(spec: PointSpec) -> dict[str, Any]:
    """A trace replayed at fixed times. Each prompt is sampled fresh from AIPerf's corpus, so
    no two share a long beginning and prefix caching can't shorten the long ones. (Block ids
    would guarantee it, but AIPerf 0.13 decodes those in a process pool that crashes here.)"""
    return {
        "type": "file",
        "format": "mooncake_trace",
        "records": [
            {"timestamp": at_ms, "input_length": isl, "output_length": osl}
            for at_ms, isl, osl in spec.schedule
        ],
    }


def _phases(spec: PointSpec) -> list[dict[str, Any]]:
    if spec.mode is LoadMode.SCHEDULE:
        return [{"name": "profiling", "type": "fixed_schedule"}]
    phases: list[dict[str, Any]] = []
    if spec.warmup_count:
        phases.append(
            {
                "name": "warmup",
                "type": "concurrency",
                "excludeFromResults": True,
                "requests": spec.warmup_count,
                "concurrency": spec.concurrency or 1,
            }
        )

    profiling: dict[str, Any] = {"name": "profiling"}
    if spec.mode is LoadMode.CONCURRENCY:
        profiling.update(type="concurrency", concurrency=spec.concurrency)
        if spec.request_count:
            profiling["requests"] = spec.request_count
        if spec.duration_s:
            profiling["duration"] = spec.duration_s
        if spec.cancel_after_s is not None:
            profiling["cancellation"] = {"rate": 100.0, "delay": spec.cancel_after_s}
    else:
        profiling.update(type=spec.arrival.value, rate=spec.request_rate)
        if spec.arrival is Arrival.GAMMA and spec.burstiness is not None:
            profiling["smoothness"] = spec.burstiness
        if spec.concurrency:
            profiling["concurrency"] = spec.concurrency
        if spec.warmup_s:
            # Same arrivals, excluded from results; seamless so the server doesn't drain
            # to idle before profiling starts.
            warm = {
                k: v for k, v in profiling.items() if k not in ("name", "gracePeriod", "requests")
            }
            phases.append(
                {**warm, "name": "warmup", "excludeFromResults": True, "duration": spec.warmup_s}
            )
            profiling["seamless"] = True
        if spec.duration_s:
            profiling["duration"] = spec.duration_s
            profiling["gracePeriod"] = _RATE_GRACE_S
        if spec.request_count:
            profiling["requests"] = spec.request_count
    phases.append(profiling)
    return phases


def _artifacts(spec: PointSpec, artifact_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "dir": str(artifact_dir),
        "summary": ["json"],
        "records": ["jsonl"],
    }
    if spec.slice_duration_s:
        artifacts["sliceDuration"] = spec.slice_duration_s
    return artifacts


def _dataset_entries(spec: PointSpec) -> int:
    planned = (spec.request_count or 0) + (spec.warmup_count or 0)
    if spec.mode is LoadMode.RATE and spec.request_rate and spec.duration_s:
        seconds = spec.duration_s + (spec.warmup_s or 0)
        planned = max(planned, math.ceil(spec.request_rate * seconds * 1.2))
    return min(max(planned, 100), _MAX_ENTRIES)

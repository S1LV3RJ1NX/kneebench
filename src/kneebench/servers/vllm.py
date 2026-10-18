"""vLLM adapter: tested versions, flag defaults and metric names per version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packaging.version import InvalidVersion, Version

from kneebench import content
from kneebench.physics.gpus import GpuSpec


@dataclass(frozen=True)
class BatchDefaults:
    max_num_seqs: int
    max_num_batched_tokens: int
    verified: bool  # False when the version is outside every verified range


@dataclass(frozen=True)
class VersionStatus:
    version: str | None
    in_tested_range: bool
    tested_min: str
    tested_max_exclusive: str


def _data() -> dict[str, Any]:
    data: dict[str, Any] = content.load("vllm_versions.yaml")
    return data


def _parse(version: str | None) -> Version | None:
    if not version:
        return None
    try:
        return Version(version)
    except InvalidVersion:
        return None


def version_status(version: str | None) -> VersionStatus:
    tested = _data()["tested_range"]
    v = _parse(version)
    in_range = v is not None and Version(tested["min"]) <= v < Version(tested["max_exclusive"])
    return VersionStatus(version, in_range, tested["min"], tested["max_exclusive"])


def _range_for(version: str | None) -> tuple[dict[str, Any], bool]:
    ranges: list[dict[str, Any]] = _data()["ranges"]
    v = _parse(version)
    if v is not None:
        for r in ranges:
            if Version(r["min"]) <= v < Version(r["max_exclusive"]):
                return r, True
    # Unknown or untested version: use the newest verified range and say so.
    newest = max(ranges, key=lambda r: Version(r["min"]))
    return newest, False


def batch_defaults(version: str | None, gpu: GpuSpec) -> BatchDefaults:
    rng, verified = _range_for(version)
    name = gpu.name.lower()
    for tier in rng["batch_defaults"]:
        if gpu.memory_gib < tier["min_gib"]:
            continue
        if any(token in name for token in tier.get("exclude_name_contains", [])):
            continue
        return BatchDefaults(
            max_num_seqs=int(tier["max_num_seqs"]),
            max_num_batched_tokens=int(tier["max_num_batched_tokens"]),
            verified=verified,
        )
    raise ValueError("vllm_versions.yaml has no catch-all batch_defaults tier")


def metric_names(signal: str) -> tuple[str, ...]:
    names: list[str] = _data()["metric_names"][signal]
    return tuple(names)

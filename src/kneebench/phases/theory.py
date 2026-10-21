"""Phase 0: theory before any traffic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from kneebench.detect.metrics import CacheConfig
from kneebench.domain import BASELINE_SHAPE, Shape, fit_shape
from kneebench.physics.floors import (
    Hardware,
    exchange_rate,
    floors,
    memory_budget,
    requests_that_fit,
    startup_fit,
)
from kneebench.physics.gpus import GIB
from kneebench.physics.model import ModelSpec
from kneebench.servers.vllm import batch_defaults


@dataclass(frozen=True)
class ShapeTheory:
    name: str
    isl: int
    osl: int
    adapted: bool
    original_isl: int
    original_osl: int
    tpot_floor_ms: float
    ttft_floor_ms: float
    e2e_floor_ms: float
    fits: float | None


@dataclass(frozen=True)
class Theory:
    model_hf_id: str | None
    model_params: int
    model_weight_bytes: int
    model_weight_dtype: str
    model_quantization: str | None
    model_support: str
    model_notes: tuple[str, ...]
    kv_bytes_per_token: int
    gpu_id: str
    gpu_name: str
    gpu_count: int
    gpu_memory_gib: float
    gpu_bandwidth_tbps: float
    gpu_peak_tflops: float
    max_model_len: int | None
    pool_tokens: int | None
    kv_pool_gib_per_gpu: float | None
    gpu_memory_utilization: float | None
    cache_dtype: str
    prefix_caching: bool | None
    startup_fit: float | None
    implied_min_gpu_gib: float | None
    exchange_rate: float
    vllm_version: str | None
    default_max_num_seqs: int
    default_max_num_batched_tokens: int
    defaults_verified: bool
    shapes: tuple[ShapeTheory, ...] = field(default=())
    block_size: int | None = None  # KV cache block, from vllm:cache_config_info

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def shape(self, name: str) -> ShapeTheory | None:
        return next((s for s in self.shapes if s.name == name), None)


def compute_theory(
    model: ModelSpec,
    hw: Hardware,
    cache: CacheConfig | None,
    max_model_len: int | None,
    shapes: tuple[Shape, ...],
    vllm_version: str | None,
) -> Theory:
    pool = cache.kv_cache_size_tokens if cache else None
    util = cache.gpu_memory_utilization if cache else None
    cache_dtype = cache.cache_dtype if cache else "auto"
    block = cache.block_size if cache else None

    shape_rows = []
    for shape in (BASELINE_SHAPE, *shapes):
        fitted = fit_shape(shape, max_model_len)
        s = fitted.shape
        f = floors(model, hw, s.isl, s.osl)
        shape_rows.append(
            ShapeTheory(
                name=s.name,
                isl=s.isl,
                osl=s.osl,
                adapted=fitted.adapted,
                original_isl=fitted.original.isl,
                original_osl=fitted.original.osl,
                tpot_floor_ms=f.tpot_s * 1e3,
                ttft_floor_ms=f.ttft_s * 1e3,
                e2e_floor_ms=f.e2e_s * 1e3,
                fits=requests_that_fit(pool, s.isl, s.osl, block) if pool else None,
            )
        )

    implied = kv_pool_gib = None
    if pool and util:
        budget = memory_budget(model, hw, pool, util, cache_dtype)
        implied = budget.implied_min_gpu_bytes / GIB
        kv_pool_gib = budget.kv_pool_per_gpu_bytes / GIB

    startup = None
    if cache and cache.kv_cache_max_concurrency is not None:
        startup = cache.kv_cache_max_concurrency
    elif pool and max_model_len:
        startup = startup_fit(pool, max_model_len)

    defaults = batch_defaults(vllm_version, hw.gpu)
    return Theory(
        model_hf_id=model.hf_id,
        model_params=model.params,
        model_weight_bytes=model.weight_bytes,
        model_weight_dtype=model.weight_dtype,
        model_quantization=model.quantization,
        model_support=model.support.value,
        model_notes=model.notes,
        kv_bytes_per_token=model.kv_bytes_per_token(cache_dtype),
        gpu_id=hw.gpu.id,
        gpu_name=hw.gpu.name,
        gpu_count=hw.count,
        gpu_memory_gib=hw.gpu.memory_gib,
        gpu_bandwidth_tbps=hw.gpu.bandwidth_tbps,
        gpu_peak_tflops=hw.gpu.peak_flops(model.fp8_compute) / 1e12,
        max_model_len=max_model_len,
        pool_tokens=pool,
        kv_pool_gib_per_gpu=kv_pool_gib,
        gpu_memory_utilization=util,
        cache_dtype=cache_dtype,
        prefix_caching=cache.enable_prefix_caching if cache else None,
        startup_fit=startup,
        implied_min_gpu_gib=implied,
        exchange_rate=exchange_rate(model, hw),
        vllm_version=vllm_version,
        default_max_num_seqs=defaults.max_num_seqs,
        default_max_num_batched_tokens=defaults.max_num_batched_tokens,
        defaults_verified=defaults.verified,
        shapes=tuple(shape_rows),
        block_size=block,
    )


def theory_from_snapshot(
    snapshot: dict[str, Any], workload: dict[str, Any], hw: Hardware
) -> Theory:
    """Recompute theory from what a run stored, e.g. after correcting the GPU entry."""
    import json

    from kneebench.physics.model import model_spec_from_hf

    if not snapshot.get("hf_config_json"):
        raise ValueError("this run has no stored model config; rerun it to change the GPU")
    params = json.loads(snapshot["safetensors_json"]) if snapshot.get("safetensors_json") else None
    model = model_spec_from_hf(
        json.loads(snapshot["hf_config_json"]), params, hf_id=snapshot.get("hf_model_id")
    )
    cache = (
        CacheConfig(**json.loads(snapshot["cache_config_json"]))
        if snapshot.get("cache_config_json")
        else None
    )
    shapes = tuple(Shape(**s) for s in json.loads(workload["shapes_json"]))
    return compute_theory(
        model, hw, cache, snapshot.get("max_model_len"), shapes, snapshot.get("vllm_version")
    )


def theory_from_json(data: dict[str, Any]) -> Theory:
    shapes = tuple(ShapeTheory(**s) for s in data.get("shapes", ()))
    fields = {k: v for k, v in data.items() if k != "shapes"}
    fields["model_notes"] = tuple(fields.get("model_notes") or ())
    return Theory(**fields, shapes=shapes)

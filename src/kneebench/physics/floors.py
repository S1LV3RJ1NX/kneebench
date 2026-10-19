"""Theory before traffic (phase 0): the floors and memory math from notes 24, 26 and 27."""

from __future__ import annotations

from dataclasses import dataclass

from kneebench.physics.gpus import GIB, GpuSpec
from kneebench.physics.model import ModelSpec

# Note 26 section 1: prefill is assumed to run at ~50% of peak for the exchange rate.
PREFILL_EFFICIENCY_FOR_EXCHANGE = 0.5
# Activations, CUDA graphs and allocator slack that sit inside vLLM's memory budget.
MIN_RUNTIME_OVERHEAD_BYTES = 1 * GIB


@dataclass(frozen=True)
class Hardware:
    gpu: GpuSpec
    count: int = 1  # tensor-parallel size

    @property
    def bandwidth_bytes_s(self) -> float:
        return self.gpu.bandwidth_bytes_s * self.count

    def peak_flops(self, fp8: bool) -> float:
        return self.gpu.peak_flops(fp8) * self.count


@dataclass(frozen=True)
class Floors:
    """Fastest possible single-request timings (note 24 section 2)."""

    isl: int
    osl: int
    tpot_s: float
    ttft_s: float

    @property
    def e2e_s(self) -> float:
        # The first token's time is the TTFT, so only osl - 1 decode steps follow.
        return self.ttft_s + (self.osl - 1) * self.tpot_s


def flops_per_token(model: ModelSpec) -> float:
    return 2.0 * model.params


def floors(model: ModelSpec, hw: Hardware, isl: int, osl: int) -> Floors:
    tpot = model.weight_bytes / hw.bandwidth_bytes_s
    ttft = flops_per_token(model) * isl / hw.peak_flops(model.fp8_compute)
    return Floors(isl=isl, osl=osl, tpot_s=tpot, ttft_s=ttft)


def mbu(model: ModelSpec, hw: Hardware, measured_tpot_s: float) -> float:
    """Model bandwidth utilization: weights read per step over time, against peak."""
    return model.weight_bytes / measured_tpot_s / hw.bandwidth_bytes_s


def prefill_mfu(model: ModelSpec, hw: Hardware, isl: int, measured_ttft_s: float) -> float:
    return flops_per_token(model) * isl / measured_ttft_s / hw.peak_flops(model.fp8_compute)


def exchange_rate(model: ModelSpec, hw: Hardware) -> float:
    """How many prompt tokens cost the same time as one output token at batch 1."""
    t_out = model.weight_bytes / hw.bandwidth_bytes_s
    t_in = flops_per_token(model) / (
        PREFILL_EFFICIENCY_FOR_EXCHANGE * hw.peak_flops(model.fp8_compute)
    )
    return t_out / t_in


def requests_that_fit(pool_tokens: int, isl: int, osl: int) -> float:
    """Sized by final length, not starting length (note 26 section 4)."""
    return pool_tokens / (isl + osl)


def startup_fit(pool_tokens: int, max_model_len: int) -> float:
    """How many max-length requests the cache holds; below 1 the engine refuses to start."""
    return pool_tokens / max_model_len


@dataclass(frozen=True)
class MemoryBudget:
    weights_per_gpu_bytes: float
    kv_pool_per_gpu_bytes: float
    gpu_memory_utilization: float

    @property
    def implied_min_gpu_bytes(self) -> float:
        """Smallest per-GPU memory that could hold this budget at this utilization."""
        used = self.weights_per_gpu_bytes + self.kv_pool_per_gpu_bytes
        return (used + MIN_RUNTIME_OVERHEAD_BYTES) / self.gpu_memory_utilization


def memory_budget(
    model: ModelSpec,
    hw: Hardware,
    pool_tokens: int,
    gpu_memory_utilization: float,
    cache_dtype: str = "auto",
) -> MemoryBudget:
    kv_per_gpu = pool_tokens * model.kv_bytes_per_token_per_gpu(hw.count, cache_dtype)
    return MemoryBudget(
        weights_per_gpu_bytes=model.weight_bytes / hw.count,
        kv_pool_per_gpu_bytes=kv_per_gpu,
        gpu_memory_utilization=gpu_memory_utilization,
    )

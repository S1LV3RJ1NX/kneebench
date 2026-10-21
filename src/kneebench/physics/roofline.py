"""Roofline (note 61) and precision what-ifs (notes 50, 51, 53) from theory alone.

A kernel takes about as long as the larger of its math and its bytes. A decode step reads
the weights once for the whole batch plus every request's cache, so its intensity is about
the batch size until the cache reads catch up with the weights.
"""

from __future__ import annotations

from dataclasses import dataclass

# Note 50 section 2: 4-bit weights with one 16-bit scale per 128 weights.
FOUR_BIT_BITS = 4 + 16 / 128
# INT8 tensor cores run at twice the BF16 rate on every GPU in the catalog that has them
# (kneebench estimate; the catalog lists FP8 but not INT8).
INT8_SPEEDUP = 2.0


@dataclass(frozen=True)
class Chip:
    bandwidth_bytes_s: float
    bf16_flops: float
    fp8_flops: float | None

    def peak(self, compute: str) -> float:
        if compute == "fp8" and self.fp8_flops:
            return self.fp8_flops
        if compute == "int8":
            return self.bf16_flops * INT8_SPEEDUP
        return self.bf16_flops

    def ridge(self, compute: str) -> float:
        """FLOPs per byte where the slope meets the roof (note 61 section 3)."""
        return self.peak(compute) / self.bandwidth_bytes_s


@dataclass(frozen=True)
class Model:
    params: int
    weight_bytes: float
    kv_bytes_per_token: float
    attn_width: int | None  # layers x attention heads x head dim

    def decode_flops(self, batch: float, context: float) -> float:
        # Weights: 2 FLOPs per parameter per token. Attention: scores and values over the
        # context, 2 x 2 FLOPs per head dimension per cached token per layer.
        attention = 4 * self.attn_width * context if self.attn_width else 0.0
        return batch * (2 * self.params + attention)

    def decode_bytes(self, batch: float, context: float) -> float:
        return self.weight_bytes + batch * context * self.kv_bytes_per_token

    def intensity(self, batch: float, context: float) -> float:
        return self.decode_flops(batch, context) / self.decode_bytes(batch, context)

    def intensity_limit(self, context: float) -> float:
        """Intensity as the batch grows without bound: the cache reads dominate the weights."""
        return self.decode_flops(1, context) / (context * self.kv_bytes_per_token)


def step_s(model: Model, chip: Chip, compute: str, batch: float, context: float) -> float:
    """Idealized decode step: the larger of math time and memory time (note 61 section 1)."""
    math_s = model.decode_flops(batch, context) / chip.peak(compute)
    memory_s = model.decode_bytes(batch, context) / chip.bandwidth_bytes_s
    return max(math_s, memory_s)


def ridge_batch(model: Model, chip: Chip, compute: str, context: float) -> float | None:
    """Batch at which a decode step reaches the ridge, or None if the cache reads never let it."""
    ridge = chip.ridge(compute)
    per_request = model.decode_flops(1, context) - ridge * context * model.kv_bytes_per_token
    if per_request <= 0:
        return None
    return ridge * model.weight_bytes / per_request


@dataclass(frozen=True)
class Precision:
    id: str
    weight_bits: float | None  # None keeps the served weights
    compute: str | None  # None keeps the served compute path
    kv_factor: float = 1.0
    needs_fp8: bool = False


# Notes 50 section 5 and 51 section 3: what each format saves, and where it runs.
WHAT_IFS = (
    Precision("fp8_kv", None, None, kv_factor=0.5),
    Precision("fp8", 8, "fp8", needs_fp8=True),
    Precision("int8", 8, "int8"),
    Precision("w4a16", FOUR_BIT_BITS, "bf16"),
)


@dataclass(frozen=True)
class Projection:
    id: str
    weight_bytes: float
    kv_bytes_per_token: float
    pool_tokens: float | None
    compute: str
    tpot_ratio: float  # idealized step time against the served setup, at this batch
    tpot_ratio_one: float  # the same at one request
    ttft_ratio: float


def project(
    served: Model,
    chip: Chip,
    served_compute: str,
    pool_tokens: int | None,
    batch: float,
    context: float,
    what: Precision,
) -> Projection:
    weights = (
        served.weight_bytes if what.weight_bits is None else served.params * what.weight_bits / 8
    )
    compute = what.compute or served_compute
    model = Model(
        served.params, weights, served.kv_bytes_per_token * what.kv_factor, served.attn_width
    )
    pool = None
    if pool_tokens:
        # Freed weight memory goes to the cache (note 42 section 4).
        pool_bytes = pool_tokens * served.kv_bytes_per_token + served.weight_bytes - weights
        pool = pool_bytes / model.kv_bytes_per_token
    return Projection(
        id=what.id,
        weight_bytes=weights,
        kv_bytes_per_token=model.kv_bytes_per_token,
        pool_tokens=pool,
        compute=compute,
        tpot_ratio=step_s(model, chip, compute, batch, context)
        / step_s(served, chip, served_compute, batch, context),
        tpot_ratio_one=step_s(model, chip, compute, 1, context)
        / step_s(served, chip, served_compute, 1, context),
        # Prefill is on the roof (note 61 section 4): only faster math shortens it.
        ttft_ratio=chip.peak(served_compute) / chip.peak(compute),
    )

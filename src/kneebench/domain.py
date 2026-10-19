"""Inputs that describe a run: the server setup and the team's workload."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from kneebench.physics.gpus import GpuSpec


class ProductType(StrEnum):
    CHAT = "chat"
    RAG = "rag"
    REASONING = "reasoning"
    BATCH = "batch"


class Source(StrEnum):
    DETECTED = "detected"
    DEFAULT = "default"
    USER = "user"


@dataclass(frozen=True)
class Shape:
    name: str
    isl: int
    osl: int


@dataclass(frozen=True)
class Slo:
    """Latency targets at a percentile (P90 or P99), in milliseconds."""

    ttft_ms: float | None
    tpot_ms: float | None
    percentile: int = 99

    def goodput(self) -> tuple[tuple[str, float], ...]:
        pairs = []
        if self.ttft_ms is not None:
            pairs.append(("time_to_first_token", self.ttft_ms))
        if self.tpot_ms is not None:
            pairs.append(("inter_token_latency", self.tpot_ms))
        return tuple(pairs)


@dataclass(frozen=True)
class Workload:
    product: ProductType
    shapes: tuple[Shape, ...]
    slo: Slo
    time_budget_s: float | None = None
    gpu_hourly_usd: float | None = None


@dataclass(frozen=True)
class ServerSetup:
    gpu: GpuSpec
    gpu_count: int = 1
    max_num_seqs: int | None = None
    enforce_eager: bool | None = None
    sources: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# The standard public shapes (note 26), before capping to the server's context length.
STANDARD_SHAPES = (
    Shape("chat", 1024, 1024),
    Shape("reasoning", 1024, 8192),
    Shape("summarization", 8192, 1024),
)
# Which measured shape stands for each product until the shape sweep covers all of them.
PRIMARY_SHAPE = {
    ProductType.CHAT: "chat",
    ProductType.RAG: "summarization",
    ProductType.REASONING: "reasoning",
    ProductType.BATCH: "chat",
}
# Shapes the user describes from their own traffic are named "yours", "yours-2" ...
USER_SHAPE_PREFIX = "yours"


def user_shapes(spec: str) -> tuple[Shape, ...]:
    """Parse "2000/500, 512/128" (prompt tokens/output tokens) into named shapes."""
    shapes = []
    for i, part in enumerate(p.strip() for p in spec.split(",") if p.strip()):
        try:
            isl, osl = (int(x) for x in part.split("/"))
        except ValueError:
            raise ValueError(
                f"shape {part!r} should look like 2000/500 (prompt/output tokens)"
            ) from None
        if isl <= 0 or osl <= 0:
            raise ValueError(f"shape {part!r} needs positive token counts")
        shapes.append(
            Shape(USER_SHAPE_PREFIX if i == 0 else f"{USER_SHAPE_PREFIX}-{i + 1}", isl, osl)
        )
    return tuple(shapes)


def primary_shape_name(product: ProductType, shape_names: tuple[str, ...]) -> str:
    """The user's own shape when given, else the standard shape for their product."""
    return next((n for n in shape_names if n.startswith(USER_SHAPE_PREFIX)), PRIMARY_SHAPE[product])


# Single-request baseline shape (note 24).
BASELINE_SHAPE = Shape("baseline", 512, 256)
# Room for chat-template tokens added by the server on top of the prompt.
CONTEXT_MARGIN_TOKENS = 128


@dataclass(frozen=True)
class FittedShape:
    shape: Shape
    original: Shape
    adapted: bool


def fit_shape(shape: Shape, max_model_len: int | None) -> FittedShape:
    """Shrink the longer side so prompt + output + margin fits the context (note 26)."""
    if max_model_len is None or shape.isl + shape.osl + CONTEXT_MARGIN_TOKENS <= max_model_len:
        return FittedShape(shape, shape, adapted=False)
    budget = max_model_len - CONTEXT_MARGIN_TOKENS
    if budget < 2:
        raise ValueError(f"max_model_len {max_model_len} is too small to benchmark")
    isl, osl = shape.isl, shape.osl
    if isl >= osl:
        isl = max(budget - osl, budget // 2)
        osl = budget - isl if isl + osl > budget else osl
    else:
        osl = max(budget - isl, budget // 2)
        isl = budget - osl if isl + osl > budget else isl
    return FittedShape(Shape(shape.name, isl, osl), shape, adapted=True)

"""GPU catalog: peak bandwidth, compute and memory per GPU."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files

import yaml

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class GpuSpec:
    id: str
    name: str
    memory_mib: int
    bandwidth_tbps: float
    bf16_tflops: float
    fp8_tflops: float | None = None
    aliases: tuple[str, ...] = ()
    source: str | None = None
    custom: bool = False

    @property
    def memory_bytes(self) -> int:
        return self.memory_mib * MIB

    @property
    def memory_gib(self) -> float:
        return self.memory_bytes / GIB

    @property
    def bandwidth_bytes_s(self) -> float:
        return self.bandwidth_tbps * 1e12

    def peak_flops(self, fp8: bool = False) -> float:
        tflops = self.fp8_tflops if fp8 and self.fp8_tflops else self.bf16_tflops
        return tflops * 1e12


def gpu_from_json(data: dict[str, object]) -> GpuSpec:
    fp8 = data.get("fp8_tflops")
    aliases = data.get("aliases") or ()
    return GpuSpec(
        id=str(data["id"]),
        name=str(data["name"]),
        memory_mib=int(str(data["memory_mib"])),
        bandwidth_tbps=float(str(data["bandwidth_tbps"])),
        bf16_tflops=float(str(data["bf16_tflops"])),
        fp8_tflops=float(str(fp8)) if fp8 is not None else None,
        aliases=tuple(str(a) for a in aliases) if isinstance(aliases, list | tuple) else (),
        source=str(data["source"]) if data.get("source") else None,
        custom=bool(data.get("custom", False)),
    )


def custom_gpu(
    name: str,
    memory_gib: float,
    bandwidth_tbps: float,
    bf16_tflops: float,
    fp8_tflops: float | None = None,
) -> GpuSpec:
    return GpuSpec(
        id="custom",
        name=name,
        memory_mib=round(memory_gib * 1024),
        bandwidth_tbps=bandwidth_tbps,
        bf16_tflops=bf16_tflops,
        fp8_tflops=fp8_tflops,
        custom=True,
    )


@cache
def catalog() -> tuple[GpuSpec, ...]:
    raw = yaml.safe_load(files("kneebench.physics").joinpath("gpus.yaml").read_text())
    return tuple(
        GpuSpec(
            id=g["id"],
            name=g["name"],
            memory_mib=int(g["memory_mib"]),
            bandwidth_tbps=float(g["bandwidth_tbps"]),
            bf16_tflops=float(g["bf16_tflops"]),
            fp8_tflops=float(g["fp8_tflops"]) if g.get("fp8_tflops") else None,
            aliases=tuple(a.lower() for a in g.get("aliases", [])),
            source=g.get("source"),
        )
        for g in raw["gpus"]
    )


def get_gpu(gpu_id: str) -> GpuSpec:
    for gpu in catalog():
        if gpu.id == gpu_id:
            return gpu
    raise KeyError(f"unknown GPU id {gpu_id!r}; known: {', '.join(g.id for g in catalog())}")


def match_gpu(text: str) -> GpuSpec | None:
    """Match a free-form GPU name (e.g. from nvidia-smi) to a catalog entry."""
    needle = text.strip().lower()
    for gpu in catalog():
        if needle in (gpu.id, gpu.name.lower(), *gpu.aliases):
            return gpu
    return None

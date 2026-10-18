"""Everything a rule may look at, built from stored data (never from a live server)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kneebench.domain import ProductType, Shape, Slo
from kneebench.phases.theory import Theory, theory_from_json
from kneebench.servers.vllm import metric_names

# Network share of TTFT at or above which subtracting it leaves too little to trust.
UNRELIABLE_NETWORK_SHARE = 0.5


@dataclass(frozen=True)
class PointView:
    id: int
    seq: int
    phase: str
    isl: int
    osl: int
    mode: str
    concurrency: int | None
    status: str
    metrics: dict[str, dict[str, float]]
    server: dict[str, dict[str, float]]
    warnings: tuple[dict[str, str], ...] = ()

    def m(self, metric: str, stat: str = "p50") -> float | None:
        return self.metrics.get(metric, {}).get(stat)

    def server_stat(self, signal: str, stat: str) -> float | None:
        for name in metric_names(signal):
            if name in self.server and stat in self.server[name]:
                return self.server[name][stat]
        return None

    def has_warning(self, title: str) -> bool:
        return any(title.lower() in w["code"].lower() for w in self.warnings)


@dataclass(frozen=True)
class SetupView:
    gpu_id: str
    gpu_name: str
    gpu_custom: bool
    gpu_count: int
    max_num_seqs: int | None
    enforce_eager: bool | None
    sources: dict[str, str]


@dataclass(frozen=True)
class WorkloadView:
    product: ProductType
    shapes: tuple[Shape, ...]
    slo: Slo


@dataclass(frozen=True)
class RunContext:
    run_id: int
    status: str
    started_at: str
    kneebench_version: str
    aiperf_version: str
    vllm_version: str | None
    served_model: str | None
    hf_model_id: str | None
    max_model_len: int | None
    metrics_available: bool
    theory: Theory
    setup: SetupView
    workload: WorkloadView
    network: dict[str, Any] | None
    client: dict[str, Any]
    points: tuple[PointView, ...] = field(default=())

    def points_in(self, phase: str) -> list[PointView]:
        return [p for p in self.points if p.phase == phase and p.status == "done"]

    @property
    def baseline(self) -> PointView | None:
        pts = self.points_in("baseline")
        return pts[0] if pts else None

    @property
    def http_rtt_ms(self) -> float | None:
        return (self.network or {}).get("http_rtt_ms")

    def server_ttft_ms(self, point: PointView, stat: str = "p50") -> float | None:
        """TTFT with the network round trip removed, or None when the network dominates."""
        ttft = point.m("time_to_first_token", stat)
        rtt = self.http_rtt_ms
        if ttft is None:
            return None
        if rtt is None:
            return ttft
        if rtt / ttft >= UNRELIABLE_NETWORK_SHARE:
            return None
        return ttft - rtt


def _loads(value: str | None) -> Any:
    return json.loads(value) if value else None


def context_from_bundle(bundle: dict[str, Any]) -> RunContext:
    run, setup, snap, workload = (
        bundle["run"],
        bundle["setup"],
        bundle["snapshot"],
        bundle["workload"],
    )
    gpu = _loads(setup["gpu_json"])
    slo = _loads(workload["slo_json"])
    points = tuple(
        PointView(
            id=p["id"],
            seq=p["seq"],
            phase=p["phase"],
            isl=p["isl"],
            osl=p["osl"],
            mode=p["mode"],
            concurrency=p["concurrency"],
            status=p["status"],
            metrics=p["metrics"],
            server=p["server"],
            warnings=tuple(p["warnings"]),
        )
        for p in bundle["points"]
    )
    return RunContext(
        run_id=run["id"],
        status=run["status"],
        started_at=run["started_at"],
        kneebench_version=run["kneebench_version"],
        aiperf_version=run["aiperf_version"],
        vllm_version=run["vllm_version"],
        served_model=snap["served_model"],
        hf_model_id=snap["hf_model_id"],
        max_model_len=snap["max_model_len"],
        metrics_available=snap["cache_config_json"] is not None,
        theory=theory_from_json(_loads(run["theory_json"])),
        setup=SetupView(
            gpu_id=gpu["id"],
            gpu_name=gpu["name"],
            gpu_custom=bool(gpu.get("custom")),
            gpu_count=setup["gpu_count"],
            max_num_seqs=setup["max_num_seqs"],
            enforce_eager=None if setup["enforce_eager"] is None else bool(setup["enforce_eager"]),
            sources=_loads(setup["source_json"]) or {},
        ),
        workload=WorkloadView(
            product=ProductType(workload["product_type"]),
            shapes=tuple(Shape(**s) for s in _loads(workload["shapes_json"])),
            slo=Slo(**slo),
        ),
        network=_loads(run["network_json"]),
        client=_loads(run["client_json"]) or {},
        points=points,
    )

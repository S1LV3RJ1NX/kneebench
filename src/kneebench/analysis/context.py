"""Everything a rule may look at, built from stored data (never from a live server)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kneebench.domain import ProductType, Shape, Slo, primary_shape_name
from kneebench.phases.theory import Theory, theory_from_json
from kneebench.servers.vllm import metric_names

# Network share of TTFT at or above which subtracting it leaves too little to trust.
UNRELIABLE_NETWORK_SHARE = 0.5


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear interpolation between closest ranks, as numpy and AIPerf compute it."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


@dataclass(frozen=True)
class RequestView:
    start_ns: int
    end_ns: int
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float | None
    isl: int | None
    osl: int | None
    error: str | None = None
    error_code: int | None = None
    connect_ms: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


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
    requests: tuple[RequestView, ...] = ()
    stop_reason: str | None = None
    request_rate: float | None = None
    arrival: str | None = None
    burstiness: float | None = None

    def m(self, metric: str, stat: str = "p50") -> float | None:
        return self.metrics.get(metric, {}).get(stat)

    @property
    def new_connections(self) -> list[float]:
        """Setup times of requests that opened a new connection during the measurement."""
        return [r.connect_ms for r in self.requests if r.ok and r.connect_ms]

    def server_ttft(self, offset_ms: float = 0.0) -> dict[str, float]:
        """TTFT percentiles without the network round trip and without connection setup.

        A client far from the server pays DNS, TCP and TLS on every new connection; that
        isn't the server's time, and it only lands on some requests, so it inflates the tail.
        """
        timed = [r for r in self.requests if r.ok and r.ttft_ms is not None]
        if not self.new_connections or not timed:
            stats = self.metrics.get("time_to_first_token", {})
            return {
                k: max(v - offset_ms, 0.0)
                for k, v in stats.items()
                if k in ("p50", "p90", "p99", "avg")
            }
        values = sorted(
            max((r.ttft_ms or 0.0) - (r.connect_ms or 0.0) - offset_ms, 0.0) for r in timed
        )
        return {
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p99": _percentile(values, 99),
            "avg": sum(values) / len(values),
        }

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
    error: str | None = None

    def points_in(self, phase: str) -> list[PointView]:
        return [p for p in self.points if p.phase == phase and p.status == "done"]

    @property
    def baseline(self) -> PointView | None:
        pts = self.points_in("baseline")
        return pts[0] if pts else None

    @property
    def primary_shape(self) -> str:
        return primary_shape_name(
            self.workload.product, tuple(s.name for s in self.workload.shapes)
        )

    @property
    def http_rtt_ms(self) -> float | None:
        return (self.network or {}).get("http_rtt_ms")

    @property
    def prompt_overhead_tokens(self) -> float | None:
        """Tokens the server adds to every prompt (the chat template), from usage counts."""
        offsets = [
            server - client
            for p in self.points
            if (client := p.m("input_sequence_length", "avg"))
            and (server := p.m("usage_prompt_tokens", "avg"))
        ]
        return sum(offsets) / len(offsets) if offsets else None

    @property
    def ttft_offset_ms(self) -> float:
        """Network time taken off every first token (see `ttft_offset`)."""
        base = self.theory.shape("baseline")
        return ttft_offset(self.http_rtt_ms, self.baseline, base.ttft_floor_ms if base else None)

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
        return ttft - self.ttft_offset_ms


def ttft_offset(rtt_ms: float | None, baseline: PointView | None, floor_ms: float | None) -> float:
    """The network time to take off every first token.

    It is the keep-alive round trip to /health, capped so the fastest baseline request still
    leaves the server its physics floor: a slow /health path (a gateway, a cold handler) can
    take longer than a whole streamed first token, and removing all of it would make the
    server look faster than physics allows.
    """
    if not rtt_ms:
        return 0.0
    if baseline is None or floor_ms is None:
        return rtt_ms
    times = [
        r.ttft_ms - (r.connect_ms or 0.0)
        for r in baseline.requests
        if r.ok and r.ttft_ms is not None
    ]
    fastest = min(times) if times else baseline.m("time_to_first_token", "min")
    if fastest is None:
        return rtt_ms
    return max(0.0, min(rtt_ms, fastest - floor_ms))


def _loads(value: str | None) -> Any:
    return json.loads(value) if value else None


def point_from_dict(p: dict[str, Any]) -> PointView:
    spec = _loads(p.get("spec_json")) or {}
    return PointView(
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
        requests=tuple(RequestView(**r) for r in p.get("requests", [])),
        stop_reason=p.get("stop_reason"),
        request_rate=p.get("request_rate"),
        arrival=spec.get("arrival"),
        burstiness=spec.get("burstiness"),
    )


def context_from_bundle(bundle: dict[str, Any]) -> RunContext:
    run, setup, snap, workload = (
        bundle["run"],
        bundle["setup"],
        bundle["snapshot"],
        bundle["workload"],
    )
    gpu = _loads(setup["gpu_json"])
    slo = _loads(workload["slo_json"])
    points = tuple(point_from_dict(p) for p in bundle["points"])
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
        error=run.get("error"),
    )

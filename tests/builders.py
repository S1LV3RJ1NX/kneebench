"""Build RunContexts for rule tests, and stored runs from recorded fixtures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from kneebench.analysis.context import (
    PointView,
    RequestView,
    RunContext,
    SetupView,
    WorkloadView,
)
from kneebench.analysis.engine import REGISTRY, Finding, Status, _load_rules
from kneebench.content.render import TermTracker, make_env, render_finding
from kneebench.detect.metrics import CacheConfig, cache_config, parse_samples
from kneebench.detect.netprobe import NetworkProfile
from kneebench.detect.server import ModelMetadata, ServedModel, ServerSnapshot
from kneebench.domain import STANDARD_SHAPES, ProductType, ServerSetup, Slo, Workload
from kneebench.loadgen.aiperf.parse import parse_point
from kneebench.loadgen.base import LoadMode, PointResult, PointSpec
from kneebench.phases.theory import Theory, compute_theory
from kneebench.physics.floors import Hardware
from kneebench.physics.gpus import get_gpu
from kneebench.report.composer import common_vars
from kneebench.store import repo
from tests.fixture_data import FIXTURES, LLAMA_8B, hf_fixture, model_fixture

CACHE_65GIB = CacheConfig(
    gpu_memory_utilization=0.9,
    kv_cache_size_tokens=532_768,
    kv_cache_max_concurrency=65.03,
    block_size=16,
    enable_prefix_caching=True,
    cache_dtype="auto",
    num_gpu_blocks=33_298,
)


def theory_for(
    model_id: str = LLAMA_8B,
    gpu: str = "h100-nvl",
    count: int = 1,
    cache: CacheConfig | None = CACHE_65GIB,
    max_model_len: int | None = 8192,
    vllm_version: str | None = "0.29.0",
) -> Theory:
    return compute_theory(
        model_fixture(model_id),
        Hardware(get_gpu(gpu), count),
        cache,
        max_model_len,
        STANDARD_SHAPES,
        vllm_version,
    )


def point(
    phase: str = "baseline",
    isl: int = 512,
    osl: int = 256,
    concurrency: int = 1,
    seq: int = 0,
    tpot_p50: float = 5.8,
    tpot_avg: float | None = None,
    tpot_std: float = 0.1,
    tpot_p99: float | None = None,
    ttft_p50: float = 30.0,
    osl_mean: float | None = None,
    requests: int = 30,
    warnings: tuple[str, ...] = (),
    status: str = "done",
    extra: dict[str, dict[str, float]] | None = None,
) -> PointView:
    metrics: dict[str, dict[str, float]] = {
        "inter_token_latency": {
            "p50": tpot_p50,
            "avg": tpot_avg or tpot_p50,
            "std": tpot_std,
            "p99": tpot_p99 or tpot_p50 * 1.1,
        },
        "time_to_first_token": {"p50": ttft_p50, "p99": ttft_p50 * 1.2, "avg": ttft_p50},
        "output_sequence_length": {"avg": osl_mean if osl_mean is not None else float(osl)},
        "osl_mismatch_count": {"avg": 0.0},
        "request_count": {"avg": float(requests)},
        **(extra or {}),
    }
    return PointView(
        id=seq + 1,
        seq=seq,
        phase=phase,
        isl=isl,
        osl=osl,
        mode="concurrency",
        concurrency=concurrency,
        status=status,
        metrics=metrics,
        server={},
        warnings=tuple({"source": "aiperf", "code": w, "message": ""} for w in warnings),
    )


def closed_loop_requests(
    concurrency: int,
    rounds: int,
    e2e_s: float,
    osl: int,
    ttft_ms: float = 50.0,
    tpot_ms: float = 10.0,
    errors: int = 0,
    error_code: int | None = None,
    isl: int = 1024,
) -> tuple[RequestView, ...]:
    """Requests as a closed loop issues them: c slots, each running `rounds` back to back."""
    out: list[RequestView] = []
    span = int(e2e_s * 1e9)
    for slot in range(concurrency):
        for k in range(rounds):
            start = k * span + slot * 1000  # tiny offsets keep events ordered
            failed = len(out) < errors
            out.append(
                RequestView(
                    start_ns=start,
                    end_ns=start + span,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    e2e_ms=e2e_s * 1000,
                    isl=isl,
                    osl=osl,
                    error="error" if failed else None,
                    error_code=error_code if failed else None,
                )
            )
    return tuple(out)


def sweep_point(
    concurrency: int,
    tok_s: float,
    tpot_p50: float,
    tpot_p99: float | None = None,
    ttft_p99: float = 200.0,
    isl: int = 1024,
    osl: int = 1024,
    kv_max: float | None = 0.3,
    preemptions: float | None = 0.0,
    waiting_max: float | None = 0.0,
    running_max: float | None = None,
    errors: int = 0,
    error_code: int | None = None,
    effective: float | None = None,
    rounds: int = 5,
    seq: int = 0,
    stop_reason: str | None = None,
) -> PointView:
    """A concurrency point whose steady-state throughput works out to about `tok_s`."""
    e2e_s = concurrency * osl / tok_s
    requests = closed_loop_requests(
        concurrency,
        rounds,
        e2e_s,
        osl,
        tpot_ms=tpot_p50,
        errors=errors,
        error_code=error_code,
        isl=isl,
    )
    server: dict[str, dict[str, float]] = {}
    if kv_max is not None:
        server["vllm:kv_cache_usage_perc"] = {"max": kv_max}
    if preemptions is not None:
        server["vllm:num_preemptions"] = {"total": preemptions}
    if waiting_max is not None:
        server["vllm:num_requests_waiting"] = {"max": waiting_max}
    server["vllm:num_requests_running"] = {
        "max": running_max if running_max is not None else float(concurrency)
    }
    metrics = {
        "inter_token_latency": {
            "p50": tpot_p50,
            "p90": (tpot_p99 or tpot_p50 * 1.2) * 0.95,
            "p99": tpot_p99 or tpot_p50 * 1.2,
            "avg": tpot_p50,
        },
        "time_to_first_token": {"p50": ttft_p99 / 2, "p90": ttft_p99 * 0.9, "p99": ttft_p99},
        "output_sequence_length": {"avg": float(osl)},
        "osl_mismatch_count": {"avg": 0.0},
        "request_count": {"avg": float(len(requests))},
        "effective_concurrency": {
            "avg": effective if effective is not None else float(concurrency)
        },
    }
    return PointView(
        id=100 + seq,
        seq=seq,
        phase="concurrency",
        isl=isl,
        osl=osl,
        mode="concurrency",
        concurrency=concurrency,
        status="done",
        metrics=metrics,
        server=server,
        requests=requests,
        stop_reason=stop_reason,
    )


def healthy_sweep(**overrides: Any) -> tuple[PointView, ...]:
    """1..64 users: near-linear growth, knee at 16, saturation at 32, TPOT P99 crosses 50 ms."""
    rows = [  # c, tok/s, TPOT P50, TPOT P99
        (1, 170, 5.8, 6.2),
        (2, 335, 5.9, 6.4),
        (4, 650, 6.1, 6.8),
        (8, 1250, 6.4, 7.5),
        (16, 2300, 7.0, 9.0),
        (32, 3000, 10.5, 30.0),
        (64, 3080, 20.0, 80.0),
    ]
    return tuple(
        sweep_point(c, tok, p50, p99, seq=i + 1, **overrides)
        for i, (c, tok, p50, p99) in enumerate(rows)
    )


def ctx(
    theory: Theory | None = None,
    points: tuple[PointView, ...] | None = None,
    network: dict[str, Any] | None = None,
    gpu: str = "h100-nvl",
    gpu_count: int = 1,
    enforce_eager: bool | None = False,
    metrics_available: bool = True,
    vllm_version: str | None = "0.29.0",
    product: ProductType = ProductType.CHAT,
) -> RunContext:
    g = get_gpu(gpu)
    return RunContext(
        run_id=1,
        status="done",
        started_at="2026-01-01T00:00:00+00:00",
        kneebench_version="0.0.test",
        aiperf_version="0.13.0",
        vllm_version=vllm_version,
        served_model="served-model",
        hf_model_id=LLAMA_8B,
        max_model_len=8192,
        metrics_available=metrics_available,
        theory=theory or theory_for(gpu=gpu, count=gpu_count, vllm_version=vllm_version),
        setup=SetupView(
            gpu_id=g.id,
            gpu_name=g.name,
            gpu_custom=False,
            gpu_count=gpu_count,
            max_num_seqs=None,
            enforce_eager=enforce_eager,
            sources={},
        ),
        workload=WorkloadView(
            product=product, shapes=STANDARD_SHAPES, slo=Slo(ttft_ms=500, tpot_ms=50)
        ),
        network=network
        if network is not None
        else {"http_rtt_ms": 1.0, "tcp_connect_ms": 0.5, "tls_handshake_ms": None},
        client={},
        points=points if points is not None else (point(),),
    )


def run_rule(rule_id: str, context: RunContext) -> Finding | None:
    _load_rules()
    found = REGISTRY[rule_id].fn(context)
    if found is not None:
        assert_renders(found, context)
    return found


def assert_renders(found: Finding, context: RunContext) -> None:
    """Every finding a test produces must render with the content layer (StrictUndefined)."""
    rendered = render_finding(
        Finding(found.rule_id, found.status, found.data, REGISTRY[found.rule_id].tags),
        common_vars(context),
        make_env(TermTracker()),
    )
    assert rendered.status_line
    if found.status in (Status.WARN, Status.BAD):
        assert rendered.why, f"{found.rule_id} {found.status} has no explanation"


# ---- a stored run built from recorded, sanitized fixtures ----

BASELINE_FIXTURE = FIXTURES / "aiperf" / "vllm-baseline-c1-512-256"
ENDPOINT_FIXTURE = FIXTURES / "endpoint" / "vllm-0.29"


def fixture_snapshot() -> ServerSnapshot:
    metrics = (ENDPOINT_FIXTURE / "metrics.prom").read_text()
    models = json.loads((ENDPOINT_FIXTURE / "models.json").read_text())["data"]
    net = json.loads((ENDPOINT_FIXTURE / "network.json").read_text())
    net["http_rtt_samples"] = tuple(net["http_rtt_samples"])
    return ServerSnapshot(
        base_url="http://vllm.example:8000",
        reachable=True,
        vllm_version=json.loads((ENDPOINT_FIXTURE / "version.json").read_text())["version"],
        models=[
            ServedModel(
                id=m["id"], root=m["root"], max_model_len=m["max_model_len"], hf_id=LLAMA_8B
            )
            for m in models
        ],
        cache=cache_config(parse_samples(metrics)),
        metrics_available=True,
        metrics_text=metrics,
        network=NetworkProfile(**net),
    )


def store_fixture_run(
    conn: Connection,
    gpu: str = "h100-nvl",
    product: ProductType = ProductType.CHAT,
    snapshot: ServerSnapshot | None = None,
    artifact_dir: Path = BASELINE_FIXTURE,
) -> int:
    snap = snapshot or fixture_snapshot()
    served = snap.models[0]
    config, params = hf_fixture(LLAMA_8B)
    metadata = ModelMetadata(hf_id=LLAMA_8B, config=config, safetensors_params=params)
    workload = Workload(product=product, shapes=STANDARD_SHAPES, slo=Slo(500, 50))
    theory = compute_theory(
        model_fixture(LLAMA_8B),
        Hardware(get_gpu(gpu)),
        snap.cache,
        served.max_model_len,
        STANDARD_SHAPES,
        snap.vllm_version,
    )
    endpoint_id = repo.upsert_endpoint(conn, snap.base_url)
    snapshot_id = repo.insert_snapshot(conn, endpoint_id, snap, served, metadata)
    setup_id = repo.insert_setup(
        conn, endpoint_id, snapshot_id, ServerSetup(gpu=get_gpu(gpu), enforce_eager=False)
    )
    workload_id = repo.insert_workload(conn, workload)
    net = snap.network
    run_id = repo.create_run(
        conn,
        endpoint_id=endpoint_id,
        setup_id=setup_id,
        workload_id=workload_id,
        kneebench_version="0.0.test",
        aiperf_version="0.13.0",
        vllm_version=snap.vllm_version,
        client={"os": "test"},
        network=None
        if net is None
        else {
            k: getattr(net, k)
            for k in ("dns_ms", "tcp_connect_ms", "tls_handshake_ms", "http_rtt_ms")
        },
        theory=theory.to_json(),
    )
    spec = PointSpec(
        isl=512,
        osl=256,
        mode=LoadMode.CONCURRENCY,
        concurrency=1,
        request_count=30,
        warmup_count=10,
    )
    point_id = repo.create_point(conn, run_id, 0, "baseline", spec)
    result = PointResult(
        spec=spec,
        artifact_dir=artifact_dir,
        exit_code=0,
        wall_time_s=82.8,
        argv_redacted=("aiperf", "profile"),
    )
    repo.finish_point(conn, point_id, result, parse_point(artifact_dir), overhead_s=27.5)
    repo.finish_run(conn, run_id, "done")
    return run_id


def without_metrics(snap: ServerSnapshot) -> ServerSnapshot:
    return replace(snap, cache=None, metrics_available=False, metrics_text=None)


def rate_point(
    requested: float,
    ttft_ms: float = 150.0,
    ttft_end_ms: float | None = None,
    arrival: str = "poisson",
    issued_share: float = 1.0,
    duration_s: float = 90.0,
    tpot_p99: float = 12.0,
    seq: int = 50,
    errors: int = 0,
) -> PointView:
    """A rate point whose requests start evenly over the run; TTFT ramps if ttft_end_ms."""
    count = max(int(requested * issued_share * duration_s), 4)
    end_ttft = ttft_end_ms if ttft_end_ms is not None else ttft_ms
    reqs = []
    for i in range(count):
        frac = i / (count - 1)
        start = int(frac * duration_s * 1e9)
        ttft = ttft_ms + (end_ttft - ttft_ms) * frac
        reqs.append(
            RequestView(
                start_ns=start,
                end_ns=start + int((ttft + 5000) * 1e6),
                ttft_ms=ttft,
                tpot_ms=8.0,
                e2e_ms=ttft + 5000,
                isl=1024,
                osl=1024,
                error="e" if i < errors else None,
            )
        )
    tail = max(ttft_ms, end_ttft)
    return PointView(
        id=200 + seq,
        seq=seq,
        phase="rate",
        isl=1024,
        osl=1024,
        mode="rate",
        concurrency=2048,
        status="done",
        metrics={
            "time_to_first_token": {
                "p50": (ttft_ms + end_ttft) / 2,
                "p90": tail * 0.95,
                "p99": tail,
            },
            "inter_token_latency": {"p50": 8.0, "p90": tpot_p99 * 0.9, "p99": tpot_p99},
            "request_count": {"avg": float(count)},
        },
        server={},
        requests=tuple(reqs),
        request_rate=requested,
        arrival=arrival,
        burstiness=0.5 if arrival == "gamma" else None,
    )


def rate_sweep() -> tuple[PointView, ...]:
    """Capacity from healthy_sweep is ~2.9 req/s; flat TTFT below, a growing queue past it."""
    rows = [
        (0.73, 120, None),
        (1.46, 140, None),
        (2.2, 260, None),
        (2.6, 420, None),
        (3.2, 600, 9000),
    ]
    pts = [rate_point(r, t, e, seq=60 + i) for i, (r, t, e) in enumerate(rows)]
    pts += [
        rate_point(2.2, 700, arrival="gamma", seq=70),
        rate_point(2.6, 1400, arrival="gamma", seq=71),
    ]
    return tuple(pts)

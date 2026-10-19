"""Run preparation shared by the CLI and the web UI: detect, describe, plan."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Connection

from kneebench import content
from kneebench.detect.server import ServerSnapshot, detect_server, fetch_model_metadata
from kneebench.domain import STANDARD_SHAPES, ProductType, ServerSetup, Slo, Source, Workload
from kneebench.loadgen.base import Endpoint
from kneebench.phases.plan import DEFAULT_MAX_CONCURRENCY, RunPlan, plan_run
from kneebench.phases.runner import RunInputs
from kneebench.phases.theory import compute_theory, theory_from_snapshot
from kneebench.physics.floors import Hardware
from kneebench.physics.gpus import GpuSpec, get_gpu, gpu_from_json
from kneebench.physics.model import model_spec_from_hf
from kneebench.store import repo

ALL_PHASES = ("baseline", "concurrency", "rate")


class PrepareError(ValueError):
    pass


@dataclass(frozen=True)
class RunRequest:
    """Everything the user tells kneebench; the rest is detected."""

    url: str
    gpu: GpuSpec
    api_key: str | None = field(default=None, repr=False)
    gpu_count: int = 1
    model: str | None = None
    tokenizer: str | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool | None = None
    product: ProductType = ProductType.CHAT
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None
    slo_percentile: int | None = None
    phases: tuple[str, ...] = ALL_PHASES
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    budget_s: float | None = None
    label: str | None = None


def slo_for(req: RunRequest) -> Slo:
    preset = content.load("slo_presets.yaml")["presets"][req.product.value]
    return Slo(
        ttft_ms=req.slo_ttft_ms if req.slo_ttft_ms is not None else preset["ttft_ms"],
        tpot_ms=req.slo_tpot_ms if req.slo_tpot_ms is not None else preset["tpot_ms"],
        percentile=req.slo_percentile or preset["percentile"],
    )


def prepare(req: RunRequest, snapshot: ServerSnapshot | None = None) -> tuple[RunInputs, RunPlan]:
    snap = snapshot or detect_server(req.url, req.api_key)
    if not snap.models:
        raise PrepareError(f"{snap.base_url} didn't list any models at /v1/models")
    served = snap.model(req.model)
    if served is None:
        raise PrepareError(f"model {req.model!r} not served; found {[m.id for m in snap.models]}")
    hf_id = req.tokenizer or served.hf_id
    if not hf_id:
        raise PrepareError(
            "couldn't map the served model to a Hugging Face id; enter the tokenizer"
        )
    metadata = fetch_model_metadata(hf_id)
    model_spec = model_spec_from_hf(metadata.config, metadata.safetensors_params, hf_id=hf_id)

    slo = slo_for(req)
    workload = Workload(
        product=req.product, shapes=STANDARD_SHAPES, slo=slo, time_budget_s=req.budget_s
    )
    setup = ServerSetup(
        gpu=req.gpu,
        gpu_count=req.gpu_count,
        max_num_seqs=req.max_num_seqs,
        enforce_eager=req.enforce_eager,
        sources={
            "gpu": Source.USER,
            "gpu_count": Source.USER,
            "max_num_seqs": Source.USER if req.max_num_seqs else Source.DEFAULT,
            "enforce_eager": Source.USER if req.enforce_eager is not None else Source.DEFAULT,
        },
    )
    theory = compute_theory(
        model_spec,
        Hardware(req.gpu, req.gpu_count),
        snap.cache,
        served.max_model_len,
        workload.shapes,
        snap.vllm_version,
    )
    if "rate" in req.phases and "concurrency" not in req.phases:
        raise PrepareError(
            "The request-rate sweep needs the concurrency sweep (it measures "
            "the capacity the rates are based on)."
        )
    plan = plan_run(
        theory,
        req.phases,
        product=req.product,
        slo=slo,
        max_concurrency=req.max_concurrency,
        max_num_seqs=req.max_num_seqs,
        budget_s=req.budget_s,
    )
    endpoint = Endpoint(
        base_url=snap.base_url, model=served.id, tokenizer=hf_id, api_key=req.api_key
    )
    inputs = RunInputs(
        endpoint=endpoint,
        snapshot=snap,
        served=served,
        metadata=metadata,
        setup=setup,
        workload=workload,
        theory=theory,
        label=req.label,
    )
    return inputs, plan


def correct_gpu(
    conn: Connection, run_id: int, gpu: str | None, gpu_count: int | None
) -> tuple[GpuSpec, int]:
    """Replace a run's GPU entry and recompute everything derived from it; no rerun."""
    bundle = repo.run_bundle(conn, run_id)
    old = bundle["setup"]
    spec = get_gpu(gpu) if gpu else gpu_from_json(json.loads(old["gpu_json"]))
    count = gpu_count or old["gpu_count"]
    sources = {**json.loads(old["source_json"]), "gpu": Source.USER, "gpu_count": Source.USER}
    setup = ServerSetup(
        gpu=spec,
        gpu_count=count,
        max_num_seqs=old["max_num_seqs"],
        enforce_eager=None if old["enforce_eager"] is None else bool(old["enforce_eager"]),
        sources=sources,
    )
    theory = theory_from_snapshot(bundle["snapshot"], bundle["workload"], Hardware(spec, count))
    repo.correct_setup(conn, run_id, setup, theory.to_json())
    return spec, count


def delete_run_data(conn: Connection, home: Path, run_id: int) -> None:
    """Remove a run from the database and its AIPerf artifacts from disk."""
    repo.delete_run(conn, run_id)
    shutil.rmtree(home / "runs" / str(run_id), ignore_errors=True)
    (home / "reports" / f"run-{run_id}.md").unlink(missing_ok=True)

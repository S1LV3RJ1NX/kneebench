"""The whole M1 pipeline against a mock server that is not vLLM (no /version, no /metrics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kneebench.analysis.engine import Status
from kneebench.detect.server import ModelMetadata, detect_server
from kneebench.domain import STANDARD_SHAPES, ProductType, ServerSetup, Slo, Workload
from kneebench.loadgen.aiperf.runner import AiperfLoadGenerator
from kneebench.loadgen.base import Endpoint
from kneebench.phases.plan import plan_run
from kneebench.phases.runner import RunInputs, execute_run
from kneebench.phases.theory import compute_theory
from kneebench.physics.floors import Hardware
from kneebench.physics.gpus import get_gpu
from kneebench.report.build import build_report
from kneebench.report.render_md import render_markdown
from kneebench.store.db import connect_memory
from tests.fixture_data import LLAMA_8B, hf_fixture, model_fixture
from tests.integration.conftest import MOCK_MODEL

pytestmark = pytest.mark.integration


def test_detect_run_and_report_against_non_vllm_server(
    mock_server_url: str, tmp_path: Path
) -> None:
    snap = detect_server(mock_server_url, None)
    assert snap.vllm_version is None
    assert not snap.metrics_available
    served = snap.model(MOCK_MODEL)
    assert served is not None

    config, params = hf_fixture(LLAMA_8B)
    gpu = get_gpu("l40s")
    theory = compute_theory(
        model_fixture(LLAMA_8B), Hardware(gpu), snap.cache, 4096, STANDARD_SHAPES, snap.vllm_version
    )
    plan = plan_run(theory, ("baseline",))
    inputs = RunInputs(
        endpoint=Endpoint(base_url=snap.base_url, model=served.id, tokenizer="builtin"),
        snapshot=snap,
        served=served,
        metadata=ModelMetadata(hf_id=LLAMA_8B, config=config, safetensors_params=params),
        setup=ServerSetup(gpu=gpu),
        theory=theory,
        workload=Workload(ProductType.CHAT, STANDARD_SHAPES, Slo(500, 50)),
    )
    conn = connect_memory()
    run_id = execute_run(conn, inputs, plan, AiperfLoadGenerator(), tmp_path)

    report = build_report(conn, run_id)
    statuses = {f["rule_id"]: f["status"] for f in report.findings_json}
    assert statuses["H08"] == Status.WARN  # no /metrics: memory findings are blind
    assert statuses["H07"] == Status.WARN  # no vLLM version reported
    assert statuses["H04"] == Status.BAD  # the mock ignores max tokens, like a missing ignore_eos
    assert statuses["T02"] == Status.SKIP
    assert "Server metrics not visible" in render_markdown(report)

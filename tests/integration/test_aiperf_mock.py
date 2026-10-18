"""Run the bundled AIPerf end to end against a mock OpenAI server (no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kneebench.loadgen.aiperf.runner import STDOUT_LOG, AiperfLoadGenerator
from kneebench.loadgen.base import Endpoint, LoadMode, PointSpec
from tests.integration.conftest import MOCK_MODEL

pytestmark = pytest.mark.integration

SENTINEL = "sk-kneebench-integration-sentinel"


def _endpoint(url: str) -> Endpoint:
    return Endpoint(base_url=url, model=MOCK_MODEL, tokenizer="builtin", api_key=SENTINEL)


def test_point_produces_exports_and_never_leaks_key(mock_server_url: str, tmp_path: Path) -> None:
    spec = PointSpec(
        isl=128,
        osl=32,
        mode=LoadMode.CONCURRENCY,
        concurrency=2,
        request_count=12,
        warmup_count=2,
        slice_duration_s=1.0,
    )
    result = AiperfLoadGenerator().run_point(_endpoint(mock_server_url), spec, tmp_path / "pt")

    assert result.ok, result.stderr_tail or result.stdout_tail
    art = result.artifact_dir
    summary = json.loads((art / "profile_export_aiperf.json").read_text())
    assert summary["request_count"]["avg"] == 12
    assert "p99" in summary["time_to_first_token"]
    assert "p99" in summary["inter_token_latency"]
    assert (art / "profile_export.jsonl").exists()
    assert (art / "profile_export_aiperf_timeslices.json").exists()

    for path in art.rglob("*"):
        if path.is_file():
            assert SENTINEL.encode() not in path.read_bytes(), f"key leaked into {path}"
    assert all(SENTINEL not in arg for arg in result.argv_redacted)


def test_short_outputs_surface_aiperf_mismatch_warning(
    mock_server_url: str, tmp_path: Path
) -> None:
    """The mock ignores max tokens, like a server that ignores ignore_eos (note 26 trap)."""
    spec = PointSpec(
        isl=64,
        osl=512,
        mode=LoadMode.CONCURRENCY,
        concurrency=1,
        request_count=3,
        slice_duration_s=None,
    )
    result = AiperfLoadGenerator().run_point(_endpoint(mock_server_url), spec, tmp_path / "pt")

    assert result.ok
    assert "Output Sequence Length Mismatch" in (result.artifact_dir / STDOUT_LOG).read_text()

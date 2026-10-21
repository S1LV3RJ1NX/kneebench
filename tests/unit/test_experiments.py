"""Guided experiments: parsing a change, rendering it per deployment, checking it landed."""

from dataclasses import replace

from kneebench.experiments import Change, applied_check, parse_flag, snippet
from tests.builders import ENDPOINT_FIXTURE, fixture_snapshot


def test_parse_flag() -> None:
    assert parse_flag("--max-num-seqs 118") == Change("--max-num-seqs", "118", False)
    assert parse_flag("--gpu-memory-utilization=0.95") == Change(
        "--gpu-memory-utilization", "0.95", False
    )
    assert parse_flag("remove --enforce-eager") == Change("--enforce-eager", None, True)
    assert parse_flag("--enable-chunked-prefill") == Change("--enable-chunked-prefill", None, False)


def test_snippet_per_deployment_style() -> None:
    cli = snippet("cli", "--max-num-seqs 118", "meta-llama/Llama-3.1-8B-Instruct")
    assert "vllm serve meta-llama/Llama-3.1-8B-Instruct" in cli["code"]
    assert cli["code"].endswith("--max-num-seqs 118")
    k8s = snippet("kubernetes", "--max-num-seqs 118", None)
    assert '- "--max-num-seqs=118"' in k8s["code"] and k8s["lang"] == "yaml"
    assert "remove" in snippet("kubernetes", "remove --enforce-eager", None)["code"]
    managed = snippet("managed", "--max-num-seqs 118", None)
    assert managed["code"] == "--max-num-seqs 118" and "engine arguments" in managed["intro"]


def test_applied_check_sees_restart_and_visible_settings() -> None:
    metrics = (ENDPOINT_FIXTURE / "metrics.prom").read_text()
    snap = fixture_snapshot()
    same = applied_check(metrics, 8192, snap, "--max-model-len 16384", None)
    assert same["visible"] and same["landed"] is False  # still 8,192
    later = metrics.replace("process_start_time_seconds", "process_start_time_seconds_old")
    later += "\nprocess_start_time_seconds 9999999999.0\n"
    restarted = applied_check(
        metrics, 8192, replace(snap, metrics_text=later), "--max-num-seqs 64", None
    )
    assert restarted["restarted"] is True
    assert not restarted["visible"]  # max_num_seqs isn't published; the run shows its effect

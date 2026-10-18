"""vLLM defaults per version range (verified against source by tests/live)."""

import pytest

from kneebench.physics.gpus import catalog, get_gpu
from kneebench.servers.vllm import batch_defaults, metric_names, version_status


@pytest.mark.parametrize(
    ("gpu_id", "seqs", "batched"),
    [
        ("h100-sxm", 1024, 8192),
        ("h100-nvl", 1024, 8192),
        ("h200", 1024, 8192),
        ("b200", 1024, 16384),
        ("mi300x", 1024, 16384),
        ("a100-sxm-80", 256, 2048),
        ("a100-40", 256, 2048),
        ("l40s", 256, 2048),
        ("l4", 256, 2048),
    ],
)
def test_batch_defaults_by_gpu(gpu_id: str, seqs: int, batched: int) -> None:
    d = batch_defaults("0.29.0", get_gpu(gpu_id))
    assert (d.max_num_seqs, d.max_num_batched_tokens) == (seqs, batched)
    assert d.verified


def test_rule_holds_for_every_catalog_gpu() -> None:
    """1024 seqs on GPUs with >= 70 GiB except A100, otherwise 256 (vLLM 0.29 to 0.32)."""
    for gpu in catalog():
        d = batch_defaults("0.31.2", gpu)
        large = gpu.memory_gib >= 70 and "a100" not in gpu.name.lower()
        assert d.max_num_seqs == (1024 if large else 256), gpu.id


def test_untested_version_falls_back_and_says_so() -> None:
    d = batch_defaults("0.40.0", get_gpu("h100-sxm"))
    assert d.max_num_seqs == 1024
    assert not d.verified
    assert not batch_defaults(None, get_gpu("l4")).verified


def test_version_status() -> None:
    assert version_status("0.29.0").in_tested_range
    assert version_status("0.32.9").in_tested_range
    assert not version_status("0.33.0").in_tested_range
    assert not version_status("not-a-version").in_tested_range
    assert not version_status(None).in_tested_range


def test_metric_name_candidates_cover_old_and_new_names() -> None:
    assert metric_names("kv_cache_usage")[0] == "vllm:kv_cache_usage_perc"
    assert "vllm:gpu_cache_usage_perc" in metric_names("kv_cache_usage")
    assert "vllm:num_preemptions" in metric_names("preemptions")

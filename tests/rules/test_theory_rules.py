"""T01-T08: theory and setup checks against physics."""

import pytest

from kneebench.analysis.engine import Status
from kneebench.detect.metrics import CacheConfig
from tests.builders import CACHE_65GIB, ctx, point, run_rule, theory_for
from tests.fixture_data import LLAMA_8B


def _cache(**changes: object) -> CacheConfig:
    return CacheConfig(**{**CACHE_65GIB.__dict__, **changes})


def test_t01_floors_always_reported() -> None:
    f = run_rule("T01", ctx())
    assert f is not None
    assert f.status is Status.INFO
    # Weights plus the cache of the average context (512 + 256/2 tokens at 128 KiB each).
    kv = 640 * 131_072
    assert f.data["baseline"]["tpot_ms"] == pytest.approx((16.06e9 + kv) / 3.9e12 * 1e3, rel=1e-3)
    assert {s["name"] for s in f.data["shapes"]} == {
        "baseline",
        "chat",
        "reasoning",
        "summarization",
    }


def test_t02_t03_report_cache_or_skip_without_metrics() -> None:
    assert run_rule("T02", ctx()).status is Status.INFO  # type: ignore[union-attr]
    t03 = run_rule("T03", ctx())
    assert t03 is not None
    chat = next(s for s in t03.data["shapes"] if s["name"] == "chat")
    assert chat["fits"] == pytest.approx(532_768 / 2048)
    blind = ctx(theory=theory_for(cache=None))
    assert run_rule("T02", blind).status is Status.SKIP  # type: ignore[union-attr]
    assert run_rule("T03", blind).status is Status.SKIP  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("fit", "status"), [(65.0, Status.OK), (1.6, Status.WARN), (0.8, Status.BAD)]
)
def test_t04_startup_fit(fit: float, status: Status) -> None:
    f = run_rule("T04", ctx(theory=theory_for(cache=_cache(kv_cache_max_concurrency=fit))))
    assert f is not None
    assert f.status is status
    assert f.data["suggest_max_model_len"] <= 532_768 / 2


def test_t04_falls_back_to_pool_over_context() -> None:
    f = run_rule("T04", ctx(theory=theory_for(cache=_cache(kv_cache_max_concurrency=None))))
    assert f is not None
    assert f.data["startup_fit"] == pytest.approx(532_768 / 8192)


def test_t05_only_when_shapes_were_adapted() -> None:
    f = run_rule("T05", ctx())
    assert f is not None
    names = {s["name"] for s in f.data["shapes"]}
    assert names == {"reasoning", "summarization"}
    assert run_rule("T05", ctx(theory=theory_for(max_model_len=131_072))) is None


def test_t06_tpot_below_floor_means_wrong_gpu() -> None:
    # 4.0 ms is below an H100 PCIe's 8 ms floor for an 8B model: the entry must be wrong.
    f = run_rule("T06", ctx(gpu="h100-pcie", points=(point(tpot_p50=4.0),)))
    assert f is not None
    assert f.status is Status.BAD
    assert f.data["implied_tbps"] > 2.0
    assert "NVIDIA H200 141GB" in f.data["candidates"]
    assert run_rule("T06", ctx(points=(point(tpot_p50=5.8),))).status is Status.OK  # type: ignore[union-attr]


def test_t06_needs_a_baseline() -> None:
    assert run_rule("T06", ctx(points=())) is None


def test_t07_implied_vram_catches_80gb_entry_for_65gib_cache() -> None:
    f = run_rule("T07", ctx(theory=theory_for(gpu="h100-sxm"), gpu="h100-sxm"))
    assert f is not None
    assert f.status is Status.BAD
    assert f.data["implied_min_gib"] > 79.6
    assert "NVIDIA H100 NVL 94GB" in f.data["candidates"]
    assert run_rule("T07", ctx()).status is Status.OK  # type: ignore[union-attr]
    assert run_rule("T07", ctx(theory=theory_for(cache=None))).status is Status.SKIP  # type: ignore[union-attr]


def test_t08_fp32_weights() -> None:
    assert run_rule("T08", ctx()) is None
    t = theory_for()
    fp32 = ctx(theory=t.__class__(**{**t.__dict__, "model_weight_dtype": "float32"}))
    assert run_rule("T08", fp32).status is Status.WARN  # type: ignore[union-attr]


def test_tensor_parallel_scales_floor() -> None:
    one = run_rule("T01", ctx(theory=theory_for(LLAMA_8B, count=1)))
    two = run_rule("T01", ctx(theory=theory_for(LLAMA_8B, count=2), gpu_count=2))
    assert one is not None and two is not None
    assert two.data["baseline"]["tpot_ms"] == pytest.approx(one.data["baseline"]["tpot_ms"] / 2)

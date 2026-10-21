"""Physics checked against the worked numbers in notes 24 to 27."""

import pytest

from kneebench.physics.floors import (
    Hardware,
    exchange_rate,
    floors,
    mbu,
    memory_budget,
    prefill_mfu,
    requests_that_fit,
    startup_fit,
)
from kneebench.physics.gpus import GIB, catalog, custom_gpu, get_gpu, match_gpu
from kneebench.physics.model import Support, UnsupportedModelError, model_spec_from_hf
from tests.fixture_data import (
    LLAMA_8B,
    LLAMA_8B_AWQ,
    LLAMA_8B_FP8,
    LLAMA_70B,
    QWEN_7B,
    hf_fixture,
    model_fixture,
)

H100 = Hardware(get_gpu("h100-sxm"))


def test_llama_8b_spec_matches_note_numbers() -> None:
    m = model_fixture(LLAMA_8B)
    assert m.params == 8_030_261_248
    assert m.weight_bytes == pytest.approx(16.06e9, rel=1e-3)
    assert m.kv_bytes_per_token() == 128 * 1024  # note 26: 128 KiB per token
    assert m.support is Support.EXACT


def test_note_24_floors_for_8b_on_h100() -> None:
    f = floors(model_fixture(LLAMA_8B), H100, isl=512, osl=256)
    assert f.tpot_s * 1e3 == pytest.approx(4.8, abs=0.05)
    assert f.ttft_s * 1e3 == pytest.approx(8.3, abs=0.1)  # note rounds 15 GFLOP/token to 7.8
    assert f.e2e_s == pytest.approx(1.23, abs=0.01)


def test_note_24_mbu_band() -> None:
    m = model_fixture(LLAMA_8B)
    assert mbu(m, H100, 6.4e-3) == pytest.approx(0.75, abs=0.01)
    assert prefill_mfu(m, H100, 512, 21e-3) == pytest.approx(0.40, abs=0.02)


def test_note_26_exchange_rate_is_about_150() -> None:
    assert exchange_rate(model_fixture(LLAMA_8B), H100) == pytest.approx(150, rel=0.05)


def test_note_26_requests_that_fit() -> None:
    assert requests_that_fit(450_000, 1000, 1000) == 225
    assert requests_that_fit(450_000, 8000, 1000) == 50


def test_note_27_startup_fit() -> None:
    assert startup_fit(105_000, 131_072) == pytest.approx(0.8, abs=0.01)


def test_fp8_checkpoint_halves_weight_bytes_and_uses_fp8_compute() -> None:
    m = model_fixture(LLAMA_8B_FP8)
    assert m.quantization == "fp8"
    assert m.fp8_compute
    assert m.quant_scheme == "W8A8-FP8"  # activations are FP8 too, so prefill uses FP8 math
    assert m.weight_bytes == pytest.approx(9.08e9, rel=0.01)
    assert m.params == pytest.approx(8.03e9, rel=0.01)
    dense = floors(model_fixture(LLAMA_8B), H100, 512, 256)
    assert floors(m, H100, 512, 256).ttft_s == pytest.approx(dense.ttft_s / 2, rel=0.01)


def test_awq_uses_stored_bytes_and_estimated_logical_params() -> None:
    m = model_fixture(LLAMA_8B_AWQ)
    assert m.quantization == "awq"
    assert not m.fp8_compute
    assert m.quant_scheme == "W4A16"  # weight-only: 16-bit math after dequantizing
    assert m.weight_bytes == pytest.approx(5.73e9, rel=0.01)
    assert m.params == pytest.approx(8.03e9, rel=0.01)


def test_gqa_qwen_kv_bytes() -> None:
    m = model_fixture(QWEN_7B)
    assert m.head_dim == 128
    assert m.kv_bytes_per_token() == 2 * 28 * 4 * 128 * 2


def test_fp8_kv_cache_halves_kv_bytes() -> None:
    m = model_fixture(LLAMA_8B)
    assert m.kv_bytes_per_token("fp8") == m.kv_bytes_per_token() // 2


def test_tensor_parallel_splits_weights_and_kv() -> None:
    m = model_fixture(LLAMA_70B)
    hw = Hardware(get_gpu("h100-sxm"), count=4)
    kv = 640 * m.kv_bytes_per_token()
    assert floors(m, hw, 512, 256).tpot_s == pytest.approx((m.weight_bytes + kv) / (4 * 3.35e12))


def test_floors_add_kv_reads_and_attention() -> None:
    from kneebench.physics.floors import attention_flops

    m = model_fixture(LLAMA_8B)
    hw = Hardware(get_gpu("h100-sxm"))
    short, long = floors(m, hw, 1024, 256), floors(m, hw, 32_768, 256)
    assert long.tpot_s > 1.2 * short.tpot_s  # a 32k context rereads ~4 GB of cache per step
    share = attention_flops(m, 32_768) / (2 * m.params * 32_768 + attention_flops(m, 32_768))
    assert 0.3 < share < 0.45  # lesson 59: attention is about 38% of prefill math at 32k
    assert attention_flops(m, 2048) / (2 * m.params * 2048) < 0.05
    assert m.kv_bytes_per_token_per_gpu(4) == m.kv_bytes_per_token() / 4
    assert m.kv_bytes_per_token_per_gpu(16) == m.kv_bytes_per_token() / 8  # 8 heads replicate


def test_implied_vram_rules_out_80gb_card_for_dev_endpoint_numbers() -> None:
    m = model_fixture(LLAMA_8B)
    budget = memory_budget(m, Hardware(get_gpu("h100-sxm")), 532_768, 0.9)
    assert budget.kv_pool_per_gpu_bytes / GIB == pytest.approx(65.03, abs=0.01)
    assert budget.implied_min_gpu_bytes > get_gpu("h100-sxm").memory_bytes
    assert budget.implied_min_gpu_bytes < get_gpu("h100-nvl").memory_bytes


def test_without_safetensors_metadata_falls_back_to_estimate() -> None:
    config, _ = hf_fixture(LLAMA_8B)
    m = model_spec_from_hf(config, None)
    assert m.support is Support.APPROXIMATE
    assert m.params == pytest.approx(8.03e9, rel=0.01)


@pytest.mark.parametrize(
    ("extra", "match"),
    [({"kv_lora_rank": 512}, "latent attention"), ({"num_local_experts": 8}, "mixture")],
)
def test_mla_and_moe_are_rejected_until_supported(extra: dict[str, int], match: str) -> None:
    config, params = hf_fixture(LLAMA_8B)
    with pytest.raises(UnsupportedModelError, match=match):
        model_spec_from_hf({**config, **extra}, params)


def test_gpu_catalog_is_complete_and_matchable() -> None:
    ids = {g.id for g in catalog()}
    assert {"h100-sxm", "h100-nvl", "h200", "a100-sxm-80", "l40s", "l4", "mi300x"} <= ids
    for g in catalog():
        assert g.bandwidth_tbps > 0
        assert g.bf16_tflops > 0
        assert g.memory_mib > 0
    nvl = match_gpu("NVIDIA H100 NVL")
    assert nvl is not None
    assert nvl.id == "h100-nvl"
    assert match_gpu("some future gpu") is None


def test_custom_gpu() -> None:
    g = custom_gpu("My accelerator", memory_gib=64, bandwidth_tbps=2.5, bf16_tflops=500)
    assert g.custom
    assert g.memory_gib == pytest.approx(64)


def test_fp8_weights_without_fp8_activations_keep_16_bit_math() -> None:
    from kneebench.physics.model import quant_scheme

    weight_only = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}},
        }
    }
    assert quant_scheme(weight_only) == ("W8A16", False)
    int8 = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "g": {
                    "weights": {"num_bits": 8, "type": "int"},
                    "input_activations": {"num_bits": 8, "type": "int"},
                }
            },
        }
    }
    assert quant_scheme(int8) == ("W8A8-INT8", False)
    assert quant_scheme({}) == (None, False)

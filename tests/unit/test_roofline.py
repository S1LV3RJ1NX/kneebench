"""Roofline and precision physics against the numbers in notes 50, 53 and 61."""

from __future__ import annotations

import pytest

from kneebench.physics.roofline import (
    FOUR_BIT_BITS,
    WHAT_IFS,
    Chip,
    Model,
    project,
    ridge_batch,
    step_s,
)

H100 = Chip(3.35e12, 989e12, 1979e12)
A100 = Chip(2.039e12, 312e12, None)
# Llama 3.1 8B in BF16; KV and attention left out where a lesson leaves them out.
LLAMA = Model(params=8_030_000_000, weight_bytes=16.06e9, kv_bytes_per_token=0, attn_width=None)
WITH_KV = Model(8_030_000_000, 16.06e9, kv_bytes_per_token=131_072, attn_width=32 * 32 * 128)


def _what(what_id: str):  # type: ignore[no-untyped-def]
    return next(w for w in WHAT_IFS if w.id == what_id)


def test_ridges_match_note_61() -> None:
    assert H100.ridge("bf16") == pytest.approx(295, abs=1)
    assert H100.ridge("fp8") == pytest.approx(590, abs=2)
    assert A100.ridge("fp8") == A100.ridge("bf16")  # no FP8 tensor cores: BF16 roof


def test_decode_intensity_is_about_the_batch() -> None:
    for b in (1, 8, 64):
        assert LLAMA.intensity(b, 0) == pytest.approx(b, rel=0.01)
    assert ridge_batch(LLAMA, H100, "bf16", 0) == pytest.approx(295, rel=0.01)


def test_cache_reads_cap_intensity_at_long_contexts() -> None:
    limit = WITH_KV.intensity_limit(2250)
    assert 50 < limit < 60
    assert WITH_KV.intensity(10_000, 2250) < limit
    assert ridge_batch(WITH_KV, H100, "bf16", 2250) is None


def test_decode_floors_halve_with_the_bits_note_50() -> None:
    assert step_s(LLAMA, H100, "bf16", 1, 0) * 1000 == pytest.approx(4.8, abs=0.05)
    fp8 = project(LLAMA, H100, "bf16", None, 1, 0, _what("fp8"))
    assert 4.8 * fp8.tpot_ratio_one == pytest.approx(2.4, abs=0.05)
    assert fp8.ttft_ratio == pytest.approx(0.5, abs=0.01)  # FP8 math is 2x


def test_four_bit_ideal_and_shrinking_gain_note_53() -> None:
    w4 = _what("w4a16")
    assert FOUR_BIT_BITS == 4.125
    one = project(LLAMA, H100, "bf16", None, 1, 0, w4)
    assert 1 / one.tpot_ratio_one == pytest.approx(3.9, abs=0.05)
    assert one.ttft_ratio == 1.0  # still multiplies in 16 bits
    big = project(LLAMA, H100, "bf16", None, 256, 0, w4)
    assert 1 / big.tpot_ratio < 1.5  # past the ridge, math is the limit


def test_freed_weight_memory_goes_to_the_cache() -> None:
    kv = project(WITH_KV, H100, "bf16", 400_000, 8, 2250, _what("fp8_kv"))
    assert kv.pool_tokens == pytest.approx(800_000)
    fp8 = project(WITH_KV, H100, "bf16", 400_000, 8, 2250, _what("fp8"))
    assert fp8.pool_tokens == pytest.approx(400_000 + 8.03e9 / 131_072, rel=0.01)

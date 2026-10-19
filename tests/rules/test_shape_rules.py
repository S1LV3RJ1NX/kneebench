"""S01-S07: prompt and output length."""

import pytest

from kneebench.analysis.engine import Status
from kneebench.analysis.shape import operating_concurrency, shape_map
from kneebench.domain import STANDARD_SHAPES, user_shapes
from kneebench.phases.theory import compute_theory
from kneebench.physics.floors import Hardware
from kneebench.physics.gpus import get_gpu
from tests.builders import CACHE_65GIB, ctx, healthy_sweep, point, run_rule, shape_sweep
from tests.fixture_data import LLAMA_8B, model_fixture


def _ctx(bend: float = 1.0, shapes: bool = True):  # type: ignore[no-untyped-def]
    return ctx(points=(point(), *healthy_sweep(), *(shape_sweep(bend) if shapes else ())))


def test_operating_point_and_map_order() -> None:
    c = _ctx()
    assert operating_concurrency(c) == 32
    names = [s.name for s in shape_map(c)]
    assert names == ["chat", "reasoning", "summarization"]  # the primary shape first


def test_s01_prompt_cost_is_the_slope() -> None:
    f = run_rule("S01", _ctx())
    assert f is not None
    assert f.status is Status.INFO
    assert f.data["per_token_us"] == pytest.approx(30, rel=0.01)
    assert f.data["bend"] == pytest.approx(1.0, rel=0.01)
    bent = run_rule("S01", _ctx(bend=1.6))
    assert bent is not None
    assert bent.status is Status.WARN
    assert run_rule("S01", _ctx(shapes=False)) is None


def test_s02_long_prompts_tax_decode() -> None:
    f = run_rule("S02", _ctx())
    assert f is not None
    assert f.status is Status.INFO
    assert f.data["concurrency"] == 32
    assert f.data["growth"] == pytest.approx((10.5 + 6.5) / (10.5 + 0.125), rel=0.01)
    assert not f.data["misses_target"]


def test_s03_e2e_slope_is_tpot() -> None:
    f = run_rule("S03", _ctx())
    assert f is not None
    assert f.data["per_token_ms"] == pytest.approx(5.8, rel=0.01)
    assert f.data["intercept_ms"] == pytest.approx(40, rel=0.05)


def test_s04_long_outputs_finish_fewer_requests() -> None:
    f = run_rule("S04", _ctx())
    assert f is not None
    assert f.status is Status.INFO
    assert f.data["drop"] == pytest.approx(5888 / 128, rel=0.1)
    assert f.data["sustain"] == 532_768 // (1024 + 5888)


def test_s05_standard_and_user_shapes() -> None:
    f = run_rule("S05", _ctx())
    assert f is not None
    assert f.status is Status.INFO
    assert f.data["primary"] == "chat"
    assert run_rule("S05", ctx()) is None  # baseline only

    mine = (*user_shapes("2000/500,20000/1000"), *STANDARD_SHAPES)
    theory = compute_theory(
        model_fixture(LLAMA_8B), Hardware(get_gpu("h100-nvl")), CACHE_65GIB, 8192, mine, "0.29.0"
    )
    trimmed = run_rule("S05", ctx(theory=theory))
    assert trimmed is not None
    assert trimmed.status is Status.WARN
    assert trimmed.data["trimmed"] == 1
    fits = compute_theory(
        model_fixture(LLAMA_8B),
        Hardware(get_gpu("h100-nvl")),
        CACHE_65GIB,
        8192,
        (*user_shapes("2000/500"), *STANDARD_SHAPES),
        "0.29.0",
    )
    assert run_rule("S05", ctx(theory=fits)) is None


def test_s06_map_flags_shapes_that_fail_at_the_same_load() -> None:
    f = run_rule("S06", _ctx())
    assert f is not None
    assert f.status is Status.WARN
    assert f.data["failing"] == ["summarization"]
    assert [r["meets"] for r in f.data["rows"]] == [True, True, False]


def test_s07_exchange_rate() -> None:
    f = run_rule("S07", _ctx())
    assert f is not None
    assert f.data["rate"] == pytest.approx(5.8 / 0.03, rel=0.02)
    assert f.data["theory_rate"] > 50


def test_user_shapes_parse_and_validate() -> None:
    shapes = user_shapes(" 2000/500 , 300/50")
    assert [(s.name, s.isl, s.osl) for s in shapes] == [("yours", 2000, 500), ("yours-2", 300, 50)]
    with pytest.raises(ValueError, match="2000/500"):
        user_shapes("2000x500")
    with pytest.raises(ValueError, match="positive"):
        user_shapes("0/10")

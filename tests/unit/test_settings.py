"""Scheduler settings inferred from engine behaviour, or the defaults when nothing pins them."""

from __future__ import annotations

from dataclasses import replace

from kneebench.analysis import settings
from kneebench.analysis.context import PointView
from tests.builders import ctx, healthy_sweep, point, sweep_point


def _steps(step: float) -> dict[str, dict[str, float]]:
    return {"vllm:iteration_tokens_total": {"p99_estimate": step}}


def test_defaults_when_nothing_pins_them() -> None:
    c = ctx(points=(point(), *healthy_sweep()))
    assert settings.max_num_seqs(c) == settings.Setting(c.theory.default_max_num_seqs, "default")
    assert settings.token_budget(c).source == "default"


def test_user_value_wins() -> None:
    c = ctx(points=(point(), *healthy_sweep()))
    c = replace(c, setup=replace(c.setup, max_num_seqs=48))
    assert settings.max_num_seqs(c) == settings.Setting(48, "set")


def test_running_plateau_gives_max_num_seqs() -> None:
    pts = (
        sweep_point(32, 2500, 8.0, seq=1, waiting_max=0.0),
        sweep_point(64, 3000, 9.0, seq=2, running_max=48, waiting_max=16.0),
        sweep_point(128, 3050, 9.5, seq=3, running_max=48, waiting_max=80.0),
    )
    assert settings.max_num_seqs(ctx(points=(point(), *pts))) == settings.Setting(48, "inferred")
    # A full cache explains the same plateau, so nothing is inferred from it.
    full = tuple(replace(p, server={**p.server, "vllm:kv_cache_usage_perc": {"max": 0.97}})
                 for p in pts)  # fmt: skip
    assert settings.max_num_seqs(ctx(points=(point(), *full))).source == "default"


def test_token_budget_from_steps() -> None:
    def level(c: int, seq: int, step: float, waiting: float) -> PointView:
        p = sweep_point(c, 2000, 9.0, seq=seq, waiting_max=waiting)
        return replace(p, server={**p.server, **_steps(step)})

    c = ctx(points=(point(),))
    default = c.theory.default_max_num_batched_tokens
    bigger = ctx(points=(point(), level(64, 1, default * 2.0, 0.0)))
    assert settings.token_budget(bigger) == settings.Setting(round(default * 2.0), "inferred")
    capped = ctx(points=(point(), level(64, 1, 2048, 10.0), level(128, 2, 2050, 40.0)))
    assert settings.token_budget(capped) == settings.Setting(2050, "inferred")
    one = ctx(points=(point(), level(64, 1, 2048, 10.0)))
    assert settings.token_budget(one).source == "default"  # one level proves nothing

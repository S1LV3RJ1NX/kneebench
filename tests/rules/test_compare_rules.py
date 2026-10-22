"""X01-X02: comparing two runs."""

from dataclasses import replace

import pytest

from kneebench.analysis.compare import compare
from kneebench.analysis.context import RunContext
from kneebench.analysis.engine import Finding, Status
from kneebench.content.render import TermTracker, make_env, render_finding
from kneebench.domain import Slo
from kneebench.report.composer import common_vars
from tests.builders import ctx, healthy_sweep, point, sweep_point


def _before() -> RunContext:
    return ctx(points=(point(), *healthy_sweep()))


def _after(tpot_p99_at_64: float = 40.0, tpot_p50: float = 5.8) -> RunContext:
    sweep = (*healthy_sweep()[:-1], sweep_point(64, 3400, 15.0, tpot_p99_at_64, seq=7))
    c = ctx(points=(point(tpot_p50=tpot_p50), *sweep))
    return replace(c, run_id=2, setup=replace(c.setup, max_num_seqs=128))


def _rule(rule_id: str, a: RunContext, b: RunContext) -> Finding:
    _, findings = compare(a, b)
    found = next(f for f in findings if f.rule_id == rule_id)
    rendered = render_finding(found, common_vars(b), make_env(TermTracker()))
    assert rendered.status_line
    return found


def test_x01_one_change_is_a_clean_comparison() -> None:
    f = _rule("X01", _before(), _after())
    assert f.status is Status.OK
    assert f.data["change"]["label"] == "max_num_seqs"
    assert _rule("X02", _before(), _after()).data["comparable"]
    assert f.data["change"]["a"] == "default"
    assert f.data["change"]["b"] == 128


def test_x01_same_setup_and_different_tests() -> None:
    assert _rule("X01", _before(), _before()).status is Status.INFO
    other = _before()
    other = replace(other, workload=replace(other.workload, slo=Slo(300, 30)))
    f = _rule("X01", _before(), other)
    assert f.status is Status.WARN
    assert [w["label"] for w in f.data["workload"]] == ["latency targets"]
    unfair = _rule("X02", _before(), other)
    assert unfair.status is Status.INFO
    assert not unfair.data["comparable"]
    several = replace(_after(), vllm_version="0.30.0")
    f = _rule("X01", _before(), several)
    assert f.status is Status.WARN
    assert f.data["n_setup"] == 2


def test_x02_reports_ratios_and_noise() -> None:
    f = _rule("X02", _before(), _after())
    assert f.status is Status.OK
    assert f.data["headline"]["key"] == "capacity_users"
    assert f.data["headline"]["ratio"] == pytest.approx(2.0)
    assert f.data["grid_note"]
    same = _rule("X02", _before(), _before())
    assert same.status is Status.INFO
    worse = _rule("X02", _before(), _after(tpot_p50=7.5))
    assert worse.status is Status.WARN
    assert "Time per output token, one user" in worse.data["worse"]


def test_x02_noise_band_comes_from_the_repeated_baseline() -> None:
    steady = _rule("X02", _before(), _after())
    assert steady.data["noise_pct"] == 5.0 and not steady.data["noise_measured"]
    noisy = _before()
    noisy = replace(
        noisy, points=(*noisy.points, point(phase="baseline_end", seq=20, tpot_p50=7.25))
    )  # the baseline moved 25% between the start and the end of the run
    f = _rule("X02", noisy, _after())
    assert f.data["noise_measured"] and f.data["noise_pct"] == pytest.approx(25.0)
    rows = {r["key"]: r for r in f.data["rows"]}
    assert rows["peak_tok_s"]["verdict"] == "same"  # +10% is inside a 25% band

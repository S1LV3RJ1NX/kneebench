"""B01-B06: the single-request baseline (note 24). Floor on H100 NVL for 8B is ~4.12 ms."""

import pytest

from kneebench.analysis.engine import Status
from tests.builders import ctx, point, run_rule

FLOOR_MS = 16.06e9 / 3.9e12 * 1e3


@pytest.mark.parametrize(
    ("tpot", "status"),
    [
        (FLOOR_MS / 0.75, Status.OK),
        (FLOOR_MS / 0.5, Status.WARN),
        (FLOOR_MS / 0.3, Status.BAD),
        (FLOOR_MS * 0.9, Status.SKIP),
    ],
)
def test_b01_mbu_band(tpot: float, status: Status) -> None:
    f = run_rule("B01", ctx(points=(point(tpot_p50=tpot),)))
    assert f is not None
    assert f.status is status


def test_b01_no_baseline() -> None:
    assert run_rule("B01", ctx(points=())) is None
    assert run_rule("B01", ctx(points=(point(status="failed"),))) is None


@pytest.mark.parametrize("eager", [True, None, False])
def test_b02_suspects_when_far_from_floor(eager: bool | None) -> None:
    f = run_rule("B02", ctx(points=(point(tpot_p50=FLOOR_MS * 4),), enforce_eager=eager))
    assert f is not None
    assert f.status is Status.BAD
    assert f.data["enforce_eager"] is eager
    assert run_rule("B02", ctx(points=(point(tpot_p50=FLOOR_MS * 1.3),))) is None


def test_b03_ttft_normal_odd_and_unmeasurable() -> None:
    normal = run_rule("B03", ctx(points=(point(ttft_p50=30.0),)))
    assert normal is not None
    assert normal.status is Status.OK
    odd = run_rule("B03", ctx(points=(point(ttft_p50=400.0),)))
    assert odd is not None
    assert odd.status is Status.WARN
    far = ctx(points=(point(ttft_p50=280.0),), network={"http_rtt_ms": 272.0})
    skipped = run_rule("B03", far)
    assert skipped is not None
    assert skipped.status is Status.SKIP


def test_b03_odd_ttft_not_flagged_when_decode_is_also_slow() -> None:
    f = run_rule("B03", ctx(points=(point(ttft_p50=400.0, tpot_p50=FLOOR_MS * 4),)))
    assert f is not None
    assert f.status is Status.OK


@pytest.mark.parametrize(
    ("std", "p99", "status"),
    [(0.1, 6.0, Status.OK), (1.0, 6.0, Status.WARN), (0.1, 9.5, Status.WARN)],
)
def test_b04_spread(std: float, p99: float, status: Status) -> None:
    f = run_rule("B04", ctx(points=(point(tpot_p50=5.8, tpot_std=std, tpot_p99=p99),)))
    assert f is not None
    assert f.status is status


def test_b05_prefill_mfu_only_when_measurable() -> None:
    f = run_rule("B05", ctx(points=(point(ttft_p50=21.0),), network={"http_rtt_ms": 1.0}))
    assert f is not None
    assert 0 < f.data["mfu"] < 1
    assert (
        run_rule("B05", ctx(points=(point(ttft_p50=280.0),), network={"http_rtt_ms": 272.0}))
        is None
    )


def test_b06_drift_between_start_and_end() -> None:
    start = point(tpot_p50=5.8)
    same = point(phase="baseline_end", seq=5, tpot_p50=6.0)
    drifted = point(phase="baseline_end", seq=5, tpot_p50=7.5)
    assert run_rule("B06", ctx(points=(start,))) is None
    assert run_rule("B06", ctx(points=(start, same))).status is Status.OK  # type: ignore[union-attr]
    assert run_rule("B06", ctx(points=(start, drifted))).status is Status.WARN  # type: ignore[union-attr]

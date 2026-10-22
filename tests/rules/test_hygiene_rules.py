"""H03-H08: is the test itself trustworthy?"""

from dataclasses import replace
from typing import Any

import pytest

from kneebench.analysis.engine import Status
from tests.builders import ctx, point, run_rule


@pytest.mark.parametrize(
    ("rtt", "ttft", "status", "unmeasurable"),
    [
        (5.0, 100.0, Status.OK, False),
        (40.0, 100.0, Status.WARN, False),
        (60.0, 100.0, Status.BAD, True),
        (272.0, 269.0, Status.BAD, True),
    ],
)
def test_h03_client_distance(rtt: float, ttft: float, status: Status, unmeasurable: bool) -> None:
    f = run_rule(
        "H03",
        ctx(
            points=(point(ttft_p50=ttft),),
            network={"http_rtt_ms": rtt, "tcp_connect_ms": rtt, "tls_handshake_ms": rtt},
        ),
    )
    assert f is not None
    assert f.status is status
    assert f.data["unmeasurable"] is unmeasurable
    assert f.data["dominates"] is (rtt / ttft >= 0.9)
    if unmeasurable:
        assert f.data["server_ttft_ms"] is None


def test_h03_without_probe_is_skipped() -> None:
    f = run_rule("H03", ctx(network={}))
    assert f is not None
    assert f.status is Status.SKIP


def test_h04_short_outputs_from_length_or_aiperf_warning() -> None:
    assert run_rule("H04", ctx()).status is Status.OK  # type: ignore[union-attr]
    short = run_rule("H04", ctx(points=(point(osl=8192, osl_mean=300.0),)))
    assert short is not None
    assert short.status is Status.BAD
    assert short.data["points"][0]["requested"] == 8192
    warned = run_rule(
        "H04",
        ctx(
            points=(
                point(
                    warnings=("Output Sequence Length Mismatch Warning",),
                    extra={"osl_mismatch_count": {"avg": 10.0}},
                ),
            )
        ),
    )
    assert warned is not None
    assert warned.status is Status.BAD
    # One short answer out of 256 triggers AIPerf's warning but doesn't change the numbers.
    one_off = run_rule(
        "H04",
        ctx(
            points=(
                point(
                    requests=256,
                    osl_mean=1023.0,
                    osl=1024,
                    warnings=("Output Sequence Length Mismatch Warning",),
                    extra={"osl_mismatch_count": {"avg": 1.0}},
                ),
            )
        ),
    )
    assert one_off is not None
    assert one_off.status is Status.OK


def test_h05_token_discrepancy() -> None:
    assert run_rule("H05", ctx()) is None
    f = run_rule("H05", ctx(points=(point(warnings=("Token Count Discrepancy Warning",)),)))
    assert f is not None
    assert f.status is Status.WARN


@pytest.mark.parametrize(
    ("version", "status"),
    [
        ("0.29.0", Status.OK),
        ("0.32.4", Status.OK),
        ("0.33.0", Status.WARN),
        ("0.10.1", Status.WARN),
        (None, Status.WARN),
    ],
)
def test_h07_version_range(version: str | None, status: Status) -> None:
    f = run_rule("H07", ctx(vllm_version=version))
    assert f is not None
    assert f.status is status


def test_h08_metrics_visibility() -> None:
    assert run_rule("H08", ctx()) is None
    assert run_rule("H08", ctx(metrics_available=False)).status is Status.WARN  # type: ignore[union-attr]


def test_h04_one_short_answer_in_a_small_point_is_not_a_shortfall() -> None:
    one_of_16 = point(requests=16, extra={"osl_mismatch_count": {"avg": 1.0}})
    assert run_rule("H04", ctx(points=(one_of_16,))).status is Status.OK  # type: ignore[union-attr]
    two_of_16 = point(requests=16, extra={"osl_mismatch_count": {"avg": 2.0}})
    assert run_rule("H04", ctx(points=(two_of_16,))).status is Status.BAD  # type: ignore[union-attr]


def test_h04_trusts_the_server_count_over_aiperf_recount() -> None:
    recounted = point(
        osl=6884,
        osl_mean=6741,
        extra={"usage_completion_tokens": {"avg": 6884.0}, "osl_mismatch_count": {"avg": 30.0}},
    )
    f = run_rule("H04", ctx(points=(recounted,)))
    assert f is not None and f.status is Status.OK


def test_h05_ignores_multi_turn_template_tokens() -> None:
    def pt(phase: str, isl: float, server: float, seq: int, **spec: Any) -> Any:
        p = point(
            phase=phase,
            seq=seq,
            extra={"input_sequence_length": {"avg": isl}, "usage_prompt_tokens": {"avg": server}},
        )
        return replace(p, spec=spec)

    pts = (
        pt("baseline", 512, 547, 0),
        pt("concurrency", 2000, 2035, 1),
        pt("reuse_turns", 2036, 2092, 2, turns=6),
    )
    f = run_rule("H05", ctx(points=pts))
    assert f is not None and f.status is Status.OK and f.data["overhead"] == 35


def test_h10_other_traffic_at_start_or_during_single_user_points() -> None:
    alone = ctx(points=(point(),))
    idle = run_rule("H10", replace(alone, activity_at_start={"running": 0, "waiting": 0}))
    assert idle is not None and idle.status is Status.OK
    busy = run_rule("H10", replace(alone, activity_at_start={"running": 3, "waiting": 1}))
    assert busy is not None and busy.status is Status.WARN and busy.data["busy_at_start"]
    shared = point()
    shared = replace(shared, server={"vllm:num_requests_running": {"max": 4.0}})
    during = run_rule(
        "H10", replace(ctx(points=(shared,)), activity_at_start={"running": 0, "waiting": 0})
    )
    assert during is not None and during.status is Status.WARN
    assert during.data["single"] == [{"phase": "baseline", "running": 4.0}]

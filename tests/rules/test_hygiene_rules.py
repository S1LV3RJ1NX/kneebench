"""H03-H08: is the test itself trustworthy?"""

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

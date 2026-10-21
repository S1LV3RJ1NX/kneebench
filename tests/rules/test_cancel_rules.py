"""A01: does the server stop generating when the client hangs up?"""

from dataclasses import replace
from typing import Any

from kneebench.analysis.engine import Status
from tests.builders import ctx, healthy_sweep, point, run_rule


def _probe(generated: float, aborted: float, length: float = 0.0, **extra: Any) -> Any:
    probe = {
        "before": {"generated": 1_000.0, "abort": 2.0, "length": 50.0, "stop": 0.0},
        "after": {
            "generated": 1_000.0 + generated,
            "abort": 2.0 + aborted,
            "length": 50.0 + length,
            "stop": 0.0,
        },
        "requests": 16,
        "osl": 1000,
        "cancel_after_s": 1.0,
        "linger_s": 12.0,
        **extra,
    }
    p = replace(point(phase="cancel", isl=500, osl=1000, concurrency=8, seq=60), series={})
    return ctx(points=(point(), *healthy_sweep(), replace(p, series={"cancel": probe})))


def test_a01_server_that_aborts_passes() -> None:
    f = run_rule("A01", _probe(generated=16 * 120, aborted=16))
    assert f is not None and f.status is Status.OK
    assert f.data["per_request"] == 120


def test_a01_server_that_keeps_generating_warns() -> None:
    f = run_rule("A01", _probe(generated=16 * 1000, aborted=0, length=16))
    assert f is not None and f.status is Status.WARN
    assert f.data["wasted"] == 16_000


def test_a01_other_traffic_is_not_judged() -> None:
    f = run_rule("A01", _probe(generated=16 * 1000, aborted=0, length=40))
    assert f is not None and f.status is Status.SKIP and f.data["reason"] == "other_traffic"
    half = run_rule("A01", _probe(generated=16 * 650, aborted=0, length=16))
    assert half is not None and half.data["reason"] == "unclear"

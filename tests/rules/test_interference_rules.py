"""I01: long prompts freezing other users' streaming answers."""

from dataclasses import replace

from kneebench.analysis.engine import Status
from kneebench.loadgen.aiperf.parse import RequestRecord, stream_summary
from tests.builders import ctx, healthy_sweep, point, run_rule

NS = 1_000_000_000


def _summary(spike_ms: float | None) -> dict:  # type: ignore[type-arg]
    gaps = [10.0] * 4000
    records = []
    for _ in range(8):
        g = list(gaps)
        if spike_ms:
            for at in (500, 1500, 2500, 3500):  # 5, 15, 25 and 35 s in
                g[at - 1] = spike_ms
        records.append(
            RequestRecord(
                "profiling",
                0,
                int(41 * NS),
                50.0,
                10.0,
                41_000.0,
                128,
                4001,
                chunk_gaps_ms=tuple(g),
            )
        )
    if spike_ms:
        records += [
            RequestRecord(
                "profiling", int(at * NS), int((at + 0.3) * NS), 160.0, None, 300.0, 8000, 16
            )
            for at in (5.0, 15.0, 25.0, 35.0)
        ]
    s = stream_summary(records)
    assert s is not None
    return s


def _ctx(spike_ms: float | None):  # type: ignore[no-untyped-def]
    alone = replace(point(phase="interference_base", seq=40), series={"streams": _summary(None)})
    mixed = replace(point(phase="interference", seq=41), series={"streams": _summary(spike_ms)})
    return ctx(points=(point(), *healthy_sweep(), alone, mixed))


def test_i01_flags_stalls_past_the_target_and_suggests_a_budget() -> None:
    f = run_rule("I01", _ctx(150.0))
    assert f is not None
    assert f.status is Status.WARN
    assert f.data["aligned"] == 4
    assert f.data["mixed"]["max_ms"] == 150.0
    assert f.data["can_lower"]
    assert f.data["safe_budget"] % 128 == 0 and 512 <= f.data["safe_budget"] < f.data["budget"]


def test_i01_ok_when_pauses_stay_within_the_target() -> None:
    f = run_rule("I01", _ctx(40.0))
    assert f is not None
    assert f.status is Status.OK
    assert run_rule("I01", ctx(points=(point(),))) is None


def test_i01_says_it_cant_tell_when_the_streams_are_noisy_alone() -> None:
    noisy = _summary(None)
    noisy = {**noisy, "p99_ms": 134.0, "p999_ms": 396.0, "max_ms": 1110.0}
    c = _ctx(150.0)
    alone = next(p for p in c.points if p.phase == "interference_base")
    c = replace(
        c,
        points=tuple(replace(p, series={"streams": noisy}) if p is alone else p for p in c.points),
    )
    f = run_rule("I01", c)
    assert f is not None
    assert f.status is Status.INFO and f.data["noisy"]

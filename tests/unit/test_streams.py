"""What streaming answers feel while long prompts arrive (lesson 44)."""

from kneebench.loadgen.aiperf.parse import RequestRecord, stream_summary

NS = 1_000_000_000


def _stream(start_s: float, gaps_ms: list[float]) -> RequestRecord:
    end = start_s + 0.05 + sum(gaps_ms) / 1000
    return RequestRecord(
        phase="profiling",
        start_ns=int(start_s * NS),
        end_ns=int(end * NS),
        ttft_ms=50.0,
        tpot_ms=10.0,
        e2e_ms=(end - start_s) * 1000,
        isl=128,
        osl=len(gaps_ms) + 1,
        chunk_gaps_ms=tuple(gaps_ms),
    )


def test_stalls_line_up_with_injected_prompts() -> None:
    gaps = [10.0] * 1000
    gaps[499] = 150.0  # the step that read the long prompt, ~5 s in
    streams = [_stream(0.0, list(gaps)) for _ in range(4)]
    injected = RequestRecord(
        phase="profiling",
        start_ns=int(5.0 * NS),
        end_ns=int(5.3 * NS),
        ttft_ms=160.0,
        tpot_ms=None,
        e2e_ms=300.0,
        isl=8000,
        osl=16,
    )
    s = stream_summary([*streams, injected])
    assert s is not None
    assert s["streams"] == 4
    assert s["p50_ms"] == 10.0
    assert s["max_ms"] == 150.0
    assert s["stalls"] == 4  # one per stream, all from the same step
    assert s["injected"] == [{"at_s": 5.0, "isl": 8000, "ttft_ms": 160.0, "ok": True}]
    spike_bins = [t for t, g in s["series"] if g > s["stall_ms"]]
    assert spike_bins and 4.5 <= spike_bins[0] <= 5.5


def test_bundled_chunks_dont_count_as_gaps() -> None:
    gaps = [0.0, 20.0] * 300  # a gateway sending two tokens per chunk
    s = stream_summary([_stream(0.0, gaps)])
    assert s is not None
    assert s["p50_ms"] == 20.0 and s["stalls"] == 0

"""First-token time from far away: the lower of the client's and the server's estimate."""

from __future__ import annotations

from dataclasses import replace

from kneebench.analysis.context import engine_ttft_wins
from tests.builders import point


def _engine(p50_ms: float, p99_ms: float) -> dict[str, dict[str, float]]:
    return {"vllm:time_to_first_token_seconds": {"p50_estimate": p50_ms / 1000,
                                                 "p99_estimate": p99_ms / 1000}}  # fmt: skip


def test_engine_estimate_only_when_the_network_dominates() -> None:
    base = replace(point(ttft_p50=311.0), server=_engine(25.0, 40.0))
    assert engine_ttft_wins(279.0, base)  # the round trip is most of the first token
    assert not engine_ttft_wins(5.0, base)  # close to the server: the client is accurate
    assert not engine_ttft_wins(279.0, point(ttft_p50=311.0))  # no histogram to use


def test_lower_of_client_and_engine_per_percentile() -> None:
    # Network jitter put the client's tail at 1.5 s; the server's own P99 was 300 ms.
    p = replace(point(ttft_p50=400.0), server=_engine(90.0, 300.0))
    client = p.server_ttft(280.0)
    mixed = replace(p, ttft_engine=True).server_ttft(280.0)
    assert mixed["p99"] == min(client["p99"], 300.0)
    assert mixed["p50"] == min(client["p50"], 90.0)
    assert p.server_ttft(280.0) == client  # unflagged points keep the client's numbers

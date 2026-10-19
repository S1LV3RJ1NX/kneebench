"""First-token times without new-connection setup (a far client pays DNS, TCP and TLS)."""

from dataclasses import replace

import pytest

from kneebench.loadgen.aiperf.parse import parse_point
from tests.builders import BASELINE_FIXTURE, rate_point


def test_connection_setup_is_removed_from_ttft() -> None:
    p = rate_point(1.0, ttft_ms=100.0)
    slow = [
        replace(r, ttft_ms=(r.ttft_ms or 0) + 550, connect_ms=550.0) if i % 10 == 0 else r
        for i, r in enumerate(p.requests)
    ]
    with_setup = replace(p, requests=tuple(slow))
    assert with_setup.new_connections
    assert with_setup.server_ttft(20.0)["p99"] == pytest.approx(80.0)
    # Without connection data, the aggregate percentiles are used as before.
    assert p.server_ttft(20.0)["p50"] == pytest.approx(80.0)


def test_parse_reads_connection_setup() -> None:
    reqs = parse_point(BASELINE_FIXTURE).requests
    assert all(r.connect_ms is not None for r in reqs)

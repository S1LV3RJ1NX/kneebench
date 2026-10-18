import json
from pathlib import Path

from kneebench.loadgen.aiperf.overhead import (
    DEFAULT_STARTUP_OVERHEAD_S,
    FIRST_LAUNCH_EXTRA_S,
    estimate_point_overhead_s,
    point_overhead_s,
)


def test_estimate_uses_default_then_observed_median() -> None:
    assert estimate_point_overhead_s([]) == DEFAULT_STARTUP_OVERHEAD_S
    assert estimate_point_overhead_s([], first_launch=True) == (
        DEFAULT_STARTUP_OVERHEAD_S + FIRST_LAUNCH_EXTRA_S
    )
    assert estimate_point_overhead_s([3.0, 9.0, 4.0], first_launch=True) == 4.0


def test_point_overhead_from_export(tmp_path: Path) -> None:
    (tmp_path / "profile_export_aiperf.json").write_text(
        json.dumps({"benchmark_duration": {"unit": "sec", "avg": 1.5}})
    )
    assert point_overhead_s(9.5, tmp_path) == 8.0
    assert point_overhead_s(9.5, tmp_path / "missing") is None

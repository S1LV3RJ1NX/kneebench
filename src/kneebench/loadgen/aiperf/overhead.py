"""AIPerf per-process startup and shutdown overhead, used in the planner's ETA.

kneebench launches one AIPerf process per sweep point, and each launch pays a fixed
cost (interpreter, service processes, tokenizer load, export) on top of the benchmark
itself. A 20-point sweep pays it 20 times, so the ETA must include it.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kneebench.loadgen.aiperf.runner import AiperfLoadGenerator
from kneebench.loadgen.base import Endpoint, LoadMode, PointSpec

# Measured on macOS arm64, Python 3.12, aiperf 0.13.0, local mock server, 5 warm runs each
# (re-measure with scripts/measure_aiperf_startup.py):
#   builtin tokenizer: median 2.75 s; Llama 3.1 HF tokenizer (cached): median 7.82 s.
# The first launch after install is ~20 s while Python compiles AIPerf's modules.
# The planner starts from these and switches to the median observed in the current run.
DEFAULT_STARTUP_OVERHEAD_S = 8.0
FIRST_LAUNCH_EXTRA_S = 12.0


@dataclass(frozen=True)
class OverheadSample:
    wall_s: float
    benchmark_s: float

    @property
    def overhead_s(self) -> float:
        return max(self.wall_s - self.benchmark_s, 0.0)


def benchmark_duration_s(artifact_dir: Path) -> float | None:
    path = artifact_dir / "profile_export_aiperf.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    value = data.get("benchmark_duration", {}).get("avg")
    return float(value) if value is not None else None


def point_overhead_s(wall_s: float, artifact_dir: Path) -> float | None:
    bench = benchmark_duration_s(artifact_dir)
    return None if bench is None else max(wall_s - bench, 0.0)


def measure_startup_overhead(endpoint: Endpoint, runs: int = 5) -> list[OverheadSample]:
    gen = AiperfLoadGenerator()
    spec = PointSpec(
        isl=32,
        osl=8,
        mode=LoadMode.CONCURRENCY,
        concurrency=1,
        request_count=2,
        slice_duration_s=None,
    )
    samples: list[OverheadSample] = []
    with tempfile.TemporaryDirectory(prefix="kneebench-overhead-") as tmp:
        for i in range(runs):
            result = gen.run_point(endpoint, spec, Path(tmp) / f"run{i}")
            bench = benchmark_duration_s(result.artifact_dir)
            if not result.ok or bench is None:
                raise RuntimeError(f"overhead run {i} failed: {result.stderr_tail[-500:]}")
            samples.append(OverheadSample(wall_s=result.wall_time_s, benchmark_s=bench))
    return samples


def median_overhead(samples: list[OverheadSample]) -> float:
    return statistics.median(s.overhead_s for s in samples)


def estimate_point_overhead_s(observed: list[float], first_launch: bool = False) -> float:
    """Per-point overhead for the ETA: the run's own median once it has data, else the default."""
    base = statistics.median(observed) if observed else DEFAULT_STARTUP_OVERHEAD_S
    return base + (FIRST_LAUNCH_EXTRA_S if first_launch and not observed else 0.0)

"""The interface every load generator implements.

A sweep is a list of points. kneebench runs each point as its own load-generator
invocation so it can stop at a wall, budget time per point and resume after a crash.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class LoadMode(StrEnum):
    CONCURRENCY = "concurrency"
    RATE = "rate"
    SCHEDULE = "schedule"  # every request at a planned time, from `schedule`


class Arrival(StrEnum):
    POISSON = "poisson"
    GAMMA = "gamma"
    CONSTANT = "constant"


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    tokenizer: str
    api_key: str | None = field(default=None, repr=False)
    endpoint_type: str = "chat"
    extra_headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PointSpec:
    """One benchmark point: a fixed shape at a fixed load."""

    isl: int
    osl: int
    mode: LoadMode
    concurrency: int | None = None
    request_rate: float | None = None
    arrival: Arrival = Arrival.POISSON
    burstiness: float | None = None
    isl_stddev: float = 0.0
    osl_stddev: int = 0
    request_count: int | None = None
    warmup_count: int | None = None
    # Rate points warm up for this long at their own rate, so profiling starts at load.
    warmup_s: float | None = None
    duration_s: float | None = None
    slice_duration_s: float | None = 10.0
    goodput: tuple[tuple[str, float], ...] = ()
    random_seed: int = 42
    # (send at ms, prompt tokens, output tokens) per request, for LoadMode.SCHEDULE.
    schedule: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.mode is LoadMode.SCHEDULE and not self.schedule:
            raise ValueError("schedule mode needs a schedule")
        if self.mode is LoadMode.CONCURRENCY and not self.concurrency:
            raise ValueError("concurrency mode needs a concurrency level")
        if self.mode is LoadMode.RATE and not self.request_rate:
            # An unset rate means every request is sent at t=0 (note 25): never allow it.
            raise ValueError("rate mode needs an explicit request rate")
        if self.isl <= 0 or self.osl <= 0:
            raise ValueError("isl and osl must be positive")


@dataclass(frozen=True)
class PointResult:
    spec: PointSpec
    artifact_dir: Path
    exit_code: int
    wall_time_s: float
    argv_redacted: tuple[str, ...]
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class LoadGenerator(Protocol):
    name: str

    def version(self) -> str: ...

    def run_point(
        self,
        endpoint: Endpoint,
        spec: PointSpec,
        artifact_dir: Path,
        timeout_s: float | None = None,
        cancel: threading.Event | None = None,
    ) -> PointResult: ...

"""Runs benchmarks in a background thread, one at a time.

Two benchmarks against the same endpoint at once would measure each other, so only one
run is active per kneebench process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from kneebench.loadgen.aiperf.runner import AiperfLoadGenerator
from kneebench.loadgen.base import LoadGenerator
from kneebench.phases.plan import RunPlan
from kneebench.phases.runner import PointEvent, RunInputs, execute_run
from kneebench.store.db import DB_FILENAME, connect


class BusyError(RuntimeError):
    pass


@dataclass
class ActiveRun:
    run_id: int | None = None
    total: int = 0
    done: int = 0
    current: str | None = None
    stopped: str | None = None
    error: str | None = None
    finished: bool = False
    created: threading.Event = field(default_factory=threading.Event)


class RunManager:
    def __init__(self, home: Path, loadgen: LoadGenerator | None = None) -> None:
        self.home = home
        self.loadgen = loadgen or AiperfLoadGenerator()
        self._lock = threading.Lock()
        self.active: ActiveRun | None = None

    @property
    def busy(self) -> bool:
        return self.active is not None and not self.active.finished

    def start(self, inputs: RunInputs, plan: RunPlan, timeout_s: float = 60.0) -> int:
        with self._lock:
            if self.busy:
                raise BusyError("another benchmark is already running")
            state = ActiveRun(total=len(plan.points))
            self.active = state

        def on_created(run_id: int) -> None:
            state.run_id = run_id
            state.created.set()

        def on_event(ev: PointEvent) -> None:
            spec = ev.planned.spec
            if ev.skipped:
                state.stopped = ev.skipped
                state.done += 1
            elif ev.result is None:
                state.current = f"{ev.planned.phase}, {spec.concurrency} users in flight"
            else:
                state.done += 1

        def work() -> None:
            conn = connect(self.home / DB_FILENAME)
            try:
                execute_run(
                    conn,
                    inputs,
                    plan,
                    self.loadgen,
                    self.home,
                    on_event=on_event,
                    on_created=on_created,
                )
            except Exception as exc:  # surfaced on the progress page
                state.error = f"{type(exc).__name__}: {exc}"
            finally:
                state.finished = True
                state.created.set()
                conn.close()

        threading.Thread(target=work, name="kneebench-run", daemon=True).start()
        if not state.created.wait(timeout_s) or state.run_id is None:
            raise RuntimeError(state.error or "the run did not start")
        return state.run_id

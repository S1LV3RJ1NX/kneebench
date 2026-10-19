"""Runs benchmarks in a background thread, one at a time (shared lock with the CLI)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from kneebench.loadgen.aiperf.runner import AiperfLoadGenerator
from kneebench.loadgen.base import LoadGenerator
from kneebench.phases.plan import RunPlan
from kneebench.phases.runner import PointEvent, RunInputs, execute_run
from kneebench.runlock import RunLockedError, record_run_id, run_lock
from kneebench.store.db import DB_FILENAME, connect


class BusyError(RuntimeError):
    pass


@dataclass
class ActiveRun:
    plan: RunPlan
    run_id: int | None = None
    done: int = 0
    current: str | None = None
    current_started: float | None = None
    stopped: str | None = None
    error: str | None = None
    finished: bool = False
    created: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def total(self) -> int:
        return len(self.plan.points)

    @property
    def remaining_s(self) -> float:
        left = sum(p.eta_s for p in self.plan.points[self.done :])
        if self.current_started is not None and self.done < self.total:
            elapsed = time.monotonic() - self.current_started
            left -= min(elapsed, self.plan.points[self.done].eta_s)
        return max(left, 0.0)


class RunManager:
    def __init__(self, home: Path, loadgen: LoadGenerator | None = None) -> None:
        self.home = home
        self.loadgen = loadgen or AiperfLoadGenerator()
        self._lock = threading.Lock()
        self.active: ActiveRun | None = None

    @property
    def busy(self) -> bool:
        return self.active is not None and not self.active.finished

    def state_for(self, run_id: int) -> ActiveRun | None:
        return self.active if self.active and self.active.run_id == run_id else None

    def stop(self, run_id: int) -> bool:
        state = self.state_for(run_id)
        if state is None or state.finished:
            return False
        state.cancel.set()
        return True

    def start(self, inputs: RunInputs, plan: RunPlan, timeout_s: float = 60.0) -> int:
        with self._lock:
            if self.busy:
                raise BusyError("another benchmark is already running in this window")
            state = ActiveRun(plan=plan)
            self.active = state
        started = threading.Event()
        lock_error: list[Exception] = []

        def on_created(run_id: int) -> None:
            state.run_id = run_id
            record_run_id(self.home, run_id)
            state.created.set()

        def on_event(ev: PointEvent) -> None:
            if ev.skipped:
                state.stopped = ev.skipped
                state.done += 1
            elif ev.result is None:
                state.current = f"{ev.planned.phase}, {ev.planned.spec.concurrency} in flight"
                state.current_started = time.monotonic()
            else:
                state.done += 1
                state.current_started = None

        def work() -> None:
            try:
                with run_lock(self.home, source="web UI"):
                    started.set()
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
                            cancel=state.cancel,
                        )
                    finally:
                        conn.close()
            except RunLockedError as exc:
                lock_error.append(exc)
                state.error = str(exc)
            except Exception as exc:  # shown on the progress page
                state.error = f"{type(exc).__name__}: {exc}"
            finally:
                state.finished = True
                started.set()
                state.created.set()

        threading.Thread(target=work, name="kneebench-run", daemon=True).start()
        started.wait(timeout_s)
        if lock_error:
            raise BusyError(str(lock_error[0]))
        if not state.created.wait(timeout_s) or state.run_id is None:
            raise RuntimeError(state.error or "the run did not start")
        return state.run_id

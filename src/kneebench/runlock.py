"""One benchmark at a time per data directory, across the CLI and the web UI.

Two benchmarks against the same endpoint at once would measure each other. The lock is a
file holding the owner's pid; a lock whose process has exited is taken over.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOCK_FILENAME = "active-run.lock"


class RunLockedError(RuntimeError):
    def __init__(self, holder: dict[str, Any]) -> None:
        self.holder = holder
        where = f"run {holder['run_id']}" if holder.get("run_id") else "a run"
        super().__init__(
            f"{where} is already in progress (process {holder.get('pid')}, started from "
            f"{holder.get('source', 'kneebench')}). Wait for it or stop it first."
        )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def holder(home: Path) -> dict[str, Any] | None:
    path = home / LOCK_FILENAME
    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    if not _alive(int(data.get("pid", -1))):
        path.unlink(missing_ok=True)
        return None
    return data


@contextmanager
def run_lock(home: Path, source: str) -> Iterator[dict[str, Any]]:
    home.mkdir(parents=True, exist_ok=True)
    path = home / LOCK_FILENAME
    holder(home)  # clears a stale lock
    info: dict[str, Any] = {"pid": os.getpid(), "source": source, "run_id": None}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RunLockedError(holder(home) or {"pid": "?"}) from None
    with os.fdopen(fd, "w") as fh:
        json.dump(info, fh)
    try:
        yield info
    finally:
        path.unlink(missing_ok=True)


def record_run_id(home: Path, run_id: int) -> None:
    path = home / LOCK_FILENAME
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return
    data["run_id"] = run_id
    path.write_text(json.dumps(data))

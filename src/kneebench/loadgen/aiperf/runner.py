"""Run one benchmark point with the bundled AIPerf."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import yaml

from kneebench.loadgen.aiperf.command import CONFIG_FILENAME, build_config, build_env
from kneebench.loadgen.aiperf.redact import redact, redact_argv, scrub_tree
from kneebench.loadgen.aiperf.version import aiperf_argv, check_aiperf_version
from kneebench.loadgen.base import Endpoint, PointResult, PointSpec

STDOUT_LOG = "kneebench_aiperf_stdout.log"
STDERR_LOG = "kneebench_aiperf_stderr.log"
_TAIL_CHARS = 4000


class AiperfLoadGenerator:
    name = "aiperf"

    def version(self) -> str:
        return check_aiperf_version()

    def run_point(
        self,
        endpoint: Endpoint,
        spec: PointSpec,
        artifact_dir: Path,
        timeout_s: float | None = None,
        cancel: threading.Event | None = None,
    ) -> PointResult:
        artifact_dir = artifact_dir.resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config_path = artifact_dir / CONFIG_FILENAME
        config_path.write_text(yaml.safe_dump(build_config(endpoint, spec, artifact_dir)))
        config_path.chmod(0o600)

        # aiperf 0.13.0 overwrites the YAML networkLatency block with {enabled: false}
        # unless the CLI flag is also given (ai-dynamo/aiperf#1583), so pass it here too.
        argv = [
            *aiperf_argv(),
            "profile",
            "--config",
            str(config_path),
            "--ui",
            "none",
            "--network-latency-automatic",
        ]
        env = build_env(endpoint, dict(os.environ))
        secrets = [endpoint.api_key]

        started = time.monotonic()
        with (
            (artifact_dir / STDOUT_LOG).open("w") as out,
            (artifact_dir / STDERR_LOG).open("w") as err,
        ):
            proc = subprocess.Popen(argv, stdout=out, stderr=err, env=env, cwd=artifact_dir)
            exit_code = _wait(proc, started, timeout_s, cancel)
        wall = time.monotonic() - started

        scrub_tree(artifact_dir, secrets)
        return PointResult(
            spec=spec,
            artifact_dir=artifact_dir,
            exit_code=exit_code,
            wall_time_s=wall,
            argv_redacted=redact_argv(argv, secrets),
            stdout_tail=_tail(artifact_dir / STDOUT_LOG, secrets),
            stderr_tail=_tail(artifact_dir / STDERR_LOG, secrets),
        )


CANCELLED_EXIT = -15
TIMEOUT_EXIT = -9
_POLL_S = 0.5


def _wait(
    proc: subprocess.Popen[bytes],
    started: float,
    timeout_s: float | None,
    cancel: threading.Event | None,
) -> int:
    """Wait for AIPerf, stopping it (SIGTERM, then SIGKILL) on cancel or timeout."""
    while True:
        try:
            return proc.wait(timeout=_POLL_S)
        except subprocess.TimeoutExpired:
            pass
        cancelled = cancel is not None and cancel.is_set()
        timed_out = timeout_s is not None and time.monotonic() - started > timeout_s
        if cancelled or timed_out:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return CANCELLED_EXIT if cancelled else TIMEOUT_EXIT


def _tail(path: Path, secrets: list[str | None]) -> str:
    if not path.exists():
        return ""
    return redact(path.read_text(errors="replace")[-_TAIL_CHARS:], secrets)

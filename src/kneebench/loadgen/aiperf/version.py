"""Locate the bundled AIPerf and refuse to run any other version."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

# Upgrade deliberately: bump here and in pyproject.toml together, with a CHANGELOG entry.
AIPERF_PIN = "0.13.0"


class AiperfVersionError(RuntimeError):
    pass


def installed_aiperf_version() -> str | None:
    try:
        return importlib.metadata.version("aiperf")
    except importlib.metadata.PackageNotFoundError:
        return None


def check_aiperf_version() -> str:
    installed = installed_aiperf_version()
    if installed is None:
        raise AiperfVersionError(
            "AIPerf is missing from kneebench's environment. Reinstall kneebench "
            "(`uv sync` in the repo, or `uv tool install --reinstall kneebench`)."
        )
    if installed != AIPERF_PIN:
        raise AiperfVersionError(
            f"kneebench {_kneebench_version()} is tested with aiperf=={AIPERF_PIN}, "
            f"but aiperf=={installed} is installed in its environment. "
            "Run `uv sync` (or reinstall kneebench) to restore the pinned version."
        )
    return installed


def aiperf_argv() -> list[str]:
    """Command prefix that runs the AIPerf next to this interpreter, never one from PATH."""
    check_aiperf_version()
    if importlib.util.find_spec("aiperf.__main__") is not None:
        return [sys.executable, "-m", "aiperf"]
    script = Path(sys.executable).with_name("aiperf")
    if script.exists():
        return [str(script)]
    raise AiperfVersionError(
        f"Found aiperf=={AIPERF_PIN} but no way to launch it from {sys.executable}."
    )


def _kneebench_version() -> str:
    from kneebench import __version__

    return __version__

import sys
import tomllib
from pathlib import Path

import pytest

from kneebench.loadgen.aiperf import version as v


def test_bundled_aiperf_matches_pin() -> None:
    assert v.check_aiperf_version() == v.AIPERF_PIN


def test_argv_uses_this_interpreter_not_path() -> None:
    assert v.aiperf_argv() == [sys.executable, "-m", "aiperf"]


def test_version_mismatch_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v, "installed_aiperf_version", lambda: "9.9.9")
    with pytest.raises(v.AiperfVersionError, match=r"aiperf==9\.9\.9"):
        v.aiperf_argv()


def test_missing_aiperf_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v, "installed_aiperf_version", lambda: None)
    with pytest.raises(v.AiperfVersionError, match="missing"):
        v.check_aiperf_version()


def test_pin_matches_pyproject() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    assert f"aiperf=={v.AIPERF_PIN}" in pyproject["project"]["dependencies"]

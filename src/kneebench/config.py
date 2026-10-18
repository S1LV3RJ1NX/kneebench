"""Where kneebench keeps its data, and the optional local (gitignored) config."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import platformdirs

LOCAL_CONFIG_FILENAME = "kneebench.local.toml"
API_KEY_ENV = "KNEEBENCH_API_KEY"


def data_dir() -> Path:
    override = os.environ.get("KNEEBENCH_HOME")
    path = Path(override) if override else Path(platformdirs.user_data_dir("kneebench"))
    return path.expanduser()


@dataclass(frozen=True)
class LocalConfig:
    url: str | None = None
    model: str | None = None
    tokenizer: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> LocalConfig:
        endpoint = data.get("endpoint", {})
        return cls(
            url=endpoint.get("url"),
            model=endpoint.get("model"),
            tokenizer=endpoint.get("tokenizer"),
        )


def load_local_config(start: Path | None = None) -> LocalConfig:
    """Read ``kneebench.local.toml`` from the working directory, if present.

    It never holds secrets: API keys come from the environment (KNEEBENCH_API_KEY).
    """
    path = (start or Path.cwd()) / LOCAL_CONFIG_FILENAME
    if not path.exists():
        return LocalConfig()
    with path.open("rb") as fh:
        return LocalConfig.from_mapping(tomllib.load(fh))


def api_key_from_env() -> str | None:
    return os.environ.get(API_KEY_ENV) or None

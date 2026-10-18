"""User-facing text and reviewable reference data (YAML), kept apart from analysis code."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Any

import yaml


@cache
def load(name: str) -> Any:
    return yaml.safe_load(files("kneebench.content").joinpath(name).read_text())

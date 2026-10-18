"""kneebench: benchmark an LLM inference server and get a report you can act on."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kneebench")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]

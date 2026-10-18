"""Scrub secrets from anything AIPerf wrote or printed.

AIPerf redacts the key in its JSON exports, but its config-validation errors echo
input values verbatim, so every log and text artifact is scrubbed after each point.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

REDACTED = "<redacted>"
_TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".log", ".yaml", ".yml"}
_MAX_SCRUB_BYTES = 512 * 1024 * 1024


def redact(text: str, secrets: Iterable[str | None]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def redact_argv(argv: Iterable[str], secrets: Iterable[str | None]) -> tuple[str, ...]:
    secret_list = list(secrets)
    return tuple(redact(arg, secret_list) for arg in argv)


def scrub_tree(root: Path, secrets: Iterable[str | None]) -> list[Path]:
    """Rewrite text files under ``root`` that contain a secret. Returns the files changed."""
    secret_list = [s for s in secrets if s]
    if not secret_list or not root.exists():
        return []
    encoded = [s.encode() for s in secret_list]
    changed: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        if path.stat().st_size > _MAX_SCRUB_BYTES:
            continue
        data = path.read_bytes()
        if not any(s in data for s in encoded):
            continue
        for s in encoded:
            data = data.replace(s, REDACTED.encode())
        path.write_bytes(data)
        changed.append(path)
    return changed

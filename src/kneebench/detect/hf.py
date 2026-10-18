"""Map what a server reports as a model's root to a Hugging Face model id."""

from __future__ import annotations

import re

_HF_CACHE_DIR = re.compile(r"models--(?P<org>[^/\\]+?)--(?P<name>[^/\\]+)")
_HF_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def root_to_hf_id(root: str | None) -> str | None:
    """Best-effort mapping; returns None when the root can't be mapped with confidence.

    Handles an HF id (``org/name``) and an HF cache path
    (``.../models--org--name/snapshots/<sha>``). Arbitrary local paths are not guessed.
    """
    if not root:
        return None
    root = root.strip()
    match = _HF_CACHE_DIR.search(root)
    if match:
        return f"{match['org']}/{match['name']}"
    if _HF_ID.match(root):
        return root
    return None

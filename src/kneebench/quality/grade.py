"""Check an answer against its expected value by rule. No model is asked to judge."""

from __future__ import annotations

import re
import unicodedata

from kneebench.quality.evalset import Item

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CHOICE = re.compile(r"\b([A-J])\b")
_CHOICE_TAGGED = re.compile(r"(?:answer|option)\s*(?:is)?\s*[:\-]?\s*\(?([A-J])\b", re.IGNORECASE)


def normalize(text: str, keep_case: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text).strip().strip("\"'`").strip()
    text = re.sub(r"\s+", " ", text).rstrip(".!")
    return text if keep_case else text.lower()


def is_correct(item: Item, text: str | None) -> bool:
    if text is None:
        return False
    if item.match == "exact":
        return normalize(text) == normalize(item.answer)
    if item.match == "exact_case":
        return normalize(text, keep_case=True) == normalize(item.answer, keep_case=True)
    if item.match == "contains":
        return normalize(item.answer) in normalize(text)
    if item.match == "number":
        return _last_number(text) == _last_number(item.answer)
    if item.match == "choice":
        return _choice(text) == item.answer.strip().upper()
    if item.match == "regex":
        return re.search(item.answer, text) is not None
    raise ValueError(f"unknown match {item.match!r}")


def _last_number(text: str) -> float | None:
    found = _NUMBER.findall(text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def _choice(text: str) -> str | None:
    tagged = _CHOICE_TAGGED.search(text)
    if tagged:
        return tagged.group(1).upper()
    plain = _CHOICE.search(text.strip())
    return plain.group(1) if plain else None

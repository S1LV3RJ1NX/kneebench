"""Eval sets: questions with an expected answer and a rule for checking it.

One JSON object per line::

    {"id": "q1", "prompt": "What is 2 + 2? Reply with only the number.",
     "answer": "4", "match": "number", "category": "arithmetic"}

``messages`` (an OpenAI chat list) can replace ``prompt`` to use your own system prompt and
history (lesson 54 section 6). ``max_tokens`` is optional per question.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

MATCHES = ("exact", "exact_case", "contains", "number", "choice", "regex")
DEFAULT_MAX_TOKENS = 64
MAX_ITEMS = 5000
STARTER = "starter"


class EvalSetError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    messages: tuple[dict[str, str], ...]
    answer: str
    match: str
    category: str
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class EvalSet:
    name: str
    items: tuple[Item, ...]
    text: str  # the JSONL as given, stored so the after-change pass asks the same questions

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def categories(self) -> list[str]:
        return sorted({i.category for i in self.items})


def parse(name: str, text: str) -> EvalSet:
    items: list[Item] = []
    seen: set[str] = set()
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalSetError(f"line {n}: not JSON ({exc.msg})") from exc
        if not isinstance(raw, dict):
            raise EvalSetError(f"line {n}: expected an object")
        item = _item(raw, n)
        if item.id in seen:
            raise EvalSetError(f"line {n}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise EvalSetError("the eval set is empty")
    if len(items) > MAX_ITEMS:
        raise EvalSetError(f"at most {MAX_ITEMS} questions per set")
    return EvalSet(name=name, items=tuple(items), text=text)


def starter() -> EvalSet:
    text = files("kneebench.content").joinpath("quality/starter.jsonl").read_text()
    return parse(STARTER, text)


def _item(raw: dict[str, Any], n: int) -> Item:
    match = str(raw.get("match") or "exact")
    if match not in MATCHES:
        raise EvalSetError(f"line {n}: match must be one of {', '.join(MATCHES)}")
    if "answer" not in raw:
        raise EvalSetError(f"line {n}: missing answer")
    messages = raw.get("messages")
    if messages is None:
        if not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
            raise EvalSetError(f"line {n}: needs prompt or messages")
        messages = [{"role": "user", "content": raw["prompt"]}]
    if not isinstance(messages, list) or not all(
        isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages
    ):
        raise EvalSetError(f"line {n}: messages must be a list of {{role, content}}")
    max_tokens = raw.get("max_tokens", DEFAULT_MAX_TOKENS)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise EvalSetError(f"line {n}: max_tokens must be a positive integer")
    return Item(
        id=str(raw.get("id") or f"line{n}"),
        messages=tuple(
            {"role": str(m.get("role", "user")), "content": m["content"]} for m in messages
        ),
        answer=str(raw["answer"]),
        match=match,
        category=str(raw.get("category") or "uncategorized"),
        max_tokens=max_tokens,
    )

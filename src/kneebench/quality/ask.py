"""Ask every question once, several at a time, at temperature 0.

Answers are compared, not timed, so this is a plain chat client rather than AIPerf. The
questions go out concurrently on purpose: the noise floor has to be measured the way the
server really runs, with batches that change shape (lesson 54 section 4).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from kneebench.detect.endpoint import normalize_base_url
from kneebench.loadgen.base import Endpoint
from kneebench.quality.evalset import EvalSet, Item
from kneebench.quality.grade import is_correct

CONCURRENCY = 16
TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Answer:
    item_id: str
    category: str
    text: str | None
    correct: bool
    finish_reason: str | None = None
    output_tokens: int | None = None
    error: str | None = None


def ask_all(
    endpoint: Endpoint,
    evalset: EvalSet,
    concurrency: int = CONCURRENCY,
    on_progress: Callable[[int, int], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Answer]:
    return asyncio.run(_ask_all(endpoint, evalset, concurrency, on_progress, transport))


async def _ask_all(
    endpoint: Endpoint,
    evalset: EvalSet,
    concurrency: int,
    on_progress: Callable[[int, int], None] | None,
    transport: httpx.AsyncBaseTransport | None,
) -> list[Answer]:
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    headers.update(dict(endpoint.extra_headers))
    url = f"{normalize_base_url(endpoint.base_url)}/v1/chat/completions"
    gate = asyncio.Semaphore(concurrency)
    done = 0
    total = len(evalset.items)

    async with httpx.AsyncClient(
        timeout=TIMEOUT_S, headers=headers, transport=transport, follow_redirects=True
    ) as client:

        async def one(item: Item) -> Answer:
            nonlocal done
            async with gate:
                answer = await _ask(client, url, endpoint.model, item)
            done += 1
            if on_progress:
                on_progress(done, total)
            return answer

        return list(await asyncio.gather(*(one(i) for i in evalset.items)))


async def _ask(client: httpx.AsyncClient, url: str, model: str, item: Item) -> Answer:
    body = {
        "model": model,
        "messages": list(item.messages),
        "max_tokens": item.max_tokens,
        "temperature": 0,
    }
    try:
        resp = await client.post(url, json=body)
    except httpx.HTTPError as exc:
        return Answer(item.id, item.category, None, False, error=type(exc).__name__)
    if resp.status_code != 200:
        return Answer(item.id, item.category, None, False, error=f"HTTP {resp.status_code}")
    try:
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        return Answer(item.id, item.category, None, False, error="unexpected response")
    usage = data.get("usage") or {}
    return Answer(
        item.id,
        item.category,
        text,
        is_correct(item, text),
        finish_reason=choice.get("finish_reason"),
        output_tokens=usage.get("completion_tokens"),
    )

"""Read-only probes of an OpenAI-compatible endpoint."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

_TIMEOUT = httpx.Timeout(10.0, connect=10.0)


def normalize_base_url(url: str) -> str:
    """Strip trailing slashes and a trailing ``/v1``; keep any gateway path prefix."""
    url = url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    if "://" not in url:
        url = f"http://{url}"
    return url


@dataclass
class ProbeResult:
    path: str
    ok: bool
    status: int | None = None
    elapsed_ms: float | None = None
    error: str | None = None
    body: Any = None
    text: str | None = field(default=None, repr=False)


def probe(
    base_url: str,
    path: str,
    api_key: str | None = None,
    expect_json: bool = True,
    client: httpx.Client | None = None,
) -> ProbeResult:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT, follow_redirects=True)
    started = time.perf_counter()
    try:
        resp = client.get(f"{normalize_base_url(base_url)}{path}", headers=headers)
    except httpx.HTTPError as exc:
        return ProbeResult(path=path, ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if owns_client:
            client.close()
    elapsed = (time.perf_counter() - started) * 1000
    result = ProbeResult(path=path, ok=resp.is_success, status=resp.status_code, elapsed_ms=elapsed)
    if resp.is_success:
        if expect_json:
            try:
                result.body = resp.json()
            except ValueError:
                result.ok = False
                result.error = "response was not JSON"
        else:
            result.text = resp.text
    return result

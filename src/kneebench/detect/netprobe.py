"""Measure client distance: DNS, TCP connect, TLS handshake and an HTTP round trip.

The keep-alive GET /health round trip is the network-only baseline: it crosses the same
network and gateway as a real request but does almost no server work.
"""

from __future__ import annotations

import socket
import ssl
import statistics
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from kneebench.detect.endpoint import normalize_base_url


@dataclass(frozen=True)
class NetworkProfile:
    dns_ms: float | None
    tcp_connect_ms: float | None
    tls_handshake_ms: float | None
    http_rtt_ms: float | None
    http_rtt_samples: tuple[float, ...] = ()
    error: str | None = None


def probe_network(
    base_url: str,
    api_key: str | None = None,
    samples: int = 5,
    http_samples: int = 10,
    timeout_s: float = 10.0,
) -> NetworkProfile:
    base = normalize_base_url(base_url)
    parsed = urlparse(base)
    host = parsed.hostname or ""
    tls = parsed.scheme == "https"
    port = parsed.port or (443 if tls else 80)

    try:
        t0 = time.perf_counter()
        addr = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4]
        dns_ms = (time.perf_counter() - t0) * 1000
    except OSError as exc:
        return NetworkProfile(None, None, None, None, error=f"DNS failed: {exc}")

    tcp: list[float] = []
    tls_times: list[float] = []
    ctx = ssl.create_default_context() if tls else None
    for _ in range(samples):
        try:
            t0 = time.perf_counter()
            sock = socket.create_connection((str(addr[0]), int(addr[1])), timeout=timeout_s)
            tcp.append((time.perf_counter() - t0) * 1000)
            if ctx is not None:
                t1 = time.perf_counter()
                with ctx.wrap_socket(sock, server_hostname=host):
                    tls_times.append((time.perf_counter() - t1) * 1000)
            else:
                sock.close()
        except OSError as exc:
            return NetworkProfile(
                dns_ms, _median(tcp), _median(tls_times), None, error=f"connect failed: {exc}"
            )

    rtts: list[float] = []
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=timeout_s, headers=headers) as client:
        for i in range(http_samples + 1):
            try:
                t0 = time.perf_counter()
                client.get(f"{base}/health")
                elapsed = (time.perf_counter() - t0) * 1000
            except httpx.HTTPError:
                break
            if i > 0:  # the first request pays connection setup
                rtts.append(elapsed)

    return NetworkProfile(
        dns_ms=dns_ms,
        tcp_connect_ms=_median(tcp),
        tls_handshake_ms=_median(tls_times),
        http_rtt_ms=_median(rtts),
        http_rtt_samples=tuple(rtts),
    )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None

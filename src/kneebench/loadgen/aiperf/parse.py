"""Parse an AIPerf 0.13 artifact directory into plain data.

Files read: profile_export_aiperf.json (aggregates), profile_export.jsonl (per request),
profile_export_aiperf_timeslices.json, server_metrics_export.json and the captured
console output (for AIPerf's warning panels).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kneebench.loadgen.aiperf.runner import STDOUT_LOG

SUMMARY_FILE = "profile_export_aiperf.json"
RECORDS_FILE = "profile_export.jsonl"
TIMESLICES_FILE = "profile_export_aiperf_timeslices.json"
SERVER_METRICS_FILE = "server_metrics_export.json"

_PANEL_TITLE = re.compile(r"^╭─+\s*(?P<title>.+?)\s*─+╮\s*$")
_PANEL_END = re.compile(r"^╰─+╯\s*$")
_INFO_PANELS = {"GPU Telemetry Platform"}


@dataclass(frozen=True)
class RequestRecord:
    phase: str
    start_ns: int
    end_ns: int
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float | None
    isl: int | None
    osl: int | None
    error: str | None = None
    error_code: int | None = None
    # DNS lookup plus TCP and TLS setup when the request opened a new connection; 0 if reused.
    connect_ms: float | None = None
    # Time between consecutive streamed chunks after the first token (ms).
    chunk_gaps_ms: tuple[float, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Timeslice:
    start_ns: int
    end_ns: int
    metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ServerSeries:
    metric: str
    kind: str
    labels: dict[str, str]
    stats: dict[str, float]
    timeslices: tuple[dict[str, float], ...] = ()


@dataclass(frozen=True)
class AiperfWarning:
    title: str
    body: str


@dataclass
class ParsedPoint:
    aiperf_version: str | None
    aggregates: dict[str, dict[str, float]]
    units: dict[str, str]
    requests: list[RequestRecord]
    timeslices: list[Timeslice]
    server: dict[str, list[ServerSeries]]
    warnings: list[AiperfWarning]
    profiling_window_ns: tuple[int, int] | None
    was_cancelled: bool
    error_summary: list[Any] = field(default_factory=list)

    def stat(self, metric: str, stat: str = "avg") -> float | None:
        return self.aggregates.get(metric, {}).get(stat)

    def profiling_requests(self) -> list[RequestRecord]:
        return [r for r in self.requests if r.phase == "profiling"]

    def server_series(self, *names: str) -> list[ServerSeries]:
        """First metric name present wins (names differ across vLLM versions)."""
        for name in names:
            if self.server.get(name):
                return self.server[name]
        return []


def parse_point(artifact_dir: Path) -> ParsedPoint:
    summary = _read_json(artifact_dir / SUMMARY_FILE) or {}
    aggregates, units = _aggregates(summary)
    server_raw = _read_json(artifact_dir / SERVER_METRICS_FILE)
    return ParsedPoint(
        aiperf_version=summary.get("aiperf_version"),
        aggregates=aggregates,
        units=units,
        requests=_records(artifact_dir / RECORDS_FILE),
        timeslices=_timeslices(artifact_dir / TIMESLICES_FILE),
        server=_server(server_raw),
        warnings=parse_warnings(_read_text(artifact_dir / STDOUT_LOG)),
        profiling_window_ns=_profiling_window(server_raw, artifact_dir),
        was_cancelled=bool(summary.get("was_cancelled", False)),
        error_summary=list(summary.get("error_summary") or []),
    )


def parse_warnings(console: str) -> list[AiperfWarning]:
    warnings: list[AiperfWarning] = []
    title: str | None = None
    body: list[str] = []
    for line in console.splitlines():
        if title is None:
            match = _PANEL_TITLE.match(line)
            if match:
                title, body = match["title"], []
            continue
        if _PANEL_END.match(line):
            if title not in _INFO_PANELS:
                warnings.append(AiperfWarning(title=title, body="\n".join(body).strip()))
            title = None
            continue
        body.append(line.strip("│ ").rstrip())
    return warnings


def _aggregates(summary: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    aggregates: dict[str, dict[str, float]] = {}
    units: dict[str, str] = {}
    for name, value in summary.items():
        if not isinstance(value, dict) or "unit" not in value:
            continue
        units[name] = str(value["unit"])
        aggregates[name] = {
            k: float(v) for k, v in value.items() if k != "unit" and isinstance(v, int | float)
        }
    return aggregates, units


def _records(path: Path) -> list[RequestRecord]:
    if not path.exists():
        return []
    out: list[RequestRecord] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            metrics = rec.get("metrics", {})
            error = rec.get("error")
            out.append(
                RequestRecord(
                    phase=str(meta.get("phase_name") or meta.get("benchmark_phase") or ""),
                    start_ns=int(meta.get("request_start_ns") or 0),
                    end_ns=int(meta.get("request_end_ns") or 0),
                    ttft_ms=_value(metrics, "time_to_first_token"),
                    tpot_ms=_value(metrics, "inter_token_latency"),
                    e2e_ms=_value(metrics, "request_latency"),
                    isl=_int(_value(metrics, "input_sequence_length")),
                    osl=_int(_value(metrics, "output_sequence_length")),
                    error=_error_text(error),
                    error_code=_error_code(error),
                    connect_ms=_connect_ms(metrics),
                    chunk_gaps_ms=_gaps(metrics),
                )
            )
    return out


def _timeslices(path: Path) -> list[Timeslice]:
    data = _read_json(path)
    if not data:
        return []
    slices: list[Timeslice] = []
    for raw in data.get("timeslices", []):
        metrics = {
            k: {s: float(x) for s, x in v.items() if isinstance(x, int | float)}
            for k, v in raw.items()
            if isinstance(v, dict)
        }
        slices.append(Timeslice(int(raw["start_ns"]), int(raw["end_ns"]), metrics))
    return slices


def _server(data: dict[str, Any] | None) -> dict[str, list[ServerSeries]]:
    if not data:
        return {}
    out: dict[str, list[ServerSeries]] = {}
    for name, metric in (data.get("metrics") or {}).items():
        series = []
        for s in metric.get("series", []):
            series.append(
                ServerSeries(
                    metric=name,
                    kind=str(metric.get("type", "")),
                    labels={k: str(v) for k, v in (s.get("labels") or {}).items()},
                    stats={
                        k: float(v)
                        for k, v in (s.get("stats") or {}).items()
                        if isinstance(v, int | float)
                    },
                    timeslices=tuple(
                        {k: float(v) for k, v in t.items() if isinstance(v, int | float)}
                        for t in s.get("timeslices") or []
                    ),
                )
            )
        out[name] = series
    return out


def _profiling_window(server: dict[str, Any] | None, artifact_dir: Path) -> tuple[int, int] | None:
    manifest = _read_json(artifact_dir / "phase_manifest.json")
    for phase in (manifest or {}).get("phases", []):
        if phase.get("phase_kind") == "profiling":
            return int(phase["start_ns"]), int(phase["end_ns"])
    ranges = ((server or {}).get("summary") or {}).get("phase_time_ranges") or {}
    prof = ranges.get("profiling")
    return (int(prof["start_ns"]), int(prof["end_ns"])) if prof else None


def _gaps(metrics: dict[str, Any]) -> tuple[float, ...]:
    entry = metrics.get("inter_chunk_latency")
    values = entry.get("value") if isinstance(entry, dict) else None
    if not isinstance(values, list):
        return ()
    return tuple(float(v) for v in values if isinstance(v, int | float))


def _connect_ms(metrics: dict[str, Any]) -> float | None:
    parts = [_value(metrics, k) for k in ("http_req_dns_lookup", "http_req_connecting")]
    known = [p for p in parts if p is not None]
    return sum(known) if known else None


def _value(metrics: dict[str, Any], name: str) -> float | None:
    entry = metrics.get(name)
    if isinstance(entry, dict) and isinstance(entry.get("value"), int | float):
        return float(entry["value"])
    return None


def _int(value: float | None) -> int | None:
    return None if value is None else int(value)


def _error_text(error: Any) -> str | None:
    if error is None:
        return None
    if isinstance(error, dict):
        code = error.get("code")
        kind = error.get("type") or error.get("message") or "error"
        return f"{code} {kind}".strip() if code else str(kind)
    return str(error)


def _error_code(error: Any) -> int | None:
    if isinstance(error, dict) and isinstance(error.get("code"), int):
        return int(error["code"])
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None


def _read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


# Background streams are the short prompts in a scheduled point; anything this much longer
# than the shortest prompt is an injected one (lesson 44 section 2).
_INJECTED_ISL_FACTOR = 2
# A gap counts as a stall at this multiple of the typical gap, and never below 40 ms
# (kneebench estimate: a few decode steps).
STALL_FACTOR = 4.0
STALL_MIN_MS = 40.0
STREAM_BIN_S = 0.25


def stream_summary(requests: list[RequestRecord]) -> dict[str, Any] | None:
    """What the streaming answers felt while long prompts arrived (lesson 44).

    Rebuilds every chunk's arrival time from the request start, its first token and the
    chunk gaps, then keeps only compact results: gap percentiles, stalls, the largest gap
    every quarter second, and when each long prompt arrived.
    """
    sized = [r for r in requests if r.isl]
    if not sized:
        return None
    shortest = min(r.isl or 0 for r in sized)
    background = [
        r
        for r in sized
        if r.ok
        and r.chunk_gaps_ms
        and r.ttft_ms is not None
        and (r.isl or 0) <= _INJECTED_ISL_FACTOR * shortest
    ]
    injected = [r for r in sized if (r.isl or 0) > _INJECTED_ISL_FACTOR * shortest]
    if not background:
        return None
    t0 = min(r.start_ns for r in requests)
    # Skip the first second (streams still starting) and stop when the first stream ends.
    first_end = min((r.end_ns - t0) / 1e9 for r in background)
    events: list[tuple[float, float]] = []
    for r in background:
        t = r.start_ns + (r.ttft_ms or 0.0) * 1e6
        for gap in r.chunk_gaps_ms:
            t += gap * 1e6
            at = (t - t0) / 1e9
            if 1.0 <= at <= first_end:
                events.append((at, gap))
    # Gateways can bundle chunks; the zero gaps that leaves say nothing about the server.
    gaps = sorted(g for _, g in events if g > 0)
    if not gaps:
        return None

    def pct(q: float) -> float:
        return gaps[min(len(gaps) - 1, int(q * (len(gaps) - 1)))]

    typical = pct(0.5)
    stall_ms = max(STALL_MIN_MS, STALL_FACTOR * typical)
    bins: dict[int, float] = {}
    for at, gap in events:
        i = int(at / STREAM_BIN_S)
        bins[i] = max(bins.get(i, 0.0), gap)
    return {
        "streams": len(background),
        "gaps": len(gaps),
        "p50_ms": typical,
        "p99_ms": pct(0.99),
        "p999_ms": pct(0.999),
        "max_ms": gaps[-1],
        "stall_ms": stall_ms,
        "stalls": sum(1 for g in gaps if g > stall_ms),
        "window_s": first_end,
        "series": [[round(i * STREAM_BIN_S, 2), round(g, 1)] for i, g in sorted(bins.items())],
        "injected": [
            {
                "at_s": round((r.start_ns - t0) / 1e9, 2),
                "isl": r.isl,
                "ttft_ms": r.ttft_ms,
                "ok": r.ok,
            }
            for r in sorted(injected, key=lambda r: r.start_ns)
        ],
    }

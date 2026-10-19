"""`kneebench doctor`: check the install and the endpoint, with plain-English fixes."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kneebench.config import API_KEY_ENV, data_dir
from kneebench.detect.endpoint import normalize_base_url, probe
from kneebench.detect.hf import root_to_hf_id
from kneebench.loadgen.aiperf.version import AIPERF_PIN, AiperfVersionError, aiperf_argv

MIN_PY = (3, 11)
MAX_PY_EXCLUSIVE = (3, 14)


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    fix: str | None = None


def check_python(version: tuple[int, int, int] | None = None) -> Check:
    major, minor, micro = version or sys.version_info[:3]
    current = f"{major}.{minor}.{micro}"
    if MIN_PY <= (major, minor) < MAX_PY_EXCLUSIVE:
        return Check("Python", Status.OK, f"Python {current}")
    return Check(
        "Python",
        Status.FAIL,
        f"Python {current} is outside the supported range 3.11 to 3.13 (AIPerf's range).",
        "Install with a supported Python, e.g. `uv tool install --python 3.12 kneebench`.",
    )


def check_platform() -> Check:
    if sys.platform in ("darwin", "linux"):
        return Check("Platform", Status.OK, sys.platform)
    return Check(
        "Platform",
        Status.FAIL,
        f"{sys.platform} is not supported. kneebench runs on macOS and Linux.",
        "On Windows, run kneebench inside WSL2.",
    )


def check_aiperf() -> Check:
    try:
        argv = aiperf_argv()
    except AiperfVersionError as exc:
        return Check("AIPerf", Status.FAIL, str(exc), "Run `uv sync`, or reinstall kneebench.")
    try:
        proc = subprocess.run(
            [*argv, "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check(
            "AIPerf",
            Status.FAIL,
            f"aiperf=={AIPERF_PIN} is installed but would not start: {exc}",
            "Reinstall kneebench; if it persists, open an issue with this output.",
        )
    reported = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0 or reported != AIPERF_PIN:
        return Check(
            "AIPerf",
            Status.FAIL,
            f"Bundled AIPerf failed to report its version (exit {proc.returncode}).",
            "Reinstall kneebench; if it persists, open an issue with `kneebench doctor` output.",
        )
    return Check("AIPerf", Status.OK, f"aiperf=={AIPERF_PIN}, launched via {argv[0]}")


def check_data_dir() -> Check:
    path = data_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            pass
    except OSError as exc:
        return Check(
            "Data directory",
            Status.FAIL,
            f"Can't write to {path}: {exc}",
            "Set KNEEBENCH_HOME to a writable directory.",
        )
    return Check("Data directory", Status.OK, str(path))


def check_pdf() -> Check:
    from kneebench.report.pdf import chromium_installed

    if chromium_installed():
        return Check("PDF export", Status.OK, "headless Chromium installed")
    return Check(
        "PDF export",
        Status.WARN,
        "Headless Chromium is not installed, so `Download PDF` won't work yet.",
        "Run `kneebench doctor --install-pdf` (on a bare Linux image, also "
        "`sudo uv run playwright install-deps chromium`), or use the report's "
        '"Print or save as PDF" button.',
    )


def check_tokenizer(tokenizer: str) -> Check:
    if tokenizer == "builtin":
        return Check(
            "Tokenizer",
            Status.WARN,
            "Using AIPerf's builtin tokenizer: token counts won't match your model exactly.",
            "Pass --tokenizer with the served model's Hugging Face id or a local path.",
        )
    if Path(tokenizer).expanduser().exists():
        return Check("Tokenizer", Status.OK, f"local tokenizer at {tokenizer}")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        snapshot_download(
            tokenizer,
            allow_patterns=["tokenizer*", "special_tokens_map.json", "*.model", "config.json"],
        )
    except GatedRepoError:
        return Check(
            "Tokenizer",
            Status.FAIL,
            f"{tokenizer} is a gated model on Hugging Face.",
            "Accept its license on huggingface.co, then set HF_TOKEN to a token with access.",
        )
    except RepositoryNotFoundError:
        return Check(
            "Tokenizer",
            Status.FAIL,
            f"{tokenizer} was not found on Hugging Face.",
            "Check the id, or pass --tokenizer with a local tokenizer directory.",
        )
    except (LocalEntryNotFoundError, HfHubHTTPError, OSError) as exc:
        return Check(
            "Tokenizer",
            Status.FAIL,
            f"Couldn't download the {tokenizer} tokenizer: {exc}",
            "Check internet access to huggingface.co, or pass --tokenizer with a local path.",
        )
    return Check("Tokenizer", Status.OK, f"{tokenizer} downloaded")


def check_endpoint(url: str, api_key: str | None) -> tuple[list[Check], str | None]:
    """Probe the endpoint. Returns the checks and the HF id inferred from /v1/models."""
    base = normalize_base_url(url)
    checks: list[Check] = []

    health = probe(base, "/health", api_key, expect_json=False)
    if health.ok:
        checks.append(Check("GET /health", Status.OK, f"{health.elapsed_ms:.0f} ms"))
    elif health.status is None:
        checks.append(
            Check(
                "GET /health",
                Status.FAIL,
                f"Can't reach {base}: {health.error}",
                "Check the URL and that this machine can reach the server (VPN, firewall, DNS).",
            )
        )
        return checks, None
    else:
        checks.append(
            Check(
                "GET /health",
                Status.WARN,
                f"HTTP {health.status}",
                "Some gateways hide /health; this alone doesn't block a benchmark.",
            )
        )

    version = probe(base, "/version", api_key)
    if version.ok and isinstance(version.body, dict) and "version" in version.body:
        checks.append(Check("GET /version", Status.OK, f"vLLM {version.body['version']}"))
    else:
        checks.append(
            Check(
                "GET /version",
                Status.WARN,
                f"No vLLM version reported ({_why(version.status, version.error)}).",
                "kneebench will ask for the vLLM version; it may not be vLLM, "
                "or a gateway hides the route.",
            )
        )

    hf_id: str | None = None
    models = probe(base, "/v1/models", api_key)
    if models.ok and isinstance(models.body, dict) and models.body.get("data"):
        entries = models.body["data"]
        ids = ", ".join(str(m.get("id")) for m in entries)
        first = entries[0]
        hf_id = root_to_hf_id(first.get("root")) or root_to_hf_id(first.get("id"))
        detail = f"models: {ids}"
        if first.get("max_model_len"):
            detail += f"; max_model_len {first['max_model_len']}"
        checks.append(Check("GET /v1/models", Status.OK, detail))
    elif models.status in (401, 403):
        checks.append(
            Check(
                "GET /v1/models",
                Status.FAIL,
                f"HTTP {models.status}: the server rejected the credentials.",
                f"Set {API_KEY_ENV} to a valid API key.",
            )
        )
    else:
        checks.append(
            Check(
                "GET /v1/models",
                Status.FAIL,
                f"Couldn't list models ({_why(models.status, models.error)}).",
                "Is this an OpenAI-compatible server? The URL should be the server root "
                "(kneebench appends /v1/...).",
            )
        )

    metrics = probe(base, "/metrics", api_key, expect_json=False)
    if metrics.ok and metrics.text and "vllm:" in metrics.text:
        has_cache = "vllm:cache_config_info" in metrics.text
        checks.append(
            Check(
                "GET /metrics",
                Status.OK if has_cache else Status.WARN,
                "vLLM metrics found"
                + ("" if has_cache else ", but no vllm:cache_config_info (older vLLM?)"),
            )
        )
    else:
        checks.append(
            Check(
                "GET /metrics",
                Status.WARN,
                f"No vLLM metrics ({_why(metrics.status, metrics.error)}).",
                "Benchmarks still run, but memory and queue findings (KV cache, preemption) "
                "will be marked as not visible. Expose /metrics through your gateway to get them.",
            )
        )
    return checks, hf_id


def run_doctor(
    offline: bool,
    url: str | None,
    api_key: str | None,
    tokenizer: str | None,
    on_check: Callable[[Check], None] | None = None,
) -> list[Check]:
    checks: list[Check] = []

    def add(check: Check) -> None:
        checks.append(check)
        if on_check:
            on_check(check)

    for fn in (check_python, check_platform, check_aiperf, check_data_dir, check_pdf):
        add(fn())

    if offline:
        add(Check("Endpoint", Status.SKIP, "skipped (--offline)"))
        add(Check("Tokenizer", Status.SKIP, "skipped (--offline)"))
        return checks

    inferred: str | None = None
    if url:
        endpoint_checks, inferred = check_endpoint(url, api_key)
        for c in endpoint_checks:
            add(c)
    else:
        add(Check("Endpoint", Status.SKIP, "no --url given", "Run `kneebench doctor --url <URL>`."))

    chosen = tokenizer or inferred
    if chosen:
        add(check_tokenizer(chosen))
    else:
        add(
            Check(
                "Tokenizer",
                Status.SKIP,
                "no tokenizer to check",
                "Pass --tokenizer <hf-id>, or --url so it can be inferred from /v1/models.",
            )
        )
    return checks


def _why(status: int | None, error: str | None) -> str:
    return f"HTTP {status}" if status is not None else (error or "no response")


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status is Status.FAIL for c in checks) else 0

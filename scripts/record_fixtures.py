"""Record sanitized fixtures from a real endpoint (dev tool, works with any endpoint).

    uv run python scripts/record_fixtures.py endpoint NAME          # /version, /v1/models, /metrics
    uv run python scripts/record_fixtures.py point NAME --isl 512 --osl 256 --concurrency 1 \
        --requests 30 --warmup 10
    uv run python scripts/record_fixtures.py sanitize SRC_DIR NAME  # an existing artifact dir

Endpoint, model and tokenizer come from kneebench.local.toml (gitignored) or flags.
The endpoint URL, host and served model name are replaced with placeholders, so
fixtures never reveal where they were recorded.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

from kneebench.config import api_key_from_env, load_local_config
from kneebench.detect.endpoint import normalize_base_url
from kneebench.loadgen.aiperf.runner import STDOUT_LOG, AiperfLoadGenerator
from kneebench.loadgen.base import Endpoint, LoadMode, PointSpec

FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures"
PLACEHOLDER_URL = "http://vllm.example:8000"
PLACEHOLDER_MODEL = "served-model"
_ROOT_HF = re.compile(r'("root":\s*)"[^"]*?(models--[^"/]+)[^"]*"')
_ROOT_OTHER = re.compile(r'("root":\s*)"/(?!models/)[^"]*"')
POINT_FILES = (
    "aiperf_config.yaml",
    "phase_manifest.json",
    "profile_export_aiperf.json",
    "profile_export.jsonl",
    "profile_export_aiperf_timeslices.json",
    "server_metrics_export.json",
    STDOUT_LOG,
)


class Sanitizer:
    def __init__(self, base_url: str, model: str) -> None:
        base = normalize_base_url(base_url)
        host = urlparse(base).netloc
        # Longest first so the full URL is replaced before its host.
        self.pairs = sorted(
            {(base, PLACEHOLDER_URL), (host, urlparse(PLACEHOLDER_URL).netloc)},
            key=lambda p: -len(p[0]),
        )
        self.model = model

    def text(self, text: str) -> str:
        for old, new in self.pairs:
            text = text.replace(old, new)
        # Model root paths reveal the host's filesystem layout; keep only the HF cache part.
        text = _ROOT_HF.sub(r'\1"/models/hub/\2"', text)
        text = _ROOT_OTHER.sub(r'\1"/models/served-model"', text)
        for quoted in (f'"{self.model}"', f"'{self.model}'", f"={self.model}"):
            text = text.replace(quoted, quoted.replace(self.model, PLACEHOLDER_MODEL))
        return text.replace(f"model: {self.model}\n", f"model: {PLACEHOLDER_MODEL}\n")


def sanitize_point(src: Path, name: str, s: Sanitizer) -> Path:
    dest = FIXTURES / "aiperf" / name
    dest.mkdir(parents=True, exist_ok=True)
    for fname in POINT_FILES:
        path = src / fname
        if path.exists():
            (dest / fname).write_text(s.text(path.read_text(errors="replace")))
    return dest


def record_endpoint(base_url: str, api_key: str | None, name: str, s: Sanitizer) -> Path:
    dest = FIXTURES / "endpoint" / name
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=30, headers=headers) as client:
        for route, fname in (("/version", "version.json"), ("/v1/models", "models.json")):
            resp = client.get(f"{base_url}{route}")
            if resp.is_success:
                body = json.dumps(resp.json(), indent=2, sort_keys=True) + "\n"
                (dest / fname).write_text(s.text(body))
        resp = client.get(f"{base_url}/metrics")
        if resp.is_success:
            (dest / "metrics.prom").write_text(s.text(resp.text))
    return dest


def main() -> None:
    local = load_local_config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default=local.url)
    parser.add_argument("--model", default=local.model)
    parser.add_argument("--tokenizer", default=local.tokenizer)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("endpoint").add_argument("name")
    point = sub.add_parser("point")
    point.add_argument("name")
    point.add_argument("--isl", type=int, required=True)
    point.add_argument("--osl", type=int, required=True)
    point.add_argument("--concurrency", type=int, default=1)
    point.add_argument("--requests", type=int, default=30)
    point.add_argument("--warmup", type=int, default=0)
    san = sub.add_parser("sanitize")
    san.add_argument("src", type=Path)
    san.add_argument("name")
    args = parser.parse_args()

    if not args.url or not args.model:
        parser.error("set --url and --model, or create kneebench.local.toml")
    base = normalize_base_url(args.url)
    s = Sanitizer(base, args.model)

    if args.cmd == "endpoint":
        print(record_endpoint(base, api_key_from_env(), args.name, s))
    elif args.cmd == "sanitize":
        print(sanitize_point(args.src, args.name, s))
    else:
        endpoint = Endpoint(
            base_url=base,
            model=args.model,
            tokenizer=args.tokenizer or "builtin",
            api_key=api_key_from_env(),
        )
        spec = PointSpec(
            isl=args.isl,
            osl=args.osl,
            mode=LoadMode.CONCURRENCY,
            concurrency=args.concurrency,
            request_count=args.requests,
            warmup_count=args.warmup or None,
        )
        with tempfile.TemporaryDirectory(prefix="kneebench-fixture-") as tmp:
            result = AiperfLoadGenerator().run_point(endpoint, spec, Path(tmp) / "point")
            if not result.ok:
                raise SystemExit(f"point failed:\n{result.stderr_tail or result.stdout_tail}")
            print(sanitize_point(result.artifact_dir, args.name, s))
        shutil.rmtree(Path(tmp), ignore_errors=True)


if __name__ == "__main__":
    main()

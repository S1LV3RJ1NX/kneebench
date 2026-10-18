"""Measure AIPerf per-process startup overhead against an endpoint (dev tool).

Usage:
    uv run python scripts/measure_aiperf_startup.py --url http://127.0.0.1:18765 \
        --model mock-llm --tokenizer builtin --runs 5
"""

from __future__ import annotations

import argparse
import platform
import sys

from kneebench.config import api_key_from_env
from kneebench.loadgen.aiperf.overhead import measure_startup_overhead, median_overhead
from kneebench.loadgen.aiperf.version import AIPERF_PIN
from kneebench.loadgen.base import Endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="builtin")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    endpoint = Endpoint(
        base_url=args.url, model=args.model, tokenizer=args.tokenizer, api_key=api_key_from_env()
    )
    samples = measure_startup_overhead(endpoint, runs=args.runs)
    for i, s in enumerate(samples):
        print(
            f"run {i}: wall {s.wall_s:6.2f} s  benchmark {s.benchmark_s:5.2f} s  "
            f"overhead {s.overhead_s:6.2f} s"
        )
    print(
        f"median overhead: {median_overhead(samples):.2f} s "
        f"({platform.system()} {platform.machine()}, Python {sys.version.split()[0]}, "
        f"aiperf {AIPERF_PIN}, tokenizer {args.tokenizer})"
    )


if __name__ == "__main__":
    main()

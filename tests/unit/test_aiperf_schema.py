"""Generated configs must validate against the bundled AIPerf's own schema.

This catches schema drift when the AIPerf pin is upgraded.
"""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from kneebench.loadgen.aiperf.command import API_KEY_ENV, build_config, build_env
from kneebench.loadgen.aiperf.version import aiperf_argv
from kneebench.loadgen.base import Arrival, Endpoint, LoadMode, PointSpec

ENDPOINT = Endpoint(
    base_url="http://server:8000",
    model="m",
    tokenizer="builtin",
    api_key="sk-schema-test",
    extra_headers=(("X-Team", "infra"),),
)

SPECS = {
    "baseline": PointSpec(
        isl=512,
        osl=256,
        mode=LoadMode.CONCURRENCY,
        concurrency=1,
        request_count=30,
        warmup_count=10,
    ),
    "concurrency": PointSpec(
        isl=1024,
        osl=1024,
        mode=LoadMode.CONCURRENCY,
        concurrency=64,
        request_count=256,
        warmup_count=32,
        duration_s=600,
        goodput=(("time_to_first_token", 500.0), ("inter_token_latency", 50.0)),
    ),
    "poisson": PointSpec(
        isl=1024,
        osl=256,
        mode=LoadMode.RATE,
        request_rate=4.0,
        warmup_s=20,
        duration_s=90,
        concurrency=2048,
    ),
    "schedule": PointSpec(
        isl=128,
        osl=400,
        mode=LoadMode.SCHEDULE,
        schedule=((0, 128, 400), (0, 128, 400), (500, 8000, 16)),
    ),
    "gamma": PointSpec(
        isl=1024,
        osl=256,
        mode=LoadMode.RATE,
        request_rate=4.0,
        arrival=Arrival.GAMMA,
        burstiness=0.5,
        warmup_s=20,
        duration_s=90,
    ),
    "varied": PointSpec(
        isl=2048,
        osl=256,
        isl_stddev=102.4,
        osl_stddev=12,
        mode=LoadMode.CONCURRENCY,
        concurrency=8,
        request_count=32,
    ),
}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_config_validates_with_bundled_aiperf(name: str, tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(build_config(ENDPOINT, SPECS[name], tmp_path / "art")))
    env = build_env(ENDPOINT, dict(os.environ))
    proc = subprocess.run(
        [*aiperf_argv(), "config", "validate", str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert env[API_KEY_ENV] not in path.read_text()

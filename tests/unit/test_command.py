from pathlib import Path

import pytest
import yaml

from kneebench.loadgen.aiperf.command import API_KEY_ENV, build_config, build_env
from kneebench.loadgen.base import Arrival, Endpoint, LoadMode, PointSpec

SENTINEL = "sk-kneebench-sentinel-0000"


def _endpoint(api_key: str | None = SENTINEL) -> Endpoint:
    return Endpoint(base_url="http://server:8000", model="m", tokenizer="org/m", api_key=api_key)


def test_concurrency_point_forces_output_length_and_busts_cache(tmp_path: Path) -> None:
    spec = PointSpec(
        isl=512,
        osl=256,
        mode=LoadMode.CONCURRENCY,
        concurrency=1,
        request_count=30,
        warmup_count=10,
    )
    bench = build_config(_endpoint(), spec, tmp_path)["benchmark"]

    assert bench["endpoint"]["extra"]["ignore_eos"] is True
    assert bench["endpoint"]["streaming"] is True
    assert bench["dataset"]["prompts"]["isl"] == {"mean": 512, "stddev": 0.0}
    assert bench["dataset"]["prompts"]["osl"] == {"mean": 256, "stddev": 0}
    assert bench["dataset"]["prompts"]["cacheBust"] == {"target": "first_turn_prefix"}
    warmup, profiling = bench["phases"]
    assert warmup["excludeFromResults"] is True
    assert warmup["requests"] == 10
    assert profiling == {
        "name": "profiling",
        "type": "concurrency",
        "concurrency": 1,
        "requests": 30,
    }
    assert bench["dataset"]["entries"] >= 40


def test_api_key_never_in_config_only_in_env(tmp_path: Path) -> None:
    spec = PointSpec(isl=128, osl=32, mode=LoadMode.CONCURRENCY, concurrency=4, request_count=8)
    cfg = build_config(_endpoint(), spec, tmp_path)

    assert SENTINEL not in yaml.safe_dump(cfg)
    assert cfg["benchmark"]["endpoint"]["apiKey"] == f"${{{API_KEY_ENV}}}"
    assert build_env(_endpoint(), {"PATH": "/bin"})[API_KEY_ENV] == SENTINEL


def test_no_api_key_means_no_placeholder_and_stale_env_removed(tmp_path: Path) -> None:
    spec = PointSpec(isl=128, osl=32, mode=LoadMode.CONCURRENCY, concurrency=1, request_count=2)
    cfg = build_config(_endpoint(api_key=None), spec, tmp_path)
    assert "apiKey" not in cfg["benchmark"]["endpoint"]
    assert API_KEY_ENV not in build_env(_endpoint(api_key=None), {API_KEY_ENV: "stale"})


def test_gamma_rate_point(tmp_path: Path) -> None:
    spec = PointSpec(
        isl=1024,
        osl=1024,
        mode=LoadMode.RATE,
        request_rate=5.0,
        arrival=Arrival.GAMMA,
        burstiness=0.5,
        warmup_s=20,
        duration_s=90,
        concurrency=2048,
    )
    warmup, profiling = build_config(_endpoint(), spec, tmp_path)["benchmark"]["phases"]
    assert warmup["excludeFromResults"] is True
    assert (warmup["type"], warmup["rate"], warmup["duration"]) == ("gamma", 5.0, 20)
    assert "gracePeriod" not in warmup
    assert profiling["seamless"] is True
    assert profiling["type"] == "gamma"
    assert profiling["rate"] == 5.0
    assert profiling["smoothness"] == 0.5
    assert profiling["duration"] == 90
    assert profiling["gracePeriod"] > 0


def test_goodput_slos_pass_through(tmp_path: Path) -> None:
    spec = PointSpec(
        isl=128,
        osl=32,
        mode=LoadMode.CONCURRENCY,
        concurrency=1,
        request_count=2,
        goodput=(("time_to_first_token", 500.0), ("inter_token_latency", 50.0)),
    )
    slos = build_config(_endpoint(), spec, tmp_path)["benchmark"]["slos"]
    assert slos == {"time_to_first_token": 500.0, "inter_token_latency": 50.0}


def test_rate_mode_without_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="explicit request rate"):
        PointSpec(isl=128, osl=32, mode=LoadMode.RATE)

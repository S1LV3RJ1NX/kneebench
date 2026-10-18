"""Re-read vLLM's EngineArgs.get_batch_defaults at every verified tag (needs network).

Run with: uv run pytest -m live tests/live/test_vllm_source.py
"""

from __future__ import annotations

import re

import httpx
import pytest

from kneebench import content

pytestmark = pytest.mark.live

_TIER = re.compile(r"device_memory >= (\d+) \* GiB_bytes")
_SERVER = re.compile(r"UsageContext\.OPENAI_API_SERVER: (\d+)")


def _batch_defaults_source(tag: str) -> str:
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{tag}/vllm/engine/arg_utils.py"
    text = httpx.get(url, timeout=30, follow_redirects=True).raise_for_status().text
    start = text.index("def get_batch_defaults")
    end = text.index("# tpu specific default values", start)
    return text[start:end]


def _tags() -> list[tuple[str, dict[str, object]]]:
    return [
        (tag, r) for r in content.load("vllm_versions.yaml")["ranges"] for tag in r["verified_tags"]
    ]


@pytest.mark.parametrize(("tag", "rng"), _tags(), ids=lambda v: v if isinstance(v, str) else "")
def test_table_matches_vllm_source(tag: str, rng: dict[str, object]) -> None:
    src = _batch_defaults_source(tag)
    thresholds = [int(x) for x in _TIER.findall(src)]
    server_values = [int(x) for x in _SERVER.findall(src)]
    # Pairs per tier in source order: (max_num_batched_tokens, max_num_seqs).
    tiers = list(zip(server_values[0::2], server_values[1::2], strict=True))
    table = rng["batch_defaults"]
    assert isinstance(table, list)
    assert [t["min_gib"] for t in table] == [*thresholds, 0]
    assert [(t["max_num_batched_tokens"], t["max_num_seqs"]) for t in table] == tiers
    excluded = [t.get("exclude_name_contains", []) for t in table]
    for names in excluded:
        for name in names:
            assert f'"{name}" not in device_name' in src

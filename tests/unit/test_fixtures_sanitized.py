"""Committed fixtures must never reveal where they were recorded."""

import re

from tests.fixture_data import FIXTURES

_URL = re.compile(r"https?://([^/\s\"'<>]+)")
ALLOWED_HOSTS = {
    "vllm.example:8000",
    "vllm.example",
    "huggingface.co",
    "github.com",
    "errors.pydantic.dev",
    "arxiv.org",
    "hao-ai-lab.github.io",
    "127.0.0.1",
}


def test_fixture_urls_are_placeholders() -> None:
    for path in FIXTURES.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        hosts = set(_URL.findall(text)) - ALLOWED_HOSTS
        assert not hosts, f"{path.relative_to(FIXTURES)} contains hosts {sorted(hosts)}"


def test_fixture_model_roots_are_sanitized() -> None:
    for path in FIXTURES.rglob("models.json"):
        for root in re.findall(r'"root":\s*"([^"]+)"', path.read_text()):
            assert root.startswith("/models/"), f"{path}: unsanitized root {root}"

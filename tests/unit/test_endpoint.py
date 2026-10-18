import pytest

from kneebench.detect.endpoint import normalize_base_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://host:8000", "http://host:8000"),
        ("http://host:8000/", "http://host:8000"),
        ("http://host:8000/v1", "http://host:8000"),
        ("https://gw.example.com/team/llm/v1/", "https://gw.example.com/team/llm"),
        ("localhost:8000", "http://localhost:8000"),
    ],
)
def test_normalize_base_url(url: str, expected: str) -> None:
    assert normalize_base_url(url) == expected

"""Meta-tests: every rule cites a note, has content, and has tests."""

from importlib.resources import files
from pathlib import Path

import yaml

from kneebench.analysis.compare import COMPARE_REGISTRY
from kneebench.analysis.engine import REGISTRY, _load_rules

RULE_TESTS = Path(__file__).parent


def _content_ids() -> set[str]:
    folder = files("kneebench.content").joinpath("rules")
    return {e.name.removesuffix(".yaml") for e in folder.iterdir() if e.name.endswith(".yaml")}


def test_every_rule_has_content_and_content_has_a_rule() -> None:
    _load_rules()
    assert set(REGISTRY) | set(COMPARE_REGISTRY) == _content_ids()


def test_every_rule_cites_its_source_note() -> None:
    _load_rules()
    for r in (*REGISTRY.values(), *COMPARE_REGISTRY.values()):
        assert r.source[:2] in {"24", "25", "26", "27"}, f"{r.id} cites {r.source!r}"


def test_every_rule_is_tested() -> None:
    _load_rules()
    tested = "".join(p.read_text() for p in RULE_TESTS.glob("test_*_rules.py"))
    missing = [rid for rid in (*REGISTRY, *COMPARE_REGISTRY) if f'"{rid}"' not in tested]
    assert not missing, f"rules without tests: {missing}"


def test_content_files_have_required_parts() -> None:
    folder = files("kneebench.content").joinpath("rules")
    for rid in _content_ids():
        spec = yaml.safe_load(folder.joinpath(f"{rid}.yaml").read_text())
        assert spec["id"] == rid
        assert spec["title"]
        assert spec["status_line"]
        assert spec["why"], f"{rid} lacks 'why'"
        assert spec.get("learn_more", {}).get("note") in {"24", "25", "26", "27"}
        for action in spec.get("actions", []):
            assert action["when"] and action["title"] and action["verify"]

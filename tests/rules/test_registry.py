"""Meta-tests: every rule cites a note, has content, and has tests."""

from importlib.resources import files
from pathlib import Path

import pytest
import yaml

from kneebench.analysis.compare import COMPARE_REGISTRY
from kneebench.analysis.engine import REGISTRY, _load_rules

RULE_TESTS = Path(__file__).parent


def _notes() -> set[str]:
    path = files("kneebench.content").joinpath("notes.yaml")
    return set(yaml.safe_load(path.read_text())["notes"])


def _content_ids() -> set[str]:
    folder = files("kneebench.content").joinpath("rules")
    return {e.name.removesuffix(".yaml") for e in folder.iterdir() if e.name.endswith(".yaml")}


def test_every_rule_has_content_and_content_has_a_rule() -> None:
    _load_rules()
    assert set(REGISTRY) | set(COMPARE_REGISTRY) == _content_ids()


def test_every_rule_cites_its_source_note() -> None:
    _load_rules()
    for r in (*REGISTRY.values(), *COMPARE_REGISTRY.values()):
        assert r.source[:2] in _notes(), f"{r.id} cites {r.source!r}"


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
        assert spec.get("learn_more", {}).get("note") in _notes()
        for action in spec.get("actions", []):
            assert action["when"] and action["title"] and action["verify"]
            assert action.get("kind", "test") in {"server", "test"}, f"{rid}: {action['title']}"


NOTES_DIR = Path(__file__).resolve().parents[3] / "inference-engg" / "notes"


def _github_anchor(heading: str) -> str:
    """How GitHub turns a markdown heading into a link anchor."""
    text = heading.strip().lower()
    text = "".join(c for c in text if c.isalnum() or c in " -_")
    return text.replace(" ", "-")


@pytest.mark.skipif(not NOTES_DIR.is_dir(), reason="lesson notes repo not checked out next door")
def test_learn_more_anchors_exist_in_the_lessons() -> None:
    notes = yaml.safe_load(files("kneebench.content").joinpath("notes.yaml").read_text())["notes"]
    folder = files("kneebench.content").joinpath("rules")
    broken = []
    for rid in sorted(_content_ids()):
        lm = yaml.safe_load(folder.joinpath(f"{rid}.yaml").read_text()).get("learn_more", {})
        if not lm.get("anchor"):
            continue
        lines = (NOTES_DIR / f"{notes[str(lm['note'])]['slug']}.md").read_text().splitlines()
        anchors = {_github_anchor(line.lstrip("#")) for line in lines if line.startswith("#")}
        if lm["anchor"] not in anchors:
            broken.append(f"{rid}: {lm['note']}#{lm['anchor']}")
    assert not broken, broken

"""Content layer lint: generic text, valid links, known glossary terms."""

import re
from importlib.resources import files

import yaml

from kneebench import content
from kneebench.content.render import glossary, ms, num, pct, times
from kneebench.physics.gpus import catalog

# Note quotes may name hardware; nothing else in the content layer may.
ALLOWED_LITERALS = {"B01": ["two H100s"]}
_MODEL_FAMILIES = re.compile(r"\b(llama|qwen|mistral|mixtral|gemma|phi-\d|deepseek)\b", re.I)


def _content_texts() -> dict[str, str]:
    folder = files("kneebench.content")
    texts = {
        e.name.removesuffix(".yaml"): e.read_text()
        for e in folder.joinpath("rules").iterdir()
        if e.name.endswith(".yaml")
    }
    for name in ("report.yaml", "glossary.yaml", "slo_presets.yaml", "profiles.yaml"):
        texts[name] = folder.joinpath(name).read_text()
    return texts


def test_no_hardcoded_gpu_or_model_names() -> None:
    names = {g.name.lower() for g in catalog()} | {
        a for g in catalog() for a in g.aliases if len(a) > 3
    }
    for key, text in _content_texts().items():
        for allowed in ALLOWED_LITERALS.get(key, []):
            text = text.replace(allowed, "")
        lowered = text.lower()
        for n in names:
            assert not re.search(rf"\b{re.escape(n)}\b", lowered), f"{key} mentions {n!r}"
        assert not _MODEL_FAMILIES.search(text), f"{key} names a model family"


def test_term_calls_use_known_glossary_terms() -> None:
    known = {k.lower() for k in glossary()}
    for key, text in _content_texts().items():
        for term in re.findall(r'term\("([^"]+)"\)', text):
            assert term.lower() in known, f"{key} uses unknown term {term!r}"


def test_note_links_resolve_to_known_notes() -> None:
    notes = content.load("notes.yaml")["notes"]
    folder = files("kneebench.content").joinpath("rules")
    for e in folder.iterdir():
        spec = yaml.safe_load(e.read_text())
        assert spec["learn_more"]["note"] in notes


def test_formatters() -> None:
    assert ms(5.83) == "5.8 ms"
    assert ms(0.42) == "0.42 ms"
    assert ms(306.2) == "306 ms"
    assert ms(1840) == "1.84 s"
    assert ms(98_000) == "98.0 s"
    assert ms(None) == "n/a"
    assert pct(0.72) == "72%"
    assert pct(0.059) == "5.9%"
    assert num(532768) == "532,768"
    assert num(65.03, 1) == "65.0"
    assert times(1.36) == "1.4×"
    assert times(65.0) == "65×"

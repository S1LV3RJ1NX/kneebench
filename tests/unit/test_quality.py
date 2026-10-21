"""Quality gate: eval sets, grading, the lesson 54 statistics, and a recorded comparison."""

from __future__ import annotations

import json

import httpx
import pytest

from kneebench.loadgen.base import Endpoint
from kneebench.quality.ask import Answer, ask_all
from kneebench.quality.evalset import EvalSetError, Item, parse, starter
from kneebench.quality.grade import is_correct
from kneebench.quality.service import compare_runs, record
from kneebench.quality.stats import Paired, compare, mcnemar_p
from kneebench.store.db import connect_memory


def _item(answer: str, match: str) -> Item:
    return Item("q", ({"role": "user", "content": "?"},), answer, match, "c")


def test_starter_set_loads_and_every_answer_grades_itself() -> None:
    s = starter()
    assert len(s.items) == 290
    assert s.categories() == ["arithmetic", "extraction", "instructions", "knowledge"]
    assert all(is_correct(i, i.answer) for i in s.items)


def test_parse_rejects_bad_lines() -> None:
    with pytest.raises(EvalSetError, match="line 2"):
        parse("x", '{"prompt": "a", "answer": "1"}\n{"answer": "2"}\n')
    with pytest.raises(EvalSetError, match="match must be"):
        parse("x", '{"prompt": "a", "answer": "1", "match": "vibes"}\n')
    with pytest.raises(EvalSetError, match="duplicate"):
        parse("x", '{"id": "a", "prompt": "a", "answer": "1"}\n' * 2)
    chat = parse(
        "x",
        json.dumps(
            {
                "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                "answer": "1",
            }
        ),
    )
    assert len(chat.items[0].messages) == 2


@pytest.mark.parametrize(
    ("match", "answer", "text", "ok"),
    [
        ("number", "1972", "The answer is 1,972.", True),
        ("number", "1972", "1927", False),
        ("choice", "B", "B", True),
        ("choice", "B", "The answer is B.", True),
        ("choice", "B", "C", False),
        ("exact", "Paris", "paris.", True),
        ("exact", "Paris", "It is Paris", False),
        ("exact_case", "ROCKET", "rocket", False),
        ("contains", "asha.forest@example.com", "Email: asha.forest@example.com", True),
        ("regex", r"^\d{3}$", "123", True),
    ],
)
def test_grading(match: str, answer: str, text: str, ok: bool) -> None:
    assert is_correct(_item(answer, match), text) is ok


def test_paired_statistics_match_lesson_54() -> None:
    p = Paired(n=1000, right_a=720, right_b=712, lost=42, gained=34, changed=90)
    assert p.diff == pytest.approx(-0.008)
    assert p.ci == pytest.approx(0.017, abs=0.0005)
    assert p.unpaired_ci == pytest.approx(0.040, abs=0.001)
    assert p.flip_rate == pytest.approx(0.076)
    assert 0.3 < p.p_value < 0.5  # exact test; the lesson's chi-square gives 0.36
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(0, 12) < 0.001


def _answers(correct: list[bool], texts: list[str] | None = None) -> list[Answer]:
    texts = texts or [str(c) for c in correct]
    return [
        Answer(f"q{i}", "c" if i % 2 else "d", t, c)
        for i, (c, t) in enumerate(zip(correct, texts, strict=True))
    ]


def test_verdict_follows_the_rule() -> None:
    n = 400
    base = _answers([True] * 300 + [False] * 100)
    same = compare(base, base, base)
    assert same.verdict.overall == "pass" and same.overall.flip_rate == 0
    worse = _answers([False] * 60 + [True] * 240 + [False] * 100)
    bad = compare(base, worse, base)
    assert bad.verdict.accuracy == "fail" and bad.verdict.overall == "fail"
    assert bad.overall.diff == pytest.approx(-60 / n)
    few = _answers([False] * 3 + [True] * 297 + [False] * 100)
    shaky = compare(base, few, base, rule={"max_drop_points": 0.5})
    assert shaky.verdict.accuracy == "inconclusive"


def test_record_and_compare_through_the_api() -> None:
    evalset = parse(
        "tiny",
        "\n".join(
            json.dumps(
                {
                    "id": f"q{i}",
                    "prompt": f"What is {i} + {i}?",
                    "answer": str(2 * i),
                    "match": "number",
                    "category": "sum",
                }
            )
            for i in range(20)
        ),
    )
    wrong = {"What is 3 + 3?", "What is 7 + 7?"}

    def server(flip: bool) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["temperature"] == 0 and "authorization" not in request.headers
            q = body["messages"][-1]["content"]
            a, _, b = q.removeprefix("What is ").removesuffix("?").partition(" + ")
            answer = int(a) + int(b) + (1 if flip and q in wrong else 0)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": str(answer)}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 2},
                },
            )

        return httpx.MockTransport(handle)

    conn = connect_memory()
    ep = Endpoint(base_url="http://test", model="m", tokenizer="t")
    before = record(conn, ep, evalset, "before", transport=server(False))
    floor = record(conn, ep, evalset, "floor", transport=server(False))
    after = record(conn, ep, evalset, "after", transport=server(True))
    view = compare_runs(conn, before, after, floor)
    c = view.comparison
    assert (c.overall.n, c.overall.lost, c.overall.gained) == (20, 2, 0)
    assert c.floor is not None and c.floor.flip_rate == 0
    assert [e["kind"] for e in view.examples] == ["lost", "lost"]
    assert view.examples[0]["expected"] == "6" and view.examples[0]["after"] == "7"


def test_ask_all_reports_errors_per_question() -> None:
    evalset = parse("e", json.dumps({"prompt": "hi", "answer": "x"}))
    ep = Endpoint(base_url="http://test", model="m", tokenizer="t", api_key="secret")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(503)

    [a] = ask_all(ep, evalset, transport=httpx.MockTransport(handle))
    assert a.error == "HTTP 503" and not a.correct

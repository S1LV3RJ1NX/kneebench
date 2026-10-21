"""The abandoned-request probe and the retrieval prompts it sits beside."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from kneebench.loadgen.aiperf.parse import CANCELLED_ERROR, RECORDS_FILE, all_cancelled
from kneebench.loadgen.texts import RAG_INSTRUCTIONS, rag_prompts
from kneebench.phases.plan import CANCEL_PHASE, plan_run
from kneebench.phases.runner import _after_cancel
from tests.builders import theory_for
from tests.unit.test_run_lifecycle import _inputs


def test_probe_waits_for_full_answers_then_reads_counters() -> None:
    plan = plan_run(theory_for(), ("baseline", "concurrency", "cancel"))
    planned = next(p for p in plan.points if p.phase == CANCEL_PHASE)
    planned = replace(planned, spec=replace(planned.spec, request_count=16))
    slept: list[float] = []
    probe = _after_cancel(
        planned,
        {"generated": 100.0, "abort": 0.0},
        [],
        _inputs(),
        lambda _e: {"generated": 2_500.0, "abort": 16.0},
        None,
        sleep=slept.append,
    )
    assert probe is not None
    # No sweep measured: 10 ms per token for a 1,000-token answer, plus 2 s.
    assert slept == [12.0] and probe["linger_s"] == 12.0
    assert (probe["requests"], probe["osl"], probe["cancel_after_s"]) == (16, 1000, 1.0)
    assert probe["after"]["abort"] == 16.0
    gone = _after_cancel(planned, {}, [], _inputs(), lambda _e: None, None, sleep=slept.append)
    assert gone is None


def test_ranked_passages_share_less_than_sorted() -> None:
    def shared_start(prompts: list[str]) -> int:
        a, b = prompts[0], prompts[1]
        return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), len(a))

    ranked, ordered = rag_prompts("ranked", 4, seed=1), rag_prompts("sorted", 4, seed=1)
    assert len(ranked) == 4 and ranked[0] != ranked[1]
    words_shared = len(ordered[0][: shared_start(ordered)].split())
    assert words_shared > RAG_INSTRUCTIONS + 5 * 300  # everything up to the question
    assert shared_start(ranked) < shared_start(ordered)
    assert rag_prompts("ranked", 4, seed=1) == ranked  # deterministic


def test_all_cancelled_only_when_every_error_is_a_cancellation(tmp_path: Path) -> None:
    def write(*errors: str | None) -> Path:
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        lines = [json.dumps({"error": {"type": e} if e else None}) for e in errors]
        (d / RECORDS_FILE).write_text("\n".join(lines) + "\n")
        return d

    assert all_cancelled(write(CANCELLED_ERROR, CANCELLED_ERROR))
    assert not all_cancelled(write(CANCELLED_ERROR, None))
    assert not all_cancelled(write(CANCELLED_ERROR, "ConnectionError"))
    assert not all_cancelled(tmp_path / "missing")

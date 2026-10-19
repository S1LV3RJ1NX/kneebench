"""'Try these next': server changes first, test fixes second, one step per setting."""

from kneebench.content.render import RenderedAction, RenderedFinding
from kneebench.report.composer import _next_steps


def _finding(rule_id: str, *actions: RenderedAction) -> RenderedFinding:
    return RenderedFinding(
        rule_id=rule_id,
        status="warn",
        severity=2,
        tags=(),
        source="27#4",
        title=rule_id,
        status_line=f"{rule_id} line",
        measured=None,
        meaning=None,
        analogy=None,
        why=None,
        learn_more_url=None,
        learn_more_label=None,
        actions=actions,
        table=None,
    )


def _action(title: str, flag: str | None, kind: str) -> RenderedAction:
    return RenderedAction(title, flag, None, None, "rerun", kind)


def test_server_changes_come_first_and_duplicates_merge() -> None:
    problems = [
        _finding("H03", _action("Run closer", None, "test")),
        _finding("W06", _action("Lower the cap", "--max-num-seqs 118", "server")),
        _finding("S04", _action("Cap long answers", "--max-num-seqs 66", "server")),
        _finding("R03", _action("Run closer", None, "test")),
        _finding("S05", _action("Rerun with your size", "kneebench run --shapes 2000/500", "test")),
        _finding("C03", _action("Sweep higher", "kneebench run --max-concurrency 512", "test")),
    ]
    steps = _next_steps(problems)
    assert [(s.action.kind, s.rank, s.rule_id) for s in steps] == [
        ("server", 1, "W06"),  # S04's different --max-num-seqs is dropped
        ("test", 1, "H03"),  # R03 repeats the same advice
        ("test", 2, "S05"),
        ("test", 3, "C03"),
    ]

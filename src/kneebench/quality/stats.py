"""Paired comparison of two passes over the same questions (lesson 54 sections 4, 5 and 7)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kneebench.quality.ask import Answer
from kneebench.quality.grade import normalize

Z95 = 1.96
# Lesson 54 section 7: an example rule written before the run.
DEFAULT_RULE = {"max_drop_points": 1.0, "max_flip_excess_points": 2.0}
CATEGORY_P = 0.05


@dataclass(frozen=True)
class Paired:
    n: int
    right_a: int
    right_b: int
    lost: int  # right before, wrong after
    gained: int  # wrong before, right after
    changed: int  # answer text differs at all

    @property
    def acc_a(self) -> float:
        return self.right_a / self.n if self.n else 0.0

    @property
    def acc_b(self) -> float:
        return self.right_b / self.n if self.n else 0.0

    @property
    def diff(self) -> float:
        return (self.gained - self.lost) / self.n if self.n else 0.0

    @property
    def ci(self) -> float:
        """95% half-width of the paired difference: the questions both agree on cancel."""
        if not self.n:
            return 0.0
        var = (self.lost + self.gained) / self.n - self.diff**2
        return Z95 * math.sqrt(max(var, 0.0) / self.n)

    @property
    def unpaired_ci(self) -> float:
        """What the same scores would carry treated as two separate samples."""
        if not self.n:
            return 0.0
        a, b = self.acc_a, self.acc_b
        return Z95 * math.sqrt((a * (1 - a) + b * (1 - b)) / self.n)

    @property
    def flips(self) -> int:
        return self.lost + self.gained

    @property
    def flip_rate(self) -> float:
        return self.flips / self.n if self.n else 0.0

    @property
    def changed_rate(self) -> float:
        return self.changed / self.n if self.n else 0.0

    @property
    def p_value(self) -> float:
        return mcnemar_p(self.lost, self.gained)


def mcnemar_p(lost: int, gained: int) -> float:
    """Exact two-sided McNemar test: under no change, each flip goes either way at 50%."""
    flips = lost + gained
    if flips == 0:
        return 1.0
    k = min(lost, gained)
    tail = sum(math.comb(flips, i) for i in range(k + 1)) / 2**flips
    return float(min(1.0, 2 * tail))


def pair(a: list[Answer], b: list[Answer]) -> tuple[Paired, list[tuple[Answer, Answer]]]:
    """Questions both passes answered without an error, in a's order."""
    by_id = {x.item_id: x for x in b}
    pairs = [
        (x, by_id[x.item_id])
        for x in a
        if x.item_id in by_id and x.error is None and by_id[x.item_id].error is None
    ]
    return _count(pairs), pairs


def _count(pairs: list[tuple[Answer, Answer]]) -> Paired:
    return Paired(
        n=len(pairs),
        right_a=sum(x.correct for x, _ in pairs),
        right_b=sum(y.correct for _, y in pairs),
        lost=sum(x.correct and not y.correct for x, y in pairs),
        gained=sum(y.correct and not x.correct for x, y in pairs),
        changed=sum(normalize(x.text or "") != normalize(y.text or "") for x, y in pairs),
    )


@dataclass(frozen=True)
class Verdict:
    accuracy: str  # pass, fail or inconclusive
    flips: str | None  # pass or fail against the floor; None without a floor
    categories: tuple[str, ...]  # categories with a significant drop
    overall: str


@dataclass(frozen=True)
class Comparison:
    overall: Paired
    floor: Paired | None
    categories: dict[str, Paired]
    floor_categories: dict[str, Paired]
    rule: dict[str, float]
    verdict: Verdict
    errors_before: int
    errors_after: int
    pairs: list[tuple[Answer, Answer]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        def p(x: Paired | None) -> dict[str, Any] | None:
            if x is None:
                return None
            return {
                "n": x.n,
                "acc_a": x.acc_a,
                "acc_b": x.acc_b,
                "diff": x.diff,
                "ci": x.ci,
                "unpaired_ci": x.unpaired_ci,
                "lost": x.lost,
                "gained": x.gained,
                "flip_rate": x.flip_rate,
                "changed_rate": x.changed_rate,
                "p_value": x.p_value,
            }

        return {
            "overall": p(self.overall),
            "floor": p(self.floor),
            "categories": {k: p(v) for k, v in self.categories.items()},
            "rule": self.rule,
            "verdict": vars(self.verdict) | {"categories": list(self.verdict.categories)},
            "errors_before": self.errors_before,
            "errors_after": self.errors_after,
        }


def compare(
    before: list[Answer],
    after: list[Answer],
    floor: list[Answer] | None = None,
    rule: dict[str, float] | None = None,
) -> Comparison:
    rule = {**DEFAULT_RULE, **(rule or {})}
    overall, pairs = pair(before, after)
    floor_p, floor_pairs = pair(before, floor) if floor else (None, [])
    categories = _by_category(pairs)
    floor_categories = _by_category(floor_pairs)
    return Comparison(
        overall=overall,
        floor=floor_p,
        categories=categories,
        floor_categories=floor_categories,
        rule=rule,
        verdict=_verdict(overall, floor_p, categories, rule),
        errors_before=sum(a.error is not None for a in before),
        errors_after=sum(a.error is not None for a in after),
        pairs=pairs,
    )


def _by_category(pairs: list[tuple[Answer, Answer]]) -> dict[str, Paired]:
    groups: dict[str, list[tuple[Answer, Answer]]] = {}
    for x, y in pairs:
        groups.setdefault(x.category, []).append((x, y))
    return {k: _count(v) for k, v in sorted(groups.items())}


def _verdict(
    overall: Paired, floor: Paired | None, categories: dict[str, Paired], rule: dict[str, float]
) -> Verdict:
    budget = rule["max_drop_points"] / 100
    low, high = overall.diff - overall.ci, overall.diff + overall.ci
    # Lesson 54 section 7: the drop must be within budget at 95% confidence.
    accuracy = "pass" if low >= -budget else "fail" if high < -budget else "inconclusive"
    flips = None
    if floor is not None:
        excess = overall.flip_rate - floor.flip_rate
        flips = "pass" if excess <= rule["max_flip_excess_points"] / 100 else "fail"
    dropped = tuple(
        k for k, c in categories.items() if c.lost > c.gained and c.p_value < CATEGORY_P
    )
    if "fail" in (accuracy, flips) or dropped:
        overall_verdict = "fail"
    elif accuracy == "inconclusive":
        overall_verdict = "inconclusive"
    else:
        overall_verdict = "pass"
    return Verdict(accuracy, flips, dropped, overall_verdict)

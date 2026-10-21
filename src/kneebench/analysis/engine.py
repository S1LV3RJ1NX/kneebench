"""Rule registry and evaluation. Rules are pure functions of a RunContext."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from kneebench.analysis.context import RunContext

# Bump when a rule's logic or threshold changes, so cached findings are recomputed.
RULES_VERSION = "1"


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    BAD = "bad"
    INFO = "info"
    SKIP = "skip"  # could not be checked; data says why


SEVERITY = {Status.BAD: 3, Status.WARN: 2, Status.INFO: 1, Status.OK: 0, Status.SKIP: 0}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    status: Status
    data: dict[str, Any] = field(default_factory=dict)
    tags: frozenset[str] = frozenset()

    @property
    def severity(self) -> int:
        return SEVERITY[self.status]

    def to_json(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "status": self.status.value,
            "severity": self.severity,
            "data": self.data,
            "tags": sorted(self.tags),
        }


RuleFn = Callable[[RunContext], Finding | None]


@dataclass(frozen=True)
class Rule:
    id: str
    family: str
    tags: frozenset[str]
    source: str  # lesson note and section, e.g. "24#4"
    fn: RuleFn


REGISTRY: dict[str, Rule] = {}


def rule(rule_id: str, *, tags: set[str], source: str) -> Callable[[RuleFn], RuleFn]:
    def register(fn: RuleFn) -> RuleFn:
        if rule_id in REGISTRY:
            raise ValueError(f"duplicate rule id {rule_id}")
        REGISTRY[rule_id] = Rule(rule_id, rule_id[0], frozenset(tags), source, fn)
        return fn

    return register


def evaluate(ctx: RunContext) -> list[Finding]:
    _load_rules()
    findings: list[Finding] = []
    for r in sorted(REGISTRY.values(), key=lambda r: r.id):
        found = r.fn(ctx)
        if found is not None:
            findings.append(Finding(found.rule_id, found.status, found.data, r.tags | found.tags))
    return findings


def _load_rules() -> None:
    from kneebench.analysis.rules import (  # noqa: F401
        baseline,
        concurrency,
        hygiene,
        interference,
        rate,
        scheduler,
        shape,
        theory,
        walls,
    )


def ok(rule_id: str, **data: Any) -> Finding:
    return Finding(rule_id, Status.OK, data)


def warn(rule_id: str, **data: Any) -> Finding:
    return Finding(rule_id, Status.WARN, data)


def bad(rule_id: str, **data: Any) -> Finding:
    return Finding(rule_id, Status.BAD, data)


def info(rule_id: str, **data: Any) -> Finding:
    return Finding(rule_id, Status.INFO, data)


def skip(rule_id: str, reason: str, **data: Any) -> Finding:
    return Finding(rule_id, Status.SKIP, {"reason": reason, **data})

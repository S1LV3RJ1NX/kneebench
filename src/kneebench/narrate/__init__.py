"""Seam for an optional later layer that rewords findings in friendlier language.

The core is deterministic: a narrator may only change wording, never numbers, statuses or
order. The default passes the report through untouched.
"""

from __future__ import annotations

from typing import Protocol

from kneebench.report.composer import Report


class Narrator(Protocol):
    def rewrite(self, report: Report) -> Report: ...


class PassthroughNarrator:
    def rewrite(self, report: Report) -> Report:
        return report

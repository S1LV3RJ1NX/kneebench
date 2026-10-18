"""JSON report: every rendered sentence plus the raw finding data behind it."""

from __future__ import annotations

import json
from dataclasses import asdict

from kneebench.report.composer import Report


def render_json(report: Report) -> str:
    return json.dumps(asdict(report), indent=2, default=str, ensure_ascii=False)

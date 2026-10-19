"""HTML report: the web page, and a self-contained file for PDF export."""

from __future__ import annotations

import base64
import json
import re
from functools import cache
from importlib.resources import files
from typing import Any

import jinja2

from kneebench.physics.gpus import catalog
from kneebench.report.charts import charts
from kneebench.report.composer import Report

_FONT_URL = re.compile(r'url\("fonts/(?P<name>[\w.-]+\.woff2)"\)')


@cache
def templates() -> jinja2.Environment:
    loader = jinja2.FileSystemLoader(str(files("kneebench.web").joinpath("templates")))
    return jinja2.Environment(loader=loader, autoescape=True, undefined=jinja2.StrictUndefined)


@cache
def plotly_js() -> str:
    from plotly.offline import get_plotlyjs

    js: str = get_plotlyjs()
    return js


@cache
def inline_css() -> str:
    """The stylesheet with its fonts embedded, so exported files render the same offline."""
    static = files("kneebench.web").joinpath("static")
    css = static.joinpath("kb.css").read_text()

    def embed(match: re.Match[str]) -> str:
        data = base64.b64encode(static.joinpath("fonts", match["name"]).read_bytes()).decode()
        return f'url("data:font/woff2;base64,{data}")'

    return _FONT_URL.sub(embed, css)


def report_context(
    report: Report, inline: bool, query: str = "", demo: bool = False
) -> dict[str, Any]:
    if report.style != "html":
        raise ValueError("compose the report with style='html' for HTML output")
    figures = charts(report)
    return {
        "report": report,
        "figures": figures,
        # "</" can't appear inside a <script> block.
        "figures_json": json.dumps(figures).replace("</", "<\\/"),
        "gpus": catalog(),
        "inline_assets": inline,
        "nav_active": None,
        "inline_css": inline_css() if inline else "",
        "plotly_js": plotly_js() if inline else "",
        # The what-if query, carried into export links so a download matches the page.
        "export_query": f"?{query}" if query else "",
        "has_estimates": any(f["rule_id"] == "S01" for f in report.findings_json),
        "is_demo": demo,
    }


def render_html(report: Report, inline: bool = False, query: str = "", demo: bool = False) -> str:
    """inline=True embeds CSS and Plotly so the file works offline and prints to PDF."""
    return (
        templates()
        .get_template("report.html")
        .render(**report_context(report, inline, query, demo))
    )

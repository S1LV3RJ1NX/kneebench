"""HTML report: the web page, and a self-contained file for PDF export."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

import jinja2

from kneebench.physics.gpus import catalog
from kneebench.report.charts import charts
from kneebench.report.composer import Report


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
    return files("kneebench.web").joinpath("static", "kb.css").read_text()


def report_context(report: Report, inline: bool) -> dict[str, Any]:
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
        "inline_css": inline_css() if inline else "",
        "plotly_js": plotly_js() if inline else "",
    }


def render_html(report: Report, inline: bool = False) -> str:
    """inline=True embeds CSS and Plotly so the file works offline and prints to PDF."""
    return templates().get_template("report.html").render(**report_context(report, inline))

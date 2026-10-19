"""HTTP routes. Binds to localhost by default: the new-run form carries API keys."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from sqlite3 import Connection
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from kneebench import content
from kneebench.config import load_local_config
from kneebench.content.render import ms, num, pct
from kneebench.detect.server import detect_server
from kneebench.domain import ProductType
from kneebench.physics.gpus import catalog, get_gpu
from kneebench.report.build import build_report
from kneebench.report.pdf import PdfUnavailableError, attach, html_to_pdf
from kneebench.report.render_html import plotly_js, render_html, templates
from kneebench.report.render_json import render_json
from kneebench.report.render_md import render_markdown
from kneebench.service import PrepareError, RunRequest, correct_gpu, prepare
from kneebench.store import repo
from kneebench.store.db import DB_FILENAME, connect
from kneebench.web.manager import BusyError, RunManager


def create_app(home: Path, manager: RunManager | None = None) -> FastAPI:
    app = FastAPI(title="kneebench", docs_url=None, redoc_url=None)
    runs = manager or RunManager(home)

    @contextmanager
    def db() -> Iterator[Connection]:
        conn = connect(home / DB_FILENAME)
        try:
            yield conn
        finally:
            conn.close()

    def page(name: str, **ctx: Any) -> HTMLResponse:
        return HTMLResponse(templates().get_template(name).render(inline_assets=False, **ctx))

    @app.get("/static/plotly.min.js")
    def plotly() -> Response:
        return Response(
            plotly_js(),
            media_type="application/javascript",
            headers={"Cache-Control": "max-age=86400"},
        )

    @app.get("/", response_class=HTMLResponse)
    def home_page() -> HTMLResponse:
        with db() as conn:
            rows = [dict(r) for r in repo.list_runs(conn)]
        active = runs.active.run_id if runs.busy and runs.active else None
        return page("home.html", runs=rows, active=active)

    @app.get("/runs/new", response_class=HTMLResponse)
    def new_run() -> HTMLResponse:
        profiles = content.load("profiles.yaml")["profiles"]
        presets = content.load("slo_presets.yaml")["presets"]
        return page(
            "new.html",
            gpus=catalog(),
            profiles=profiles,
            presets=presets,
            presets_json=json.dumps(presets),
            defaults={"url": load_local_config().url},
        )

    @app.post("/detect", response_class=HTMLResponse)
    async def detect(
        url: Annotated[str, Form()], api_key: Annotated[str, Form()] = ""
    ) -> HTMLResponse:
        snap = await run_in_threadpool(detect_server, url, api_key or None)
        if not snap.models:
            return page(
                "_detected.html",
                error=f"Couldn't list models at {url}. Check the URL "
                "and that this machine can reach it.",
                snap=snap,
                model=None,
            )
        return page("_detected.html", snap=snap, model=snap.models[0], error=None)

    def run_request(form: dict[str, str]) -> RunRequest:
        def opt_int(key: str) -> int | None:
            return int(form[key]) if form.get(key) else None

        def opt_float(key: str) -> float | None:
            return float(form[key]) if form.get(key) else None

        eager = form.get("enforce_eager", "")
        return RunRequest(
            url=form["url"],
            gpu=get_gpu(form["gpu"]),
            api_key=form.get("api_key") or None,
            gpu_count=opt_int("gpu_count") or 1,
            model=form.get("model") or None,
            tokenizer=form.get("tokenizer") or None,
            max_num_seqs=opt_int("max_num_seqs"),
            enforce_eager=None if eager == "" else eager == "true",
            product=ProductType(form.get("product", "chat")),
            slo_ttft_ms=opt_float("slo_ttft_ms"),
            slo_tpot_ms=opt_float("slo_tpot_ms"),
            slo_percentile=opt_int("slo_percentile"),
            max_concurrency=opt_int("max_concurrency") or 256,
            budget_s=(opt_float("budget_min") or 0) * 60 or None,
            label=form.get("label") or None,
        )

    async def form_dict(request: Request) -> dict[str, str]:
        return {k: str(v) for k, v in (await request.form()).items()}

    @app.post("/plan", response_class=HTMLResponse)
    async def plan_preview(request: Request) -> HTMLResponse:
        try:
            _, plan = await run_in_threadpool(prepare, run_request(await form_dict(request)))
        except (PrepareError, ValueError, KeyError) as exc:
            return page("_plan.html", error=str(exc), plan=None, adapted=[])
        return page("_plan.html", plan=plan, error=None, adapted=[])

    @app.post("/runs")
    async def start_run(request: Request) -> Response:
        try:
            inputs, plan = await run_in_threadpool(prepare, run_request(await form_dict(request)))
            run_id = await run_in_threadpool(runs.start, inputs, plan)
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (PrepareError, ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse(f"/runs/{run_id}/live", status_code=303)

    @app.get("/runs/{run_id}/live", response_class=HTMLResponse)
    def live(run_id: int) -> HTMLResponse:
        return page("live.html", run_id=run_id)

    @app.get("/runs/{run_id}/progress", response_class=HTMLResponse)
    def progress(run_id: int) -> HTMLResponse:
        with db() as conn:
            run = repo.get_run(conn, run_id)
            if run is None:
                raise HTTPException(404)
            rows = [
                _progress_row(repo.point_dict(conn, r["id"]))
                for r in conn.execute(
                    "SELECT id FROM points WHERE run_id = ? ORDER BY seq", (run_id,)
                )
            ]
        state = runs.active if runs.active and runs.active.run_id == run_id else None
        finished = run["status"] != "running" and (state is None or state.finished)
        total = state.total if state else len(rows)
        done = state.done if state else len(rows)
        return page(
            "_progress.html",
            run_id=run_id,
            rows=rows,
            finished=finished,
            status=run["status"],
            total=total,
            done=done,
            pct=round(100 * done / total) if total else 0,
            current=state.current if state else None,
            stopped=state.stopped if state else None,
            error=state.error if state else None,
        )

    def _report(run_id: int, style: str) -> Any:
        with db() as conn:
            if repo.get_run(conn, run_id) is None:
                raise HTTPException(404)
            return build_report(conn, run_id, style=style)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def report_page(run_id: int) -> HTMLResponse:
        return HTMLResponse(render_html(_report(run_id, "html")))

    @app.get("/runs/{run_id}/report.md")
    def report_md(run_id: int) -> Response:
        return Response(
            render_markdown(_report(run_id, "md")),
            media_type="text/markdown",
            headers=_download(f"kneebench-run-{run_id}.md"),
        )

    @app.get("/runs/{run_id}/report.json")
    def report_json(run_id: int) -> Response:
        return Response(
            render_json(_report(run_id, "md")),
            media_type="application/json",
            headers=_download(f"kneebench-run-{run_id}.json"),
        )

    @app.get("/runs/{run_id}/report.pdf")
    async def report_pdf(run_id: int) -> Response:
        html = render_html(_report(run_id, "html"), inline=True)
        md = render_markdown(_report(run_id, "md"))
        js = render_json(_report(run_id, "md"))
        try:
            pdf = await run_in_threadpool(html_to_pdf, html)
        except PdfUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        pdf = attach(pdf, {"report.md": md.encode(), "report.json": js.encode()})
        return Response(
            pdf, media_type="application/pdf", headers=_download(f"kneebench-run-{run_id}.pdf")
        )

    @app.post("/runs/{run_id}/gpu")
    def correct_gpu_route(run_id: int, gpu: Annotated[str, Form()]) -> Response:
        with db() as conn:
            correct_gpu(conn, run_id, gpu, None)
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    # Mounted last so the generated /static/plotly.min.js route above takes precedence.
    static = files("kneebench.web").joinpath("static")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    return app


def _download(name: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{name}"'}


def _progress_row(p: dict[str, Any]) -> dict[str, Any]:
    metrics, server = p["metrics"], p["server"]
    kv = server.get("vllm:kv_cache_usage_perc", {}).get("max")
    pre = server.get("vllm:num_preemptions", {}).get("total")
    return {
        "seq": p["seq"] + 1,
        "phase": p["phase"],
        "concurrency": p["concurrency"],
        "status": p["status"],
        "tok_s": num(metrics.get("output_token_throughput", {}).get("avg"), 0),
        "tpot": ms(metrics.get("inter_token_latency", {}).get("p99")),
        "ttft": ms(metrics.get("time_to_first_token", {}).get("p99")),
        "kv": pct(kv) if kv is not None else "n/a",
        "preempt": num(pre, 0) if pre is not None else "n/a",
    }

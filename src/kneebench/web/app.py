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
from kneebench.doctor import exit_code, run_doctor
from kneebench.domain import ProductType
from kneebench.physics.gpus import catalog, get_gpu
from kneebench.report.build import build_report
from kneebench.report.pdf import PdfUnavailableError, attach, html_to_pdf, install_chromium
from kneebench.report.render_html import plotly_js, render_html, templates
from kneebench.report.render_json import render_json
from kneebench.report.render_md import render_markdown
from kneebench.runlock import holder
from kneebench.service import ALL_PHASES, PrepareError, RunRequest, correct_gpu, prepare
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
        nav_active = holder(home)
        if nav_active is None and runs.busy and runs.active:
            nav_active = {"run_id": runs.active.run_id, "source": "web UI"}
        return HTMLResponse(
            templates().get_template(name).render(inline_assets=False, nav_active=nav_active, **ctx)
        )

    def notice(message: str, kind: str = "bad") -> HTMLResponse:
        return page("_notice.html", message=message, kind=kind)

    @app.get("/static/plotly.min.js")
    def plotly() -> Response:
        return Response(
            plotly_js(),
            media_type="application/javascript",
            headers={"Cache-Control": "max-age=86400"},
        )

    def live_run_ids() -> set[int]:
        ids = set()
        lock = holder(home)
        if lock and lock.get("run_id"):
            ids.add(int(lock["run_id"]))
        if runs.busy and runs.active and runs.active.run_id:
            ids.add(runs.active.run_id)
        return ids

    @app.get("/", response_class=HTMLResponse)
    def home_page() -> HTMLResponse:
        live_ids = live_run_ids()
        with db() as conn:
            rows = [dict(r) for r in repo.list_runs(conn)]
        for r in rows:
            # "running" with no live process holding the lock means the process died.
            if r["status"] == "running" and r["id"] not in live_ids:
                r["status"] = "interrupted"
        return page("home.html", runs=rows, active=holder(home))

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
        if not url.strip():
            return page(
                "_detected.html", error="Enter the endpoint URL first.", snap=None, model=None
            )
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

        if not form.get("url", "").strip():
            raise PrepareError("Enter the endpoint URL.")
        phases = tuple(p for p in ALL_PHASES if form.get(f"phase_{p}"))
        if not phases:
            raise PrepareError("Choose at least one phase to run.")
        eager = form.get("enforce_eager", "")
        return RunRequest(
            url=form["url"].strip(),
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
            phases=phases,
            max_concurrency=opt_int("max_concurrency") or 256,
            budget_s=(opt_float("budget_min") or 0) * 60 or None,
            label=form.get("label") or None,
        )

    async def form_dict(request: Request) -> dict[str, str]:
        return {k: str(v) for k, v in (await request.form()).items()}

    def explain(exc: Exception) -> str:
        if isinstance(exc, PrepareError | BusyError):
            return str(exc)
        return f"Couldn't prepare the run: {type(exc).__name__}: {exc}"

    @app.post("/plan", response_class=HTMLResponse)
    async def plan_preview(request: Request) -> HTMLResponse:
        try:
            _, plan = await run_in_threadpool(prepare, run_request(await form_dict(request)))
        except Exception as exc:
            return page("_plan.html", error=explain(exc), plan=None)
        return page("_plan.html", plan=plan, error=None)

    @app.post("/runs")
    async def start_run(request: Request) -> Response:
        htmx = request.headers.get("HX-Request") == "true"
        try:
            if runs.busy or holder(home):
                raise BusyError(
                    "A benchmark is already running. Wait for it, or stop it "
                    "from its progress page."
                )
            inputs, plan = await run_in_threadpool(prepare, run_request(await form_dict(request)))
            run_id = await run_in_threadpool(runs.start, inputs, plan)
        except Exception as exc:
            if htmx:
                return notice(explain(exc))
            raise HTTPException(400, explain(exc)) from exc
        target = f"/runs/{run_id}/live"
        if htmx:
            return Response(status_code=204, headers={"HX-Redirect": target})
        return RedirectResponse(target, status_code=303)

    @app.get("/runs/{run_id}/live", response_class=HTMLResponse)
    def live(run_id: int) -> HTMLResponse:
        return page("live.html", run_id=run_id)

    @app.post("/runs/{run_id}/stop", response_class=HTMLResponse)
    def stop(run_id: int) -> HTMLResponse:
        if runs.stop(run_id):
            return notice("Stopping after the current AIPerf process exits…", "warn")
        return notice("This run isn't running in this window.")

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
        state = runs.state_for(run_id)
        finished = run["status"] != "running" and (state is None or state.finished)
        total = state.total if state else len(rows)
        done = state.done if state else len(rows)
        remaining = None
        if state and not state.finished:
            mins = state.remaining_s / 60
            remaining = f"{mins:.0f} min" if mins >= 1 else "under a minute"
        return page(
            "_progress.html",
            run_id=run_id,
            rows=rows,
            finished=finished,
            status=run["status"],
            total=total,
            done=done,
            pct=round(100 * done / total) if total else 0,
            remaining=remaining,
            current=state.current if state else None,
            stopped=state.stopped if state else None,
            error=state.error if state else None,
            managed=state is not None,
            can_stop=state is not None and not state.cancel.is_set(),
            cancelling=state is not None and state.cancel.is_set(),
        )

    def _report(run_id: int, style: str) -> Any:
        with db() as conn:
            if repo.get_run(conn, run_id) is None:
                raise HTTPException(404)
            return build_report(conn, run_id, style=style)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def report_page(run_id: int) -> HTMLResponse:
        return HTMLResponse(render_html(_report(run_id, "html")))

    @app.get("/runs/{run_id}/report.html")
    def report_html(run_id: int) -> Response:
        return Response(
            render_html(_report(run_id, "html"), inline=True),
            media_type="text/html",
            headers=_download(f"kneebench-run-{run_id}.html"),
        )

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
        md_report = _report(run_id, "md")
        try:
            pdf = await run_in_threadpool(html_to_pdf, html)
        except PdfUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        pdf = attach(
            pdf,
            {
                "report.md": render_markdown(md_report).encode(),
                "report.json": render_json(md_report).encode(),
            },
        )
        return Response(
            pdf, media_type="application/pdf", headers=_download(f"kneebench-run-{run_id}.pdf")
        )

    @app.post("/runs/{run_id}/gpu")
    def correct_gpu_route(run_id: int, gpu: Annotated[str, Form()]) -> Response:
        with db() as conn:
            correct_gpu(conn, run_id, gpu, None)
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/doctor", response_class=HTMLResponse)
    def doctor_page() -> HTMLResponse:
        return page("doctor.html", url=load_local_config().url)

    @app.post("/doctor", response_class=HTMLResponse)
    async def doctor_run(
        url: Annotated[str, Form()] = "",
        api_key: Annotated[str, Form()] = "",
        tokenizer: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        checks = await run_in_threadpool(
            run_doctor, False, url.strip() or None, api_key or None, tokenizer.strip() or None
        )
        return page("_doctor.html", checks=checks, failed=exit_code(checks) != 0)

    @app.post("/doctor/install-pdf", response_class=HTMLResponse)
    async def doctor_install_pdf() -> HTMLResponse:
        code = await run_in_threadpool(install_chromium)
        if code == 0:
            return notice("Installed. PDF export is ready.", "ok")
        return notice(
            "The download failed. Run `kneebench doctor --install-pdf` in a terminal to see why."
        )

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


__all__ = ["create_app"]

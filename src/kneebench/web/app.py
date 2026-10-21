"""HTTP routes. Binds to localhost by default: the new-run form carries API keys."""

from __future__ import annotations

import json
import os
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from sqlite3 import Connection
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from kneebench import content, experiments
from kneebench.analysis.context import context_from_bundle
from kneebench.analysis.whatif import estimate
from kneebench.config import load_local_config
from kneebench.content.render import ms, num, pct
from kneebench.demo import demo_run_id
from kneebench.demo import seed as demo_seed
from kneebench.detect.server import detect_server
from kneebench.doctor import exit_code, run_doctor
from kneebench.domain import ProductType, Slo, user_shapes
from kneebench.loadgen.base import Endpoint
from kneebench.physics.gpus import catalog, get_gpu
from kneebench.quality import service as qservice
from kneebench.quality import stats as qstats
from kneebench.quality import store as qstore
from kneebench.quality.evalset import EvalSetError
from kneebench.quality.evalset import parse as parse_evalset
from kneebench.quality.evalset import starter as starter_evalset
from kneebench.report.build import build_report
from kneebench.report.compare import build_comparison, render_comparison_md
from kneebench.report.pdf import PdfUnavailableError, attach, html_to_pdf, install_chromium
from kneebench.report.render_html import plotly_js, render_html, templates
from kneebench.report.render_json import render_json
from kneebench.report.render_md import render_markdown
from kneebench.runlock import holder
from kneebench.service import (
    ALL_PHASES,
    PrepareError,
    RunRequest,
    correct_gpu,
    delete_run_data,
    prepare,
)
from kneebench.store import repo
from kneebench.store.db import DB_FILENAME, connect
from kneebench.web.manager import BusyError, RunManager


def create_app(home: Path, manager: RunManager | None = None, demo: bool = True) -> FastAPI:
    app = FastAPI(title="kneebench", docs_url=None, redoc_url=None)
    runs = manager or RunManager(home)

    @contextmanager
    def db() -> Iterator[Connection]:
        conn = connect(home / DB_FILENAME)
        try:
            yield conn
        finally:
            conn.close()

    if demo and os.environ.get("KNEEBENCH_NO_DEMO") != "1":
        with db() as conn:
            demo_seed(conn, home)

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

    def reconcile(conn: Connection) -> None:
        # A run still "running" with no live process behind it was killed or crashed.
        repo.mark_interrupted(conn, live_run_ids())

    @app.get("/", response_class=HTMLResponse)
    def home_page() -> HTMLResponse:
        with db() as conn:
            reconcile(conn)
            rows = [dict(r) for r in repo.list_runs(conn)]
        return page("home.html", runs=rows, active=holder(home), demo_id=demo_run_id(home))

    def comparison(a: int, b: int) -> Any:
        if a == b:
            raise HTTPException(400, "Pick two different runs.")
        with db() as conn:
            try:
                return build_comparison(conn, a, b, style="html")
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(a: int, b: int) -> HTMLResponse:
        c = comparison(a, b)
        figures = json.dumps(c.figures).replace("</", "<\\/")
        return page("compare.html", c=c, figures_json=figures)

    @app.get("/compare.md")
    def compare_md(a: int, b: int) -> Response:
        return Response(
            render_comparison_md(comparison(a, b)),
            media_type="text/markdown",
            headers=_download(f"kneebench-compare-{a}-{b}.md"),
        )

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
        isl, osl = form.get("shape_isl", "").strip(), form.get("shape_osl", "").strip()
        if bool(isl) != bool(osl):
            raise PrepareError("Give both prompt and output tokens for your typical request.")
        try:
            shapes = user_shapes(f"{isl}/{osl}") if isl else ()
        except ValueError as exc:
            raise PrepareError(str(exc)) from exc
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
            shapes=shapes,
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
            if repo.get_run(conn, run_id) is None:
                raise HTTPException(404)
            reconcile(conn)
            prog = repo.progress(conn, run_id)
            rows = [
                _progress_row(repo.point_dict(conn, r["id"]))
                for r in conn.execute(
                    "SELECT id FROM points WHERE run_id = ? ORDER BY seq", (run_id,)
                )
            ]
        state = runs.state_for(run_id)  # only this window's runs can be stopped from here
        finished = prog["status"] != "running"
        total, done = prog["total"], prog["done"]
        remaining = None
        if state and not state.finished:
            mins = state.remaining_s / 60
            remaining = f"{mins:.0f} min" if mins >= 1 else "under a minute"
        return page(
            "_progress.html",
            run_id=run_id,
            rows=rows,
            finished=finished,
            status=prog["status"],
            total=total,
            done=done,
            pct=round(100 * done / total) if total else 0,
            remaining=remaining,
            current=prog["current"],
            stopped=state.stopped if state else None,
            error=prog["error"] or (state.error if state else None),
            managed=state is not None,
            can_stop=state is not None and not state.cancel.is_set(),
            cancelling=state is not None and state.cancel.is_set(),
        )

    @app.post("/runs/{run_id}/delete")
    def delete_run(run_id: int) -> Response:
        if run_id in live_run_ids():
            raise HTTPException(409, "Stop the run before deleting it.")
        with db() as conn:
            if repo.get_run(conn, run_id) is None:
                raise HTTPException(404)
            delete_run_data(conn, home, run_id)
        return RedirectResponse("/", status_code=303)

    def what_if_slo(request: Request) -> Slo | None:
        """Latency targets from the report's what-if form; an empty box means no target."""
        q = request.query_params
        if not q.get("whatif"):
            return None

        def number(key: str) -> float | None:
            try:
                value = float(q.get(key) or "")
            except ValueError:
                return None
            return value if value > 0 else None

        pct = q.get("slo_percentile")
        return Slo(number("slo_ttft_ms"), number("slo_tpot_ms"), 90 if pct == "90" else 99)

    def what_if_query(request: Request) -> str:
        keys = ("whatif", "slo_ttft_ms", "slo_tpot_ms", "slo_percentile")
        q = request.query_params
        return urlencode({k: q[k] for k in keys if k in q}) if q.get("whatif") else ""

    def _report(run_id: int, style: str, request: Request | None = None) -> Any:
        with db() as conn:
            if repo.get_run(conn, run_id) is None:
                raise HTTPException(404)
            slo = what_if_slo(request) if request else None
            return build_report(conn, run_id, style=style, slo=slo)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def report_page(run_id: int, request: Request) -> HTMLResponse:
        with db() as conn:
            verifying = repo.experiment_for_verify_run(conn, run_id)
        return HTMLResponse(
            render_html(
                _report(run_id, "html", request),
                query=what_if_query(request),
                demo=run_id == demo_run_id(home),
                experiment=dict(verifying) if verifying else None,
            )
        )

    # ---- guided experiments ----

    def experiment_or_404(conn: Connection, experiment_id: int) -> Any:
        row = repo.get_experiment(conn, experiment_id)
        if row is None:
            raise HTTPException(404)
        return row

    @app.post("/runs/{run_id}/experiments")
    def start_experiment(run_id: int, step: Annotated[int, Form()]) -> Response:
        with db() as conn:
            if repo.get_run(conn, run_id) is None:
                raise HTTPException(404)
            try:
                experiment_id = experiments.propose(conn, run_id, step)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
        return RedirectResponse(f"/experiments/{experiment_id}", status_code=303)

    @app.get("/experiments", response_class=HTMLResponse)
    def experiments_page() -> HTMLResponse:
        with db() as conn:
            rows = [dict(r) for r in repo.list_experiments(conn)]
        return page("experiments.html", experiments=rows)

    @app.get("/experiments/{experiment_id}", response_class=HTMLResponse)
    def experiment_page(experiment_id: int) -> HTMLResponse:
        with db() as conn:
            e = experiment_or_404(conn, experiment_id)
            verify = repo.get_run(conn, e["verify_run_id"]) if e["verify_run_id"] else None
            comparison = None
            before = experiments.base_value(conn, e)
            if verify is not None and verify["status"] == "done" and e["base_run_id"]:
                comparison = build_comparison(
                    conn,
                    e["base_run_id"],
                    verify["id"],
                    style="html",
                    declared=experiments.declared_change(e, before),
                )
                if e["status"] in ("verifying", "applied", "proposed"):
                    repo.update_experiment(
                        conn, experiment_id, status="verified", outcome=" ".join(comparison.summary)
                    )
                    e = experiment_or_404(conn, experiment_id)
            model = None
            if e["base_run_id"]:
                bundle = repo.run_bundle(conn, e["base_run_id"])
                model = bundle["snapshot"]["hf_model_id"] or bundle["snapshot"]["served_model"]
            q = quality_state(conn, e)
        style = e["deploy_style"] or "cli"
        return page(
            "experiment.html",
            **q,
            e=dict(e),
            before=before,
            styles=experiments.DEPLOY_STYLES,
            style=style,
            how=experiments.snippet(style, e["flag"], model),
            check=json.loads(e["check_json"]) if e["check_json"] else None,
            error=None,
            verify_run=dict(verify) if verify else None,
            comparison=comparison,
        )

    @app.post("/experiments/{experiment_id}/style")
    def experiment_style(experiment_id: int, style: Annotated[str, Form()]) -> Response:
        if style not in experiments.DEPLOY_STYLES:
            raise HTTPException(400, "unknown deployment style")
        with db() as conn:
            e = experiment_or_404(conn, experiment_id)
            repo.set_deploy_style(conn, e["endpoint_id"], style)
        return RedirectResponse(f"/experiments/{experiment_id}", status_code=303)

    @app.post("/experiments/{experiment_id}/check", response_class=HTMLResponse)
    async def experiment_check(
        experiment_id: int, api_key: Annotated[str, Form()] = ""
    ) -> HTMLResponse:
        with db() as conn:
            e = experiment_or_404(conn, experiment_id)
            base_text = None
            model = None
            if e["base_run_id"]:
                snap_row = repo.run_bundle(conn, e["base_run_id"])["snapshot"]
                if snap_row.get("metrics_text_z"):
                    base_text = zlib.decompress(snap_row["metrics_text_z"]).decode()
                model = snap_row["served_model"]
            before = experiments.base_value(conn, e)
        try:
            snap = await run_in_threadpool(detect_server, e["base_url"], api_key or None)
        except Exception as exc:
            return page("_check.html", error=f"Couldn't reach the server: {exc}", e=e, check=None)
        check = experiments.applied_check(base_text, before, snap, e["flag"], model)
        with db() as conn:
            status = (
                "applied" if check["reachable"] and check["landed"] is not False else e["status"]
            )
            if e["status"] in ("verifying", "verified", "kept", "reverted"):
                status = e["status"]
            repo.update_experiment(conn, experiment_id, check_json=json.dumps(check), status=status)
        return page("_check.html", error=None, e=e, check=check)

    @app.post("/experiments/{experiment_id}/verify")
    async def experiment_verify(
        experiment_id: int, api_key: Annotated[str, Form()] = ""
    ) -> Response:
        try:
            if runs.busy or holder(home):
                raise BusyError("A benchmark is already running. Wait for it to finish.")
            with db() as conn:
                e = experiment_or_404(conn, experiment_id)
                req = experiments.verification_request(conn, e, api_key or None)
            inputs, plan = await run_in_threadpool(prepare, req)
            run_id = await run_in_threadpool(runs.start, inputs, plan)
        except HTTPException:
            raise
        except Exception as exc:
            return notice(explain(exc))
        with db() as conn:
            repo.update_experiment(conn, experiment_id, verify_run_id=run_id, status="verifying")
        return Response(status_code=204, headers={"HX-Redirect": f"/runs/{run_id}/live"})

    def quality_state(conn: Connection, e: Any) -> dict[str, Any]:
        passes = qstore.for_experiment(conn, e["id"])
        out: dict[str, Any] = {
            "q_before": None, "q_floor": None, "q_floor_flips": None, "quality": None,
            "quality_summary": [], "q_error": None,
        }  # fmt: skip
        if "before" not in passes:
            return out
        before = dict(qstore.get_run(conn, passes["before"]) or {})
        before["questions"] = before.get("set_items")
        rows = qstore.answers(conn, passes["before"])
        before["accuracy"] = sum(a.correct for a in rows) / len(rows) if rows else 0.0
        out["q_before"] = before
        if "floor" in passes:
            out["q_floor"] = dict(qstore.get_run(conn, passes["floor"]) or {})
            floor_pair, _ = qstats.pair(rows, qstore.answers(conn, passes["floor"]))
            out["q_floor_flips"] = floor_pair.flip_rate
        if "after" in passes:
            rule = json.loads(e["quality_rule_json"]) if e["quality_rule_json"] else None
            view = qservice.compare_runs(
                conn, passes["before"], passes["after"], passes.get("floor"), rule
            )
            out["quality"] = view
            out["quality_summary"] = qservice.summary(view)
        return out

    def experiment_endpoint(conn: Connection, e: Any, api_key: str | None) -> Endpoint:
        if not e["base_run_id"]:
            raise HTTPException(400, "the run that proposed this change was deleted")
        snap = repo.run_bundle(conn, e["base_run_id"])["snapshot"]
        return Endpoint(
            base_url=e["base_url"], model=snap["served_model"], tokenizer="", api_key=api_key
        )

    def quality_busy() -> str | None:
        if runs.busy or holder(home):
            return "A benchmark is running; asking now would disturb it. Wait for it to finish."
        return None

    @app.post("/experiments/{experiment_id}/quality/before", response_class=HTMLResponse)
    async def quality_before(
        experiment_id: int,
        evalset: Annotated[str, Form()] = "starter",
        file: Annotated[UploadFile | None, File()] = None,
        max_drop: Annotated[float, Form()] = 1.0,
        max_flip_excess: Annotated[float, Form()] = 2.0,
        api_key: Annotated[str, Form()] = "",
    ) -> Response:
        if busy := quality_busy():
            return page("_quality.html", q_error=busy, quality=None)
        try:
            if evalset == "file":
                if file is None or not file.filename:
                    raise EvalSetError("choose a JSONL file, or use the starter set")
                chosen = parse_evalset(file.filename, (await file.read()).decode())
            else:
                chosen = starter_evalset()
        except (EvalSetError, UnicodeDecodeError) as exc:
            return page(
                "_quality.html", q_error=f"That eval set can't be used: {exc}", quality=None
            )
        if max_drop <= 0 or max_flip_excess < 0:
            return page(
                "_quality.html", q_error="The allowed drop must be above zero.", quality=None
            )
        rule = {"max_drop_points": max_drop, "max_flip_excess_points": max_flip_excess}
        with db() as conn:
            e = experiment_or_404(conn, experiment_id)
            endpoint = experiment_endpoint(conn, e, api_key or None)

        def job() -> None:
            with db() as conn:
                qservice.record_before(conn, endpoint, chosen, experiment_id, rule)

        try:
            await run_in_threadpool(job)
        except Exception as exc:
            return page("_quality.html", q_error=explain(exc), quality=None)
        return Response(status_code=204, headers={"HX-Redirect": f"/experiments/{experiment_id}"})

    @app.post("/experiments/{experiment_id}/quality/after", response_class=HTMLResponse)
    async def quality_after(experiment_id: int, api_key: Annotated[str, Form()] = "") -> Response:
        if busy := quality_busy():
            return page("_quality.html", q_error=busy, quality=None)
        with db() as conn:
            e = experiment_or_404(conn, experiment_id)
            passes = qstore.for_experiment(conn, experiment_id)
            if "before" not in passes:
                raise HTTPException(400, "record answers before the change first")
            before = qstore.get_run(conn, passes["before"])
            assert before is not None
            chosen = qstore.load_set(conn, before["set_id"])
            endpoint = experiment_endpoint(conn, e, api_key or None)

        def job() -> None:
            with db() as conn:
                qservice.record(conn, endpoint, chosen, "after", experiment_id)

        try:
            await run_in_threadpool(job)
        except Exception as exc:
            return page("_quality.html", q_error=explain(exc), quality=None)
        return Response(status_code=204, headers={"HX-Redirect": f"/experiments/{experiment_id}"})

    @app.post("/experiments/{experiment_id}/decide")
    def experiment_decide(experiment_id: int, decision: Annotated[str, Form()]) -> Response:
        if decision not in ("kept", "reverted"):
            raise HTTPException(400, "decide kept or reverted")
        with db() as conn:
            experiment_or_404(conn, experiment_id)
            repo.update_experiment(conn, experiment_id, status=decision)
        return RedirectResponse(f"/experiments/{experiment_id}", status_code=303)

    @app.get("/runs/{run_id}/estimate", response_class=HTMLResponse)
    def estimate_route(run_id: int, isl: int, osl: int) -> HTMLResponse:
        with db() as conn:
            if repo.get_run(conn, run_id) is None:
                raise HTTPException(404)
            ctx = context_from_bundle(repo.run_bundle(conn, run_id))
        if isl <= 0 or osl <= 0:
            return page("_estimate.html", error="Enter positive token counts.", e=None)
        e = estimate(ctx, isl, osl)
        if e is None:
            return page(
                "_estimate.html",
                error="This run has no request-size sweep to estimate from.",
                e=None,
            )
        return page("_estimate.html", error=None, e=e, ms=ms)

    @app.get("/runs/{run_id}/report.html")
    def report_html(run_id: int, request: Request) -> Response:
        return Response(
            render_html(_report(run_id, "html", request), inline=True),
            media_type="text/html",
            headers=_download(f"kneebench-run-{run_id}.html"),
        )

    @app.get("/runs/{run_id}/report.md")
    def report_md(run_id: int, request: Request) -> Response:
        return Response(
            render_markdown(_report(run_id, "md", request)),
            media_type="text/markdown",
            headers=_download(f"kneebench-run-{run_id}.md"),
        )

    @app.get("/runs/{run_id}/report.json")
    def report_json(run_id: int, request: Request) -> Response:
        return Response(
            render_json(_report(run_id, "md", request)),
            media_type="application/json",
            headers=_download(f"kneebench-run-{run_id}.json"),
        )

    @app.get("/runs/{run_id}/report.pdf")
    async def report_pdf(run_id: int, request: Request) -> Response:
        html = render_html(_report(run_id, "html", request), inline=True)
        md_report = _report(run_id, "md", request)
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

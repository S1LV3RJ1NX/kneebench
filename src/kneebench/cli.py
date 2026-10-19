"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kneebench import __version__
from kneebench.config import api_key_from_env, data_dir, load_local_config
from kneebench.doctor import Check, Status, exit_code, run_doctor

app = typer.Typer(
    name="kneebench",
    help="Benchmark an existing LLM inference server and get a report you can act on.",
    no_args_is_help=False,
    add_completion=False,
)
console = Console()

_STATUS_LABEL = {
    Status.OK: "[green]  ok[/]",
    Status.WARN: "[yellow]warn[/]",
    Status.FAIL: "[red]fail[/]",
    Status.SKIP: "[dim]skip[/]",
}


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context) -> None:
    """With no command, start the web UI."""
    if ctx.invoked_subcommand is None:
        ui()


@app.command()
def ui(
    host: Annotated[
        str, typer.Option(help="Interface to bind; keep localhost unless you know who can reach it")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open a browser")] = True,
) -> None:
    """Start the web UI."""
    import threading
    import webbrowser

    import uvicorn

    from kneebench.web.app import create_app

    url = f"http://{host}:{port}"
    console.print(f"kneebench is running at [bold]{url}[/] (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(data_dir()), host=host, port=port, log_level="warning")


@app.command()
def version() -> None:
    """Print kneebench and bundled AIPerf versions."""
    from kneebench.loadgen.aiperf.version import AIPERF_PIN, installed_aiperf_version

    console.print(
        f"kneebench {__version__} (aiperf {installed_aiperf_version()}, pin {AIPERF_PIN})"
    )


@app.command()
def doctor(
    url: Annotated[
        str | None, typer.Option(help="Endpoint base URL, e.g. http://host:8000")
    ] = None,
    tokenizer: Annotated[
        str | None, typer.Option(help="Hugging Face tokenizer id or local path")
    ] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Only check the local install; no network.")
    ] = False,
    install_pdf: Annotated[
        bool, typer.Option("--install-pdf", help="Download the headless Chromium for PDF export")
    ] = False,
) -> None:
    """Check Python, the bundled AIPerf, the tokenizer download and the endpoint."""
    if install_pdf:
        from kneebench.report.pdf import install_chromium

        raise typer.Exit(install_chromium())
    local = load_local_config()
    url = url or local.url
    tokenizer = tokenizer or local.tokenizer

    def show(check: Check) -> None:
        console.print(f"{_STATUS_LABEL[check.status]}  [bold]{check.name}[/]: {check.detail}")
        if check.fix and check.status is not Status.OK:
            console.print(f"      [dim]fix:[/] {check.fix}")

    checks = run_doctor(
        offline=offline, url=url, api_key=api_key_from_env(), tokenizer=tokenizer, on_check=show
    )
    code = exit_code(checks)
    console.print(
        "\n[green]All required checks passed.[/]" if code == 0 else "\n[red]Some checks failed.[/]"
    )
    raise typer.Exit(code)


@app.command()
def detect(
    url: Annotated[str | None, typer.Option(help="Endpoint base URL")] = None,
) -> None:
    """Show what kneebench can detect about an endpoint (read-only)."""
    from kneebench.detect.server import detect_server

    url = url or load_local_config().url
    if not url:
        raise typer.BadParameter("pass --url or set it in kneebench.local.toml")
    snap = detect_server(url, api_key_from_env())
    _print_detected(snap)


def _print_detected(snap: object) -> None:
    from kneebench.detect.server import ServerSnapshot

    assert isinstance(snap, ServerSnapshot)
    table = Table(title="Detected", show_header=False, box=None)
    table.add_row("Endpoint", snap.base_url)
    table.add_row("vLLM version", snap.vllm_version or "[yellow]not reported[/]")
    for m in snap.models:
        table.add_row("Served model", f"{m.id} (max_model_len {m.max_model_len}, HF id {m.hf_id})")
    if snap.cache:
        c = snap.cache
        table.add_row(
            "KV cache",
            f"{c.kv_cache_size_tokens:,} tokens, block {c.block_size}, "
            f"dtype {c.cache_dtype}, prefix caching {c.enable_prefix_caching}",
        )
        table.add_row("gpu_memory_utilization", str(c.gpu_memory_utilization))
    else:
        table.add_row("KV cache", "[yellow]not visible (/metrics unavailable)[/]")
    if snap.network:
        n = snap.network
        table.add_row(
            "Network",
            f"TCP {n.tcp_connect_ms and round(n.tcp_connect_ms)} ms, "
            f"TLS {n.tls_handshake_ms and round(n.tls_handshake_ms)} ms, "
            f"HTTP round trip {n.http_rtt_ms and round(n.http_rtt_ms)} ms",
        )
    console.print(table)


@app.command()
def run(
    gpu: Annotated[str, typer.Option(help="GPU id from the catalog (see `kneebench gpus`)")],
    url: Annotated[str | None, typer.Option(help="Endpoint base URL")] = None,
    model: Annotated[str | None, typer.Option(help="Served model name (default: first)")] = None,
    tokenizer: Annotated[str | None, typer.Option(help="HF id or path (default: detected)")] = None,
    gpu_count: Annotated[int, typer.Option(help="GPUs per replica (tensor parallel size)")] = 1,
    max_num_seqs: Annotated[
        int | None, typer.Option(help="Server's --max-num-seqs, if set")
    ] = None,
    enforce_eager: Annotated[
        bool | None,
        typer.Option(
            "--enforce-eager/--no-enforce-eager",
            help="Whether the server runs with --enforce-eager",
        ),
    ] = None,
    product: Annotated[str, typer.Option(help="chat, rag, reasoning or batch")] = "chat",
    slo_ttft_ms: Annotated[float | None, typer.Option(help="TTFT target (ms)")] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option(help="TPOT target (ms)")] = None,
    label: Annotated[str | None, typer.Option(help="A note to tell runs apart")] = None,
    phases: Annotated[
        str, typer.Option(help="Comma-separated: baseline, concurrency")
    ] = "baseline,concurrency",
    max_concurrency: Annotated[int, typer.Option(help="Highest concurrency to sweep")] = 256,
    budget_min: Annotated[
        float | None, typer.Option(help="Time budget in minutes; trims the top of the sweep")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation")] = False,
) -> None:
    """Detect the server, run the benchmark phases and write a report."""
    from kneebench.detect.server import detect_server
    from kneebench.domain import ProductType
    from kneebench.loadgen.aiperf.runner import AiperfLoadGenerator
    from kneebench.phases.runner import PointEvent, execute_run
    from kneebench.physics.gpus import get_gpu
    from kneebench.service import PrepareError, RunRequest, prepare
    from kneebench.store.db import DB_FILENAME, connect

    local = load_local_config()
    url = url or local.url
    if not url:
        raise typer.BadParameter("pass --url or set it in kneebench.local.toml")
    req = RunRequest(
        url=url,
        gpu=get_gpu(gpu),
        api_key=api_key_from_env(),
        gpu_count=gpu_count,
        model=model or local.model,
        tokenizer=tokenizer or local.tokenizer,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        product=ProductType(product),
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        phases=tuple(p.strip() for p in phases.split(",") if p.strip()),
        max_concurrency=max_concurrency,
        budget_s=budget_min * 60 if budget_min else None,
        label=label,
    )

    console.print("[bold]Detecting the server…[/]")
    snap = detect_server(url, req.api_key)
    _print_detected(snap)
    try:
        inputs, plan = prepare(req, snapshot=snap)
    except PrepareError as exc:
        raise typer.BadParameter(str(exc)) from exc

    levels = [p.spec.concurrency for p in plan.points if p.phase == "concurrency"]
    console.print(
        f"\nPlan: {len(plan.points)} points, about {plan.eta_s / 60:.0f} minutes "
        f"(includes ~8 s AIPerf startup per point)."
        + (f"\nConcurrency sweep: {', '.join(map(str, levels))}." if levels else "")
        + (
            f"\nDropped for the time budget: {plan.dropped_for_budget}."
            if plan.dropped_for_budget
            else ""
        )
        + "\nThe sweep stops early once the walls are found."
    )
    if not yes and not typer.confirm("Start?", default=True):
        raise typer.Exit(1)

    home = data_dir()
    conn = connect(home / DB_FILENAME)

    def progress(ev: PointEvent) -> None:
        spec = ev.planned.spec
        if ev.skipped:
            console.print(
                f"[{ev.index + 1}/{ev.total}] {ev.planned.phase} c={spec.concurrency}: "
                f"[dim]skipped ({ev.skipped})[/]"
            )
        elif ev.result is None:
            console.print(
                f"[{ev.index + 1}/{ev.total}] {ev.planned.phase}: {spec.isl} in / "
                f"{spec.osl} out, concurrency {spec.concurrency}, "
                f"~{ev.planned.eta_s:.0f} s…"
            )
        else:
            state = "[green]done[/]" if ev.result.ok else f"[red]failed ({ev.result.exit_code})[/]"
            extra = ""
            if ev.summary and ev.summary.output_tok_s:
                s = ev.summary
                extra = (
                    f": {s.output_tok_s:,.0f} tok/s, TPOT P99 "
                    f"{s.tpot.get('p99', 0):.1f} ms, TTFT P99 {s.ttft.get('p99', 0):.0f} ms"
                    + (f", preemptions {s.preemptions:.0f}" if s.preemptions else "")
                )
            console.print(f"      {state} in {ev.result.wall_time_s:.0f} s{extra}")

    run_id = execute_run(conn, inputs, plan, AiperfLoadGenerator(), home, on_event=progress)
    _write_report(conn, run_id, None, "md", show_verdict=True)


@app.command()
def runs() -> None:
    """List recent runs."""
    from kneebench.store import repo
    from kneebench.store.db import DB_FILENAME, connect

    conn = connect(data_dir() / DB_FILENAME)
    table = Table("Run", "Started", "Status", "Model", "Endpoint", "Label")
    for r in repo.list_runs(conn):
        table.add_row(
            str(r["id"]),
            r["started_at"],
            r["status"],
            r["served_model"] or "",
            r["base_url"],
            r["label"] or "",
        )
    console.print(table)


@app.command()
def report(
    run_id: Annotated[int, typer.Argument(help="Run id (see `kneebench runs`)")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="md, json, html or pdf")] = "md",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="File to write")] = None,
    gpu: Annotated[
        str | None, typer.Option(help="Correct the GPU entry (catalog id); no rerun needed")
    ] = None,
    gpu_count: Annotated[int | None, typer.Option(help="Correct the GPU count")] = None,
) -> None:
    """Regenerate a report from stored data (no network)."""
    from kneebench.store.db import DB_FILENAME, connect

    conn = connect(data_dir() / DB_FILENAME)
    if gpu or gpu_count:
        _correct_gpu(conn, run_id, gpu, gpu_count)
    _write_report(conn, run_id, output, fmt, show_verdict=gpu is not None or gpu_count is not None)


def _correct_gpu(conn: object, run_id: int, gpu: str | None, gpu_count: int | None) -> None:
    from sqlite3 import Connection

    from kneebench.service import correct_gpu

    assert isinstance(conn, Connection)
    spec, count = correct_gpu(conn, run_id, gpu, gpu_count)
    console.print(f"Run {run_id} now uses {count}× {spec.name}.")


@app.command()
def gpus() -> None:
    """List GPUs in the catalog."""
    from kneebench.physics.gpus import catalog

    table = Table("id", "Name", "Memory (GiB)", "Bandwidth (TB/s)", "BF16 TFLOPS")
    for g in catalog():
        table.add_row(
            g.id, g.name, f"{g.memory_gib:.1f}", f"{g.bandwidth_tbps:g}", f"{g.bf16_tflops:g}"
        )
    console.print(table)


def _write_report(
    conn: object, run_id: int, output: Path | None, fmt: str, show_verdict: bool
) -> None:
    from sqlite3 import Connection

    from kneebench.report.build import build_report
    from kneebench.report.render_html import render_html
    from kneebench.report.render_json import render_json
    from kneebench.report.render_md import render_markdown

    assert isinstance(conn, Connection)
    if fmt not in ("md", "json", "html", "pdf"):
        raise typer.BadParameter("format must be md, json, html or pdf")
    rep = build_report(conn, run_id)
    path = output or data_dir() / "reports" / f"run-{run_id}.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pdf":
        from kneebench.report.pdf import PdfUnavailableError, attach, html_to_pdf

        html = render_html(build_report(conn, run_id, style="html"), inline=True)
        try:
            pdf = html_to_pdf(html)
        except PdfUnavailableError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        path.write_bytes(
            attach(
                pdf,
                {
                    "report.md": render_markdown(rep).encode(),
                    "report.json": render_json(rep).encode(),
                },
            )
        )
    elif fmt == "html":
        path.write_text(render_html(build_report(conn, run_id, style="html"), inline=True))
    else:
        path.write_text(render_json(rep) if fmt == "json" else render_markdown(rep))
    if show_verdict:
        console.print("\n[bold]Verdict[/]\n" + "\n".join(rep.verdict))
    console.print(f"\nReport written to {path}")


def main() -> None:
    app()

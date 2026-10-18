"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kneebench import __version__, content
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
PHASES = ("baseline",)


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(
            "The kneebench web UI arrives in milestone M2. For now: "
            "[bold]kneebench doctor[/], [bold]kneebench run[/], [bold]kneebench report[/]."
        )


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
) -> None:
    """Check Python, the bundled AIPerf, the tokenizer download and the endpoint."""
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
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation")] = False,
) -> None:
    """Detect the server, run the benchmark phases and write a report."""
    from kneebench.detect.server import detect_server, fetch_model_metadata
    from kneebench.domain import STANDARD_SHAPES, ProductType, ServerSetup, Slo, Source, Workload
    from kneebench.loadgen.aiperf.runner import AiperfLoadGenerator
    from kneebench.loadgen.base import Endpoint
    from kneebench.phases.plan import plan_run
    from kneebench.phases.runner import PointEvent, RunInputs, execute_run
    from kneebench.phases.theory import compute_theory
    from kneebench.physics.floors import Hardware
    from kneebench.physics.gpus import get_gpu
    from kneebench.physics.model import model_spec_from_hf
    from kneebench.store.db import DB_FILENAME, connect

    local = load_local_config()
    url = url or local.url
    if not url:
        raise typer.BadParameter("pass --url or set it in kneebench.local.toml")
    api_key = api_key_from_env()
    gpu_spec = get_gpu(gpu)
    product_type = ProductType(product)

    console.print("[bold]Detecting the server…[/]")
    snap = detect_server(url, api_key)
    _print_detected(snap)
    served = snap.model(model or local.model)
    if served is None:
        raise typer.BadParameter(f"model {model!r} not served; found {[m.id for m in snap.models]}")
    hf_id = tokenizer or local.tokenizer or served.hf_id
    if not hf_id:
        raise typer.BadParameter("couldn't map the served model to Hugging Face; pass --tokenizer")
    metadata = fetch_model_metadata(hf_id)
    model_spec = model_spec_from_hf(metadata.config, metadata.safetensors_params, hf_id=hf_id)

    preset = content.load("slo_presets.yaml")["presets"][product_type.value]
    slo = Slo(
        ttft_ms=slo_ttft_ms if slo_ttft_ms is not None else preset["ttft_ms"],
        tpot_ms=slo_tpot_ms if slo_tpot_ms is not None else preset["tpot_ms"],
        percentile=preset["percentile"],
    )
    workload = Workload(product=product_type, shapes=STANDARD_SHAPES, slo=slo)
    setup = ServerSetup(
        gpu=gpu_spec,
        gpu_count=gpu_count,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        sources={
            "gpu": Source.USER,
            "gpu_count": Source.USER,
            "max_num_seqs": Source.USER if max_num_seqs else Source.DEFAULT,
            "enforce_eager": Source.USER if enforce_eager is not None else Source.DEFAULT,
        },
    )
    theory = compute_theory(
        model_spec,
        Hardware(gpu_spec, gpu_count),
        snap.cache,
        served.max_model_len,
        workload.shapes,
        snap.vllm_version,
    )
    plan = plan_run(theory, PHASES)

    console.print(
        f"\nPlan: {len(plan.points)} point(s), about {plan.eta_s / 60:.1f} minutes "
        f"(includes ~8 s AIPerf startup per point)."
    )
    if not yes and not typer.confirm("Start?", default=True):
        raise typer.Exit(1)

    home = data_dir()
    conn = connect(home / DB_FILENAME)
    endpoint = Endpoint(base_url=snap.base_url, model=served.id, tokenizer=hf_id, api_key=api_key)
    inputs = RunInputs(
        endpoint=endpoint,
        snapshot=snap,
        served=served,
        metadata=metadata,
        setup=setup,
        workload=workload,
        theory=theory,
        label=label,
    )

    def progress(ev: PointEvent) -> None:
        if ev.result is None:
            console.print(
                f"[{ev.index + 1}/{ev.total}] {ev.planned.phase}: "
                f"{ev.planned.spec.isl} in / {ev.planned.spec.osl} out, "
                f"concurrency {ev.planned.spec.concurrency}, ~{ev.planned.eta_s:.0f} s…"
            )
        else:
            state = "[green]done[/]" if ev.result.ok else f"[red]failed ({ev.result.exit_code})[/]"
            console.print(f"      {state} in {ev.result.wall_time_s:.0f} s")

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
    fmt: Annotated[str, typer.Option("--format", "-f", help="md or json")] = "md",
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
    import json
    from sqlite3 import Connection

    from kneebench.domain import ServerSetup, Source
    from kneebench.phases.theory import theory_from_snapshot
    from kneebench.physics.floors import Hardware
    from kneebench.physics.gpus import get_gpu, gpu_from_json
    from kneebench.store import repo

    assert isinstance(conn, Connection)
    bundle = repo.run_bundle(conn, run_id)
    old = bundle["setup"]
    spec = get_gpu(gpu) if gpu else gpu_from_json(json.loads(old["gpu_json"]))
    count = gpu_count or old["gpu_count"]
    sources = {**json.loads(old["source_json"]), "gpu": Source.USER, "gpu_count": Source.USER}
    setup = ServerSetup(
        gpu=spec,
        gpu_count=count,
        max_num_seqs=old["max_num_seqs"],
        enforce_eager=None if old["enforce_eager"] is None else bool(old["enforce_eager"]),
        sources=sources,
    )
    theory = theory_from_snapshot(bundle["snapshot"], bundle["workload"], Hardware(spec, count))
    repo.correct_setup(conn, run_id, setup, theory.to_json())
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
    from kneebench.report.render_json import render_json
    from kneebench.report.render_md import render_markdown

    assert isinstance(conn, Connection)
    rep = build_report(conn, run_id)
    text = render_json(rep) if fmt == "json" else render_markdown(rep)
    path = output or data_dir() / "reports" / f"run-{run_id}.{'json' if fmt == 'json' else 'md'}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if show_verdict:
        console.print("\n[bold]Verdict[/]\n" + "\n".join(rep.verdict))
    console.print(f"\nReport written to {path}")


def main() -> None:
    app()

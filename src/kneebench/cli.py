"""Command-line entry point."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from kneebench import __version__
from kneebench.config import api_key_from_env, load_local_config
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
    if ctx.invoked_subcommand is None:
        console.print(
            "The kneebench web UI arrives in milestone M2. For now, run "
            "[bold]kneebench doctor[/] to check your setup."
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


def main() -> None:
    app()

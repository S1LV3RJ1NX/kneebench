"""`kneebench quality`: ask an eval set, list passes, compare two of them."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kneebench.config import api_key_from_env, data_dir, load_local_config

app = typer.Typer(help="Check that a change kept answers right (lesson 54).", no_args_is_help=True)
console = Console()


@app.command()
def ask(
    url: Annotated[str | None, typer.Option(help="Endpoint base URL")] = None,
    model: Annotated[str | None, typer.Option(help="Served model name (default: first)")] = None,
    eval_file: Annotated[
        Path | None, typer.Option("--eval", help="JSONL eval set (default: the starter set)")
    ] = None,
    role: Annotated[str, typer.Option(help="before, floor or after")] = "before",
    repeat: Annotated[
        int, typer.Option(help="Passes to record; a second 'before' pass is the noise floor")
    ] = 1,
    label: Annotated[str | None, typer.Option(help="A note to tell passes apart")] = None,
) -> None:
    """Ask every question once at temperature 0 and grade the answers."""
    from kneebench.detect.server import detect_server
    from kneebench.loadgen.base import Endpoint
    from kneebench.quality.evalset import EvalSetError, parse, starter
    from kneebench.quality.service import record
    from kneebench.quality.store import get_run
    from kneebench.runlock import holder
    from kneebench.store.db import DB_FILENAME, connect

    if holder(data_dir()):
        console.print("[red]A benchmark is running; asking now would disturb it. Wait for it.[/]")
        raise typer.Exit(1)
    local = load_local_config()
    url = url or local.url
    if not url:
        raise typer.BadParameter("pass --url or set it in kneebench.local.toml")
    try:
        evalset = parse(eval_file.name, eval_file.read_text()) if eval_file else starter()
    except (OSError, EvalSetError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    api_key = api_key_from_env()
    served = model or local.model
    if not served:
        snap = detect_server(url, api_key)
        if not snap.models:
            raise typer.BadParameter("the server lists no models; pass --model")
        served = snap.models[0].id
    endpoint = Endpoint(base_url=url, model=served, tokenizer="", api_key=api_key)
    conn = connect(data_dir() / DB_FILENAME)
    roles = [role] if repeat == 1 else [role, *(["floor"] * (repeat - 1))]
    for r in roles:
        console.print(f"Asking {len(evalset.items)} questions ({evalset.name}) as '{r}'…")
        run_id = record(conn, endpoint, evalset, r, label=label)
        row = get_run(conn, run_id)
        acc = conn.execute(
            "SELECT AVG(correct), SUM(error IS NOT NULL) FROM quality_answers WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        status = row["status"] if row else "?"
        console.print(
            f"  pass {run_id}: {status}, {acc[0]:.1%} right"
            + (f", {acc[1]} failed to arrive" if acc[1] else "")
        )


@app.command()
def runs() -> None:
    """List recorded passes."""
    from kneebench.quality.store import list_runs
    from kneebench.store.db import DB_FILENAME, connect

    conn = connect(data_dir() / DB_FILENAME)
    table = Table("id", "Role", "Model", "Set", "Right", "Status", "Experiment", "When")
    for r in list_runs(conn):
        table.add_row(
            str(r["id"]), r["role"], r["model"], r["set_name"],
            f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a",
            r["status"], str(r["experiment_id"] or ""), r["created_at"],
        )  # fmt: skip
    console.print(table)


@app.command()
def compare(
    before: Annotated[int, typer.Argument(help="Pass id before the change")],
    after: Annotated[int, typer.Argument(help="Pass id after the change")],
    floor: Annotated[
        int | None, typer.Option(help="A second 'before' pass, for the noise floor")
    ] = None,
    max_drop: Annotated[float, typer.Option(help="Accuracy drop allowed, in points")] = 1.0,
    max_flip_excess: Annotated[
        float, typer.Option(help="Flips allowed above the noise floor, in points")
    ] = 2.0,
    show: Annotated[int, typer.Option(help="Changed answers to print")] = 10,
) -> None:
    """Compare two passes question by question."""
    from kneebench.quality.service import compare_runs, summary
    from kneebench.store.db import DB_FILENAME, connect

    conn = connect(data_dir() / DB_FILENAME)
    rule = {"max_drop_points": max_drop, "max_flip_excess_points": max_flip_excess}
    try:
        view = compare_runs(conn, before, after, floor, rule)
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for line in summary(view):
        console.print(line)
    table = Table("Category", "Questions", "Before", "After", "Lost", "Gained")
    for name, c in view.comparison.categories.items():
        table.add_row(
            name, str(c.n), f"{c.acc_a:.1%}", f"{c.acc_b:.1%}", str(c.lost), str(c.gained)
        )
    console.print(table)
    for e in view.examples[:show]:
        console.print(
            f"\n[bold]{e['kind']}[/] {e['id']} ({e['category']}), expected {e['expected']!r}\n"
            f"  {e['question'][:160]}\n  before: {e['before']!r}\n  after:  {e['after']!r}"
        )

# Contributing to kneebench

Thanks for helping. Bug reports, better explanations and new rules are all welcome.

## Setup

```bash
git clone https://github.com/S1LV3RJ1NX/kneebench && cd kneebench
uv sync
uv run kneebench doctor --offline
```

Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`), never pip, and commit
`uv.lock`.

## Checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest -m "not integration and not live"   # fast, no network
uv run pytest -m integration                       # bundled AIPerf vs a local mock server
```

Integration tests start `guidellm mock-server` through `uvx`, in its own environment.
`live` tests need a real endpoint: put it in `kneebench.local.toml` (gitignored; see
`kneebench.local.toml.example`). Never commit real endpoint URLs or keys, including
in test fixtures.

## Ground rules

- **Analysis is deterministic.** Verdicts and recommendations come from formulas,
  thresholds and templated text. No LLM in the core.
- **Every rule cites its source** (a lesson note and section) and has tests for its
  ok, bad and edge cases.
- **Text lives in the content layer**, not in analysis code, so explanations can be
  reviewed on their own.
- **No hardcoded models, GPUs or endpoints.** kneebench must work for any setup.
- **Small, conventional commits** (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).

## Upgrading AIPerf

Bump `aiperf==` in `pyproject.toml` and `AIPERF_PIN` in
`src/kneebench/loadgen/aiperf/version.py` together, run the full test suite (the
schema test validates generated configs against the new version), re-measure startup
overhead with `scripts/measure_aiperf_startup.py`, and add a CHANGELOG entry.

## Refreshing the demo run

The UI ships with one real run so first-time users see a full report. Refresh it whenever a
new rule or phase needs data the current demo doesn't have:

1. Run a full benchmark: `KNEEBENCH_HOME=$PWD/.kneebench uv run kneebench run --gpu <id> --yes`.
2. Export it: `uv run python scripts/make_demo.py --run <id>`. This replaces the endpoint,
   served model name and local paths with placeholders, refuses to write if any survive,
   checks that the export imports and analyzes, and writes `src/kneebench/content/demo-run.json.gz`.
3. Optionally regenerate the README images: `uv run python scripts/make_screenshots.py ...`.
4. Commit both. Users get the new demo on their next start unless they deleted the old one.

Rule and wording changes alone don't need a refresh: reports are rebuilt from the stored
measurements every time.

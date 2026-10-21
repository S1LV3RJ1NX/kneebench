"""Refresh the demo run shipped with kneebench (dev tool).

    uv run python scripts/make_demo.py --run 1            # from .kneebench/kneebench.sqlite3

Exports one finished run, replaces the endpoint, served model name, server paths and local
paths with placeholders, refuses to write if any of them survive, and saves it as
src/kneebench/content/demo-run.json.gz. Existing users get the new demo on their next start
unless they deleted the old one.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kneebench.demo import FORMAT, export_run, import_run  # noqa: E402
from kneebench.store.db import DB_FILENAME, connect  # noqa: E402

OUT = ROOT / "src" / "kneebench" / "content" / "demo-run.json.gz"
PLACEHOLDER_URL = "http://vllm.example:8000"


def _replace(value: Any, secrets: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in secrets.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(v, secrets) for v in value]
    if isinstance(value, dict):
        return {k: _replace(v, secrets) for k, v in value.items()}
    return value


def sanitize(bundle: dict[str, Any]) -> dict[str, Any]:
    t = bundle["tables"]
    endpoint, snapshot, run = t["endpoints"][0], t["snapshots"][0], t["runs"][0]
    hf_id = snapshot.get("hf_model_id") or "model"
    model = hf_id.split("/")[-1].lower()
    models = json.loads(snapshot.get("models_json") or "[]")

    secrets = {endpoint["base_url"]: PLACEHOLDER_URL}
    host = urlparse(endpoint["base_url"]).hostname
    if host:
        secrets[host] = urlparse(PLACEHOLDER_URL).hostname or "vllm.example"
    for m in models:
        if m.get("root"):
            secrets[m["root"]] = f"/models/{hf_id}"
        if m.get("id") and m["id"] != hf_id:
            secrets[m["id"]] = model
    if snapshot.get("served_model") and snapshot["served_model"] != hf_id:
        secrets[snapshot["served_model"]] = model
    secrets[str(Path.home())] = "~"
    # Longest first, so a URL is replaced before the host inside it.
    secrets = dict(sorted(secrets.items(), key=lambda kv: -len(kv[0])))

    endpoint["name"] = endpoint["base_url"] = PLACEHOLDER_URL
    run["error"] = None
    for p in t["points"]:
        p["artifact_dir"] = None
        p["argv_json"] = "[]"
    clean: dict[str, Any] = _replace(bundle, secrets)

    text = json.dumps(clean)
    leaks = [s for s in secrets if s and s in text]
    if leaks:
        raise SystemExit(f"refusing to write: {len(leaks)} identifying strings survived")
    return clean


def with_current_theory(bundle: dict[str, Any]) -> dict[str, Any]:
    """Recompute the physics from the stored server metadata, so the demo carries today's
    floors and memory math rather than those of the kneebench that measured it."""
    from kneebench.phases.theory import theory_from_snapshot
    from kneebench.physics.floors import Hardware
    from kneebench.physics.gpus import gpu_from_json

    t = bundle["tables"]
    setup = t["setups"][0]
    hw = Hardware(gpu_from_json(json.loads(setup["gpu_json"])), setup["gpu_count"])
    theory = theory_from_snapshot(t["snapshots"][0], t["workloads"][0], hw)
    t["runs"][0]["theory_json"] = json.dumps(theory.to_json())
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run", type=int, required=True, help="run id to ship as the demo")
    parser.add_argument("--db", type=Path, default=ROOT / ".kneebench" / DB_FILENAME)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    conn = connect(args.db)
    run = conn.execute("SELECT status FROM runs WHERE id = ?", (args.run,)).fetchone()
    if run is None or run["status"] != "done":
        raise SystemExit(f"run {args.run} is missing or not finished")
    bundle = sanitize(with_current_theory(export_run(conn, args.run)))

    # Prove it imports and analyzes before shipping it.
    import tempfile

    from kneebench.analysis.context import context_from_bundle
    from kneebench.analysis.engine import evaluate
    from kneebench.store import repo

    with tempfile.TemporaryDirectory() as tmp:
        check = connect(Path(tmp) / DB_FILENAME)
        findings = evaluate(context_from_bundle(repo.run_bundle(check, import_run(check, bundle))))
        check.close()

    body = json.dumps(bundle["tables"], sort_keys=True, separators=(",", ":"))
    out = {
        "format": FORMAT,
        "version": hashlib.sha256(body.encode()).hexdigest()[:12],
        "tables": bundle["tables"],
    }
    args.out.write_bytes(gzip.compress(json.dumps(out, separators=(",", ":")).encode(), mtime=0))
    points = len(bundle["tables"]["points"])
    size_kb = args.out.stat().st_size / 1024
    print(
        f"wrote {args.out.relative_to(ROOT)}: version {out['version']}, {points} points, "
        f"{len(findings)} findings, {size_kb:,.0f} KiB"
    )


if __name__ == "__main__":
    main()

"""Guided tuning: one server change at a time, applied by the user, verified by a short run.

kneebench never touches the server. It proposes a change from a report, shows it in the
syntax of the user's deployment, checks from outside whether the change landed, runs only the
part of the benchmark the change can move, and compares the result with the run that
proposed it (lesson 55: change one thing, measure again, write it down).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any

from kneebench.detect.metrics import parse_samples
from kneebench.detect.server import ServerSnapshot
from kneebench.domain import USER_SHAPE_PREFIX, ProductType, Shape
from kneebench.physics.gpus import gpu_from_json
from kneebench.service import RunRequest
from kneebench.store import repo

DEPLOY_STYLES = {
    "cli": "vllm serve",
    "kubernetes": "Kubernetes or Helm",
    "managed": "A managed platform",
}
# Statuses, in the order an experiment moves through them.
STATUSES = ("proposed", "applied", "verifying", "verified", "kept", "reverted")


@dataclass(frozen=True)
class Change:
    setting: str | None  # e.g. --max-num-seqs
    value: str | None  # None when the flag is removed or takes no value
    removal: bool


def parse_flag(flag: str) -> Change:
    """'--max-num-seqs 118', 'remove --enforce-eager', '--enable-chunked-prefill'."""
    words = flag.split()
    removal = bool(words) and words[0] == "remove"
    setting = next((w for w in words if w.startswith("--")), None)
    if setting and "=" in setting:
        setting, value = setting.split("=", 1)
        return Change(setting, value, removal)
    after = words[words.index(setting) + 1 :] if setting else []
    return Change(setting, " ".join(after) or None, removal)


def snippet(style: str, flag: str, model: str | None) -> dict[str, str]:
    """How to make the change, in the syntax of the user's deployment."""
    change = parse_flag(flag)
    arg = f"{change.setting}={change.value}" if change.value else (change.setting or flag)
    if style == "kubernetes":
        if change.removal:
            return {
                "intro": f"Delete {change.setting} from the vLLM container's args (or the "
                "matching value in your Helm chart), then roll the deployment.",
                "code": f'args:\n  # remove: - "{change.setting}"',
                "lang": "yaml",
            }
        return {
            "intro": "Add this to the vLLM container's args (or the matching value in your "
            "Helm chart), then roll the deployment.",
            "code": f'args:\n  - "{arg}"',
            "lang": "yaml",
        }
    if style == "managed":
        verb = "Remove" if change.removal else "Add"
        return {
            "intro": f"{verb} this in your platform's model settings, where it passes extra "
            "arguments to vLLM (often called engine arguments or extra args), then redeploy.",
            "code": (change.setting or flag) if change.removal else flag,
            "lang": "text",
        }
    command = f"vllm serve {model or '<model>'} \\\n  <your other flags> \\\n  "
    if change.removal:
        return {
            "intro": f"Restart vLLM without {change.setting}.",
            "code": f"vllm serve {model or '<model>'} <your other flags, without {change.setting}>",
            "lang": "bash",
        }
    return {
        "intro": "Restart vLLM with this flag added to your command.",
        "code": command + flag,
        "lang": "bash",
    }


# Settings the endpoint reports, so kneebench can confirm them after the change.
VISIBLE = {
    "--max-model-len": "context window",
    "--gpu-memory-utilization": "GPU memory utilization",
    "--kv-cache-dtype": "KV cache dtype",
    "--enable-prefix-caching": "prefix caching",
    "--block-size": "KV block size",
}


def _visible(snap: ServerSnapshot, setting: str, model: str | None) -> Any:
    served = snap.model(model) if model else (snap.models[0] if snap.models else None)
    cache = snap.cache
    return {
        "--max-model-len": served.max_model_len if served else None,
        "--gpu-memory-utilization": cache.gpu_memory_utilization if cache else None,
        "--kv-cache-dtype": cache.cache_dtype if cache else None,
        "--enable-prefix-caching": cache.enable_prefix_caching if cache else None,
        "--block-size": cache.block_size if cache else None,
    }.get(setting)


def _start_time(metrics_text: str | None) -> float | None:
    if not metrics_text:
        return None
    starts = [
        s.value for s in parse_samples(metrics_text) if s.name == "process_start_time_seconds"
    ]
    return max(starts) if starts else None


def applied_check(
    base_metrics_text: str | None,
    base_value: Any,
    snap: ServerSnapshot,
    flag: str,
    model: str | None,
) -> dict[str, Any]:
    """What the endpoint shows after the user says they applied the change."""
    change = parse_flag(flag)
    before, after = _start_time(base_metrics_text), _start_time(snap.metrics_text)
    restarted = None if before is None or after is None else after > before
    visible = change.setting in VISIBLE if change.setting else False
    now = _visible(snap, change.setting, model) if visible and change.setting else None
    expected = None if change.removal else change.value
    landed = None
    if visible and expected is not None and now is not None:
        landed = str(now).lower() == str(expected).lower()
    elif visible and change.removal is False and expected is None and now is not None:
        landed = bool(now)  # an on/off switch like --enable-prefix-caching
    return {
        "reachable": snap.reachable,
        "restarted": restarted,
        "visible": visible,
        "label": VISIBLE.get(change.setting or "", change.setting),
        "before": base_value,
        "now": now,
        "landed": landed,
        "cache_tokens": snap.cache.kv_cache_size_tokens if snap.cache else None,
    }


def propose(conn: Connection, run_id: int, step: int) -> int:
    """Create an experiment from step `step` (0-based) of a report's server changes."""
    from kneebench.report.build import build_report

    report = build_report(conn, run_id)
    server = [s for s in report.next_steps if s.action.kind == "server"]
    if not 0 <= step < len(server):
        raise KeyError(f"run {run_id} has no server change number {step + 1}")
    chosen = server[step]
    flag = chosen.action.flag or chosen.action.title
    change = parse_flag(flag)
    run = repo.get_run(conn, run_id)
    assert run is not None
    return repo.create_experiment(
        conn,
        endpoint_id=run["endpoint_id"],
        base_run_id=run_id,
        rule_id=chosen.rule_id,
        title=chosen.action.title,
        reason=chosen.reason,
        flag=flag,
        setting=change.setting,
        value=change.value,
        expected=chosen.action.expected,
        risk=chosen.action.risk,
    )


def verification_request(conn: Connection, experiment: Any, api_key: str | None) -> RunRequest:
    """A short run with the base run's setup and targets, over the load the change can move."""
    from kneebench.analysis.context import context_from_bundle
    from kneebench.analysis.rules.walls import capacity

    base_id = experiment["base_run_id"]
    if base_id is None:
        raise ValueError("the run this experiment came from was deleted")
    bundle = repo.run_bundle(conn, base_id)
    ctx = context_from_bundle(bundle)
    setup, workload = bundle["setup"], bundle["workload"]
    slo = json.loads(workload["slo_json"])
    shapes = tuple(
        Shape(**s)
        for s in json.loads(workload["shapes_json"])
        if str(s["name"]).startswith(USER_SHAPE_PREFIX)
    )
    tested = [p.concurrency for p in ctx.points_in("concurrency") if p.concurrency]
    cap = capacity(ctx)
    op_value = cap.get("op") if cap else None
    op = op_value if isinstance(op_value, int) else None
    change = parse_flag(experiment["flag"])
    max_num_seqs = setup["max_num_seqs"]
    enforce_eager = None if setup["enforce_eager"] is None else bool(setup["enforce_eager"])
    if change.setting == "--max-num-seqs" and change.value and change.value.isdigit():
        max_num_seqs = int(change.value)
    if change.setting == "--enforce-eager":
        enforce_eager = not change.removal
    return RunRequest(
        url=bundle["endpoint"]["base_url"],
        gpu=gpu_from_json(json.loads(setup["gpu_json"])),
        api_key=api_key,
        gpu_count=setup["gpu_count"],
        model=bundle["snapshot"]["served_model"],
        tokenizer=bundle["snapshot"]["hf_model_id"],
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        product=ProductType(workload["product_type"]),
        slo_ttft_ms=slo.get("ttft_ms"),
        slo_tpot_ms=slo.get("tpot_ms"),
        slo_percentile=slo.get("percentile"),
        phases=("baseline", "concurrency"),
        max_concurrency=max(tested) if tested else 256,
        min_concurrency=max(1, op // 2) if op else 1,
        label=f"experiment {experiment['id']}: {experiment['flag']}",
        shapes=shapes,
    )


def declared_change(experiment: Any, base_value: Any) -> dict[str, Any]:
    change = parse_flag(experiment["flag"])
    after = "removed" if change.removal else (change.value or "on")
    return {
        "key": change.setting,
        "label": change.setting or experiment["flag"],
        "a": base_value if base_value is not None else "not set",
        "b": after,
    }


def base_value(conn: Connection, experiment: Any) -> Any:
    """What the setting was in the run that proposed the change, if kneebench knows it."""
    if experiment["base_run_id"] is None:
        return None
    bundle = repo.run_bundle(conn, experiment["base_run_id"])
    setting = experiment["setting"]
    setup, snapshot = bundle["setup"], bundle["snapshot"]
    if setting == "--max-num-seqs":
        theory = json.loads(bundle["run"]["theory_json"])
        return setup["max_num_seqs"] or f"{theory.get('default_max_num_seqs')} (default)"
    if setting == "--enforce-eager":
        return {1: "on", 0: "off"}.get(setup["enforce_eager"], "unknown")
    if setting == "--max-model-len":
        return snapshot["max_model_len"]
    cache = json.loads(snapshot["cache_config_json"]) if snapshot["cache_config_json"] else {}
    return {
        "--gpu-memory-utilization": cache.get("gpu_memory_utilization"),
        "--kv-cache-dtype": cache.get("cache_dtype"),
        "--enable-prefix-caching": cache.get("enable_prefix_caching"),
        "--block-size": cache.get("block_size"),
    }.get(setting)

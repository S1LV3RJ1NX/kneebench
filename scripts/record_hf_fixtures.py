"""Save public Hugging Face model metadata as test fixtures (dev tool).

Usage: uv run python scripts/record_hf_fixtures.py org/model [org/model ...]
Writes tests/fixtures/hf/<org>--<model>/{config.json,safetensors_params.json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download

OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "hf"


def record(repo_id: str) -> Path:
    dest = OUT / repo_id.replace("/", "--")
    dest.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text())
    params = dict(get_safetensors_metadata(repo_id).parameter_count)
    (dest / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (dest / "safetensors_params.json").write_text(json.dumps(params, indent=2) + "\n")
    return dest


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        print(record(rid))

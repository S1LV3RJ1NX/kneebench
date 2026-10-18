"""Helpers to load committed fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kneebench.physics.model import ModelSpec, model_spec_from_hf

FIXTURES = Path(__file__).parent / "fixtures"


def hf_fixture(repo_id: str) -> tuple[dict[str, Any], dict[str, int]]:
    d = FIXTURES / "hf" / repo_id.replace("/", "--")
    return (
        json.loads((d / "config.json").read_text()),
        json.loads((d / "safetensors_params.json").read_text()),
    )


def model_fixture(repo_id: str) -> ModelSpec:
    config, params = hf_fixture(repo_id)
    return model_spec_from_hf(config, params, hf_id=repo_id)


LLAMA_8B = "unsloth/Meta-Llama-3.1-8B-Instruct"
LLAMA_8B_FP8 = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
LLAMA_8B_AWQ = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_70B = "unsloth/Meta-Llama-3.1-70B-Instruct"

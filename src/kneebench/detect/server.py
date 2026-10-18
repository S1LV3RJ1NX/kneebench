"""Auto-detect everything the endpoint will tell us; the user corrects the rest."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kneebench.detect.endpoint import normalize_base_url, probe
from kneebench.detect.hf import root_to_hf_id
from kneebench.detect.metrics import CacheConfig, cache_config, parse_samples
from kneebench.detect.netprobe import NetworkProfile, probe_network


@dataclass(frozen=True)
class ServedModel:
    id: str
    root: str | None
    max_model_len: int | None
    hf_id: str | None


@dataclass
class ServerSnapshot:
    base_url: str
    reachable: bool
    vllm_version: str | None
    models: list[ServedModel]
    cache: CacheConfig | None
    metrics_available: bool
    metrics_text: str | None = field(default=None, repr=False)
    network: NetworkProfile | None = None
    errors: list[str] = field(default_factory=list)

    def model(self, served_id: str | None = None) -> ServedModel | None:
        if served_id:
            return next((m for m in self.models if m.id == served_id), None)
        return self.models[0] if self.models else None


def detect_server(
    base_url: str,
    api_key: str | None = None,
    measure_network: bool = True,
) -> ServerSnapshot:
    base = normalize_base_url(base_url)
    errors: list[str] = []

    version_probe = probe(base, "/version", api_key)
    version = None
    if version_probe.ok and isinstance(version_probe.body, dict):
        version = version_probe.body.get("version")

    models_probe = probe(base, "/v1/models", api_key)
    models: list[ServedModel] = []
    if models_probe.ok and isinstance(models_probe.body, dict):
        for m in models_probe.body.get("data") or []:
            root = m.get("root")
            models.append(
                ServedModel(
                    id=str(m.get("id")),
                    root=root,
                    max_model_len=m.get("max_model_len"),
                    hf_id=root_to_hf_id(root) or root_to_hf_id(m.get("id")),
                )
            )
    else:
        errors.append(f"/v1/models: {models_probe.error or models_probe.status}")

    metrics_probe = probe(base, "/metrics", api_key, expect_json=False)
    text = metrics_probe.text if metrics_probe.ok else None
    cache = cache_config(parse_samples(text)) if text and "vllm:" in text else None

    network = probe_network(base, api_key) if measure_network else None
    return ServerSnapshot(
        base_url=base,
        reachable=models_probe.status is not None or version_probe.status is not None,
        vllm_version=str(version) if version else None,
        models=models,
        cache=cache,
        metrics_available=bool(text and "vllm:" in text),
        metrics_text=text,
        network=network,
        errors=errors,
    )


@dataclass(frozen=True)
class ModelMetadata:
    hf_id: str
    config: dict[str, Any]
    safetensors_params: dict[str, int] | None


def fetch_model_metadata(hf_id: str) -> ModelMetadata:
    """config.json and safetensors parameter counts from the Hub (or a local directory)."""
    local = Path(hf_id).expanduser()
    if local.is_dir():
        config = json.loads((local / "config.json").read_text())
        return ModelMetadata(hf_id=hf_id, config=config, safetensors_params=None)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import get_safetensors_metadata, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, NotASafetensorsRepoError

    config = json.loads(Path(hf_hub_download(hf_id, "config.json")).read_text())
    try:
        counts = get_safetensors_metadata(hf_id).parameter_count
        params: dict[str, int] | None = {str(k): int(v) for k, v in counts.items()}
    except (NotASafetensorsRepoError, HfHubHTTPError):
        params = None
    return ModelMetadata(hf_id=hf_id, config=config, safetensors_params=params)

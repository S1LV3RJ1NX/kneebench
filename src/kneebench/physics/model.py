"""Model physics from Hugging Face metadata: weight bytes, FLOPs and KV cache size.

Weight bytes come from safetensors metadata (element count per stored dtype), which is
exact for dense, FP8 and packed int4 checkpoints alike. Logical parameters, used for
FLOPs, come from the same metadata unless the checkpoint is quantized, where packed
storage undercounts them; then they are estimated from config.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
TORCH_DTYPE_BYTES = {
    "float32": 4,
    "float": 4,
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "float8_e4m3fn": 1,
}
FP8_KV_DTYPES = {"fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla"}


class Support(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


class UnsupportedModelError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str | None
    architecture: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    params: int
    weight_bytes: int
    weight_dtype: str
    quantization: str | None = None
    fp8_compute: bool = False
    support: Support = Support.EXACT
    notes: tuple[str, ...] = field(default=())

    @property
    def activation_dtype_bytes(self) -> int:
        return TORCH_DTYPE_BYTES.get(self.weight_dtype, 2) if self.weight_dtype else 2

    def kv_bytes_per_token(self, cache_dtype: str = "auto") -> int:
        """Bytes of K and V cache per token, summed over all layers and heads."""
        elem = 1 if cache_dtype.lower() in FP8_KV_DTYPES else self.activation_dtype_bytes
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * elem

    def kv_bytes_per_token_per_gpu(self, tp: int, cache_dtype: str = "auto") -> float:
        # KV heads are sharded across TP ranks; with fewer heads than ranks, they replicate.
        heads_per_gpu = max(self.num_kv_heads / tp, 1.0)
        return self.kv_bytes_per_token(cache_dtype) * heads_per_gpu / self.num_kv_heads


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    # Multimodal configs nest the language model under text_config.
    text = config.get("text_config")
    return {**config, **text} if isinstance(text, dict) else config


def _quant_method(config: dict[str, Any]) -> str | None:
    q = config.get("quantization_config")
    if not isinstance(q, dict):
        return None
    method = str(q.get("quant_method") or "").lower() or None
    if method == "compressed-tensors" and _compressed_tensors_is_fp8(q):
        return "fp8"
    return method


def _compressed_tensors_is_fp8(q: dict[str, Any]) -> bool:
    for group in (q.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if weights.get("type") == "float" and weights.get("num_bits") == 8:
            return True
    return "fp8" in str(q.get("format", "")).lower()


def estimate_params(config: dict[str, Any]) -> int:
    """Dense decoder parameter estimate: embeddings, attention and gated MLP per layer."""
    cfg = _text_config(config)
    hidden = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or hidden // heads)
    inter = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    attn = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2
    mlp = 3 * hidden * inter
    embed = vocab * hidden * (1 if cfg.get("tie_word_embeddings") else 2)
    return layers * (attn + mlp) + embed


def model_spec_from_hf(
    config: dict[str, Any],
    safetensors_params: dict[str, int] | None,
    hf_id: str | None = None,
) -> ModelSpec:
    cfg = _text_config(config)
    arch = (cfg.get("architectures") or ["unknown"])[0]
    notes: list[str] = []
    support = Support.EXACT

    if cfg.get("kv_lora_rank"):
        raise UnsupportedModelError(
            f"{arch} uses multi-head latent attention (MLA); kneebench supports it after the MVP."
        )
    if cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("n_routed_experts"):
        raise UnsupportedModelError(
            f"{arch} is a mixture-of-experts model; kneebench supports MoE after the MVP."
        )
    if cfg.get("sliding_window") and cfg.get("use_sliding_window", True) and cfg.get("layer_types"):
        support = Support.APPROXIMATE
        notes.append("Some layers use sliding-window attention; KV estimates are an upper bound.")

    heads = int(cfg["num_attention_heads"])
    hidden = int(cfg["hidden_size"])
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or hidden // heads)
    weight_dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
    quant = _quant_method(config)

    if safetensors_params:
        weight_bytes = sum(n * DTYPE_BYTES.get(dt, 2) for dt, n in safetensors_params.items())
        stored = sum(safetensors_params.values())
        params = estimate_params(config) if quant and quant != "fp8" else stored
    else:
        params = estimate_params(config)
        weight_bytes = params * TORCH_DTYPE_BYTES.get(weight_dtype, 2)
        support = Support.APPROXIMATE
        notes.append("No safetensors metadata: weight bytes estimated from config.json.")
        if quant:
            notes.append(f"{quant} quantization without metadata: weight bytes may be high.")

    return ModelSpec(
        hf_id=hf_id,
        architecture=arch,
        num_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        params=params,
        weight_bytes=weight_bytes,
        weight_dtype=weight_dtype,
        quantization=quant,
        fp8_compute=quant == "fp8",
        support=support,
        notes=tuple(notes),
    )

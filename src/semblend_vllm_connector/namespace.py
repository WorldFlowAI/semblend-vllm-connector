"""Namespace extraction for vLLM requests."""

from __future__ import annotations

import hashlib
from typing import Any

from semblend_vllm_connector.config import SemBlendVllmConfig


def _safe_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if obj is None:
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def model_id_from_config(config: SemBlendVllmConfig, vllm_config: Any) -> str:
    if config.model_id:
        return config.model_id
    model_config = getattr(vllm_config, "model_config", None)
    value = _safe_attr(model_config, "model", "served_model_name", "model_name")
    return str(value or "unknown-model")


def namespace_for_request(config: SemBlendVllmConfig, vllm_config: Any, request: Any) -> str:
    model_id = model_id_from_config(config, vllm_config)
    cache_config = getattr(vllm_config, "cache_config", None)
    model_config = getattr(vllm_config, "model_config", None)

    parts = {
        "model": model_id,
        "tokenizer": _safe_attr(model_config, "tokenizer", "tokenizer_name") or "unknown-tokenizer",
        "block": _safe_attr(cache_config, "block_size") or "unknown-block",
        "dtype": _safe_attr(model_config, "dtype") or "unknown-dtype",
        "salt": _safe_attr(request, "cache_salt") or "no-salt",
        "lora": _safe_attr(request, "lora_request") or "no-lora",
    }
    raw = "|".join(f"{key}={value}" for key, value in sorted(parts.items()))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"vllm:{digest}"


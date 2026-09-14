"""Namespace and tenant-key extraction for vLLM requests.

The per-request namespace below is engine-local: it digests the model,
tokenizer, block size, dtype, cache salt and adapter, so nothing outside this
process can reproduce it. The request's raw ``cache_salt`` is read separately,
by :func:`cache_salt_for_request`, and travels to SemBlend as the input to the
published tenant key — see ``docs/VLLM_CONNECTOR_CONTRACT.md``, "Tenant key".
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from semblend_vllm_connector.config import SemBlendVllmConfig

logger = logging.getLogger(__name__)

#: Warned at most once per process: a request type that has no ``cache_salt``
#: field at all is a wiring gap, not an unsalted request.
_warned_missing_cache_salt_field = False


def _warn_missing_cache_salt_field_once() -> None:
    global _warned_missing_cache_salt_field
    if _warned_missing_cache_salt_field:
        return
    _warned_missing_cache_salt_field = True
    logger.warning(
        "vLLM requests reaching this connector have no cache_salt field, so "
        "every donor it registers is published with no tenant key and a "
        "router cannot place requests by tenant. This is a version mismatch "
        "between the connector and vLLM's request type, not an unsalted "
        "deployment: see docs/VLLM_CONNECTOR_CONTRACT.md, 'Tenant key'."
    )


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


def cache_salt_for_request(request: Any) -> str | None:
    """The request's raw ``cache_salt``, or None when it carries none.

    Kept beside :func:`namespace_for_request`, which reads the same field, so
    the one place that knows how a vLLM request spells it stays one place.
    The raw value is tenant-identifying: it is passed on, never logged.

    A request object with no such field at all warns once. Both cases return
    None, but they are not the same thing: an unsalted request is a real
    answer, while a request type that never carries the field means no donor
    this process registers can ever be selected by tenant — which is worth
    one log line rather than silence.
    """
    if request is not None and not hasattr(request, "cache_salt"):
        _warn_missing_cache_salt_field_once()
        return None
    salt = _safe_attr(request, "cache_salt")
    if salt is None:
        return None
    salt = salt if isinstance(salt, str) else str(salt)
    return salt or None


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

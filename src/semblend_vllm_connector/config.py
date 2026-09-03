"""Connector configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Mapping

from semblend_vllm_connector.types import ReuseMode


def _extra_config_keys() -> tuple[str, ...]:
    # Derived from the config fields so the getter path can never silently
    # drop a key the Mapping path honors (a hand-kept list drifted twice).
    return tuple(f.name for f in fields(SemBlendVllmConfig))


def _get_extra_config(vllm_config: Any) -> Mapping[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return {}
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if isinstance(extra, Mapping):
        return extra
    getter = getattr(kv_transfer_config, "get_from_extra_config", None)
    if callable(getter):
        values: dict[str, Any] = {}
        for key in _extra_config_keys():
            value = getter(key, None)
            if value is not None:
                values[key] = value
        return values
    return {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _coerce_bool(raw, default)


def _coerce_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, int):
        return bool(raw)
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_int(extra: Mapping[str, Any], key: str, env: str, default: int) -> int:
    raw = extra.get(key, os.environ.get(env, default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _read_float(extra: Mapping[str, Any], key: str, env: str, default: float) -> float:
    raw = extra.get(key, os.environ.get(env, default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _read_bool(extra: Mapping[str, Any], key: str, env: str, default: bool) -> bool:
    return _coerce_bool(extra.get(key, os.environ.get(env, default)), default)


@dataclass(frozen=True)
class SemBlendVllmConfig:
    mode: ReuseMode = ReuseMode.DISCOVERY_ONLY
    provider: str = "local"
    provider_module: str | None = None
    provider_class: str | None = None
    model_id: str | None = None
    min_prompt_tokens: int = 256
    min_semantic_span: int = 512
    min_similarity: float = 0.70
    min_reuse_ratio: float = 0.50
    embedder_type: str | None = None
    chunk_size: int | None = None
    max_donors: int = 10_000
    register_donors: bool = True
    skip_when_exact_prefix_ratio_at_least: float = 0.50
    lookup_top_k: int = 5
    enable_prompt_text: bool = False
    log_decisions: bool = True
    audit_path: str | None = None
    kv_storage_path: str = "/tmp/semblend-vllm-kv"
    max_materialized_tokens: int = 4096
    allow_non_identical_request_only: bool = False
    # A recipient that was served donor KV is not captured as a donor itself
    # unless asked: the copy cost ~230 ms of the hit path at 3.5K tokens and
    # its donor already covers the content.
    capture_served_requests: bool = False
    # "disk" writes per-layer safetensors under kv_storage_path; "memory"
    # keeps donor layers in the worker's host RAM (no file I/O on capture or
    # load) with an LRU cap on donors.
    kv_storage_backend: str = "disk"
    kv_memory_max_donors: int = 16

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "SemBlendVllmConfig":
        extra = _get_extra_config(vllm_config)
        mode_raw = str(
            extra.get("mode", os.environ.get("SEMBLEND_VLLM_MODE", ReuseMode.DISCOVERY_ONLY.value))
        )
        try:
            mode = ReuseMode(mode_raw)
        except ValueError:
            mode = ReuseMode.DISCOVERY_ONLY

        model_id = extra.get("model_id") or os.environ.get("SEMBLEND_VLLM_MODEL_ID")
        if model_id is None:
            model_config = getattr(vllm_config, "model_config", None)
            model_id = (
                getattr(model_config, "model", None)
                or getattr(model_config, "served_model_name", None)
                or getattr(model_config, "model_name", None)
            )

        return cls(
            mode=mode,
            provider=str(extra.get("provider", os.environ.get("SEMBLEND_VLLM_PROVIDER", "local"))),
            provider_module=extra.get("provider_module")
            or os.environ.get("SEMBLEND_VLLM_PROVIDER_MODULE"),
            provider_class=extra.get("provider_class") or os.environ.get("SEMBLEND_VLLM_PROVIDER_CLASS"),
            model_id=str(model_id) if model_id is not None else None,
            min_prompt_tokens=_read_int(extra, "min_prompt_tokens", "SEMBLEND_VLLM_MIN_PROMPT_TOKENS", 256),
            min_semantic_span=_read_int(extra, "min_semantic_span", "SEMBLEND_VLLM_MIN_SEMANTIC_SPAN", 512),
            min_similarity=_read_float(extra, "min_similarity", "SEMBLEND_VLLM_MIN_SIMILARITY", 0.70),
            min_reuse_ratio=_read_float(
                extra, "min_reuse_ratio", "SEMBLEND_VLLM_MIN_REUSE_RATIO", 0.50
            ),
            embedder_type=extra.get("embedder_type") or os.environ.get("SEMBLEND_VLLM_EMBEDDER"),
            chunk_size=(
                _read_int(extra, "chunk_size", "SEMBLEND_VLLM_CHUNK_SIZE", 0) or None
            ),
            max_donors=_read_int(extra, "max_donors", "SEMBLEND_VLLM_MAX_DONORS", 10_000),
            register_donors=_read_bool(extra, "register_donors", "SEMBLEND_VLLM_REGISTER_DONORS", True),
            skip_when_exact_prefix_ratio_at_least=_read_float(
                extra,
                "skip_when_exact_prefix_ratio_at_least",
                "SEMBLEND_VLLM_SKIP_EXACT_RATIO",
                0.50,
            ),
            lookup_top_k=_read_int(extra, "lookup_top_k", "SEMBLEND_VLLM_LOOKUP_TOP_K", 5),
            enable_prompt_text=_read_bool(
                extra, "enable_prompt_text", "SEMBLEND_VLLM_ENABLE_PROMPT_TEXT", False
            ),
            log_decisions=_read_bool(extra, "log_decisions", "SEMBLEND_VLLM_LOG_DECISIONS", True),
            audit_path=(
                str(extra.get("audit_path") or os.environ.get("SEMBLEND_VLLM_AUDIT_PATH") or "")
                or None
            ),
            kv_storage_path=str(
                extra.get(
                    "kv_storage_path",
                    os.environ.get("SEMBLEND_VLLM_KV_STORAGE_PATH", "/tmp/semblend-vllm-kv"),
                )
            ),
            max_materialized_tokens=_read_int(
                extra,
                "max_materialized_tokens",
                "SEMBLEND_VLLM_MAX_MATERIALIZED_TOKENS",
                4096,
            ),
            allow_non_identical_request_only=_read_bool(
                extra,
                "allow_non_identical_request_only",
                "SEMBLEND_VLLM_ALLOW_NON_IDENTICAL_REQUEST_ONLY",
                False,
            ),
            capture_served_requests=_read_bool(
                extra, "capture_served_requests", "SEMBLEND_VLLM_CAPTURE_SERVED_REQUESTS", False
            ),
            kv_storage_backend=str(
                extra.get("kv_storage_backend", os.environ.get("SEMBLEND_VLLM_KV_STORAGE_BACKEND", "disk"))
            ).strip().lower(),
            kv_memory_max_donors=_read_int(
                extra, "kv_memory_max_donors", "SEMBLEND_VLLM_KV_MEMORY_MAX_DONORS", 16
            ),
        )

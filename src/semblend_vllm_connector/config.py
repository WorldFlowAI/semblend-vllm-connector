"""Connector configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Mapping

from semblend_vllm_connector.types import ReuseMode

logger = logging.getLogger("semblend_vllm_connector")


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
    # Prompts shorter than this never reach lookup or capture. Kept at or
    # above min_semantic_span: in semantic-span mode a prompt in the gap is
    # admitted to lookup but block_align_spans can never carve a servable
    # span out of it, so the lookup is pure miss tax (validation_warnings).
    min_prompt_tokens: int = 512
    min_semantic_span: int = 512
    # Lowest local prefix-cache boundary (num_computed_tokens) at which the
    # connector will serve; below it the request is declined and the engine
    # prefills it. 0 disables the gate. Why it exists: blocks the connector
    # fills hold donor KV hashed under the recipient's own token ids, so they
    # have to be kept out of vLLM's prefix cache. A request served from
    # boundary 0 therefore contributes nothing to that cache; if every
    # servable request is served that way, the shared leading prompt is never
    # cached and the boundary stays 0 for good. Set to block_size so unserved
    # requests prime the shared prefix cleanly; it also keeps a boundary-
    # alignment measurement from counting boundary-0 serves. Enforced in
    # the match hook, ahead of the provider lookup, so a declined request
    # costs no embedding; it applies in every mode.
    min_boundary_tokens: int = 0
    # Whether blocks this connector fills are dropped from vLLM's exact
    # prefix cache. They hold donor KV hashed under the recipient's own token
    # ids, and vLLM caches them in the same allocate_slots call that allocated
    # them, so with this off a later request whose tokens hash the same way is
    # served approximate KV through the engine's exact match -- upstream of
    # this connector, so no gate, no TTL and no decision is recorded. Leave it
    # on. Turning it off is for measurement only: it is what makes a
    # contaminated control arm, so the eviction's effect can be quantified.
    # With it off the connector still tracks the blocks it filled and audits
    # what it would have evicted.
    evict_filled_blocks_from_prefix_cache: bool = True
    min_similarity: float = 0.70
    min_reuse_ratio: float = 0.50
    embedder_type: str | None = None
    chunk_size: int | None = None
    max_donors: int = 10_000
    register_donors: bool = True
    skip_when_exact_prefix_ratio_at_least: float = 0.50
    lookup_top_k: int = 5
    enable_prompt_text: bool = False
    # Whether the text a lookup hands the provider starts at the request's
    # block-aligned exact-prefix boundary instead of at token 0. Providers
    # that embed prompt text truncate it to their embedder's window (a MiniLM
    # sentence encoder sees roughly the first 256 tokens, and SemBlend caps
    # the text by characters on top of that), so with a wrapper in front of
    # the content the whole window is wrapper: every request wrapped the same
    # way embeds alike, and none of them embeds like the donor that holds the
    # content. The boundary is where the exact prefix stops and therefore
    # where the tokens this connector wants to serve begin, so it is also
    # where the text that identifies them begins. Off restores 0.2.4
    # behaviour for an A/B.
    boundary_sliced_query_text: bool = True
    # Floor under which a boundary-sliced text is not worth embedding; below
    # it the full prompt text is sent instead and the audit names the
    # fallback. A boundary near the prompt end leaves a few characters that
    # identify nothing, and an embedding of them would match arbitrarily.
    min_query_text_chars: int = 64
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

    def validation_warnings(self) -> tuple[str, ...]:
        """Knob combinations that are legal but self-defeating.

        Returned rather than raised: none of them is a correctness hazard
        (the affected requests decline, and every decline is counted), and
        a threshold sweep may open the gap on purpose to measure its cost.
        """
        warnings: list[str] = []
        # block_align_spans only runs on the semantic-span path; in the other
        # modes min_semantic_span is inert and the gap costs nothing.
        if (
            self.mode == ReuseMode.SEMANTIC_SPAN_EXPERIMENTAL
            and self.min_prompt_tokens < self.min_semantic_span
        ):
            warnings.append(
                f"min_prompt_tokens={self.min_prompt_tokens} is below "
                f"min_semantic_span={self.min_semantic_span}: prompts in that range "
                "are admitted to lookup but can never carry a servable span, so "
                "each of those lookups is a guaranteed miss"
            )
        return tuple(warnings)

    def __post_init__(self) -> None:
        for message in self.validation_warnings():
            logger.warning("SemBlend config: %s", message)

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
            provider_class=extra.get("provider_class")
            or os.environ.get("SEMBLEND_VLLM_PROVIDER_CLASS"),
            model_id=str(model_id) if model_id is not None else None,
            min_prompt_tokens=_read_int(
                extra, "min_prompt_tokens", "SEMBLEND_VLLM_MIN_PROMPT_TOKENS", 512
            ),
            min_semantic_span=_read_int(
                extra, "min_semantic_span", "SEMBLEND_VLLM_MIN_SEMANTIC_SPAN", 512
            ),
            min_boundary_tokens=_read_int(
                extra, "min_boundary_tokens", "SEMBLEND_VLLM_MIN_BOUNDARY_TOKENS", 0
            ),
            evict_filled_blocks_from_prefix_cache=_read_bool(
                extra,
                "evict_filled_blocks_from_prefix_cache",
                "SEMBLEND_VLLM_EVICT_FILLED_BLOCKS_FROM_PREFIX_CACHE",
                True,
            ),
            min_similarity=_read_float(
                extra, "min_similarity", "SEMBLEND_VLLM_MIN_SIMILARITY", 0.70
            ),
            min_reuse_ratio=_read_float(
                extra, "min_reuse_ratio", "SEMBLEND_VLLM_MIN_REUSE_RATIO", 0.50
            ),
            embedder_type=extra.get("embedder_type") or os.environ.get("SEMBLEND_VLLM_EMBEDDER"),
            chunk_size=(_read_int(extra, "chunk_size", "SEMBLEND_VLLM_CHUNK_SIZE", 0) or None),
            max_donors=_read_int(extra, "max_donors", "SEMBLEND_VLLM_MAX_DONORS", 10_000),
            register_donors=_read_bool(
                extra, "register_donors", "SEMBLEND_VLLM_REGISTER_DONORS", True
            ),
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
            boundary_sliced_query_text=_read_bool(
                extra,
                "boundary_sliced_query_text",
                "SEMBLEND_VLLM_BOUNDARY_SLICED_QUERY_TEXT",
                True,
            ),
            min_query_text_chars=_read_int(
                extra, "min_query_text_chars", "SEMBLEND_VLLM_MIN_QUERY_TEXT_CHARS", 64
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
                extra.get(
                    "kv_storage_backend", os.environ.get("SEMBLEND_VLLM_KV_STORAGE_BACKEND", "disk")
                )
            )
            .strip()
            .lower(),
            kv_memory_max_donors=_read_int(
                extra, "kv_memory_max_donors", "SEMBLEND_VLLM_KV_MEMORY_MAX_DONORS", 16
            ),
        )

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


#: Every capture policy this connector understands; see ``capture_policy``.
CAPTURE_POLICIES = ("all", "sampled", "hinted")


def _read_capture_policy(extra: Mapping[str, Any]) -> str:
    """The capture policy, falling back to the measured default when unknown.

    An unrecognized value must not silently capture nothing: the donor pool
    would be empty and every lookup would miss with no reason recorded. It
    falls back to "all", which is what a deployment that never set the knob
    gets, and says so.
    """
    raw = extra.get("capture_policy", os.environ.get("SEMBLEND_VLLM_CAPTURE_POLICY", "all"))
    policy = str(raw).strip().lower()
    if policy in CAPTURE_POLICIES:
        return policy
    logger.warning(
        "SemBlend config: capture_policy=%r is not one of %s; capturing every eligible request",
        raw,
        ", ".join(CAPTURE_POLICIES),
    )
    return "all"


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
    # Semantic-span mode: before the lookup, ask the provider whether any
    # donor in the namespace could hold an identical run from the boundary
    # long enough to clear min_semantic_span. A provider that answers "no"
    # has ruled the load out, so the embedding and search are skipped. The
    # answer is exact in that direction; "maybe" runs the lookup as before.
    lookup_precheck: bool = True
    # Semantic-span mode: when a lookup returns runs from several donors,
    # serve one contiguous span assembled from them (pieces meet at any token;
    # only the span's outer ends are block-aligned). Needs a provider that
    # returns multi-donor segments (SemBlend with SEMBLEND_MULTI_DONOR=1).
    multi_donor_spans: bool = True
    # Segmented prefill, for an engine whose scheduler asks the connector
    # again part-way through a prefill (get_num_new_matched_tokens_mid_prefill
    # and get_prefill_compute_limit; not in stock vLLM). Later segments reuse
    # the admission lookup, so they cost a plan, not a search. A later segment
    # shorter than this floor is computed rather than loaded.
    min_mid_prefill_segment: int = 128
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
    # Which admitted requests become donors: "all" (every eligible request,
    # the measured default), "sampled" (a deterministic fraction, keyed on the
    # request id so a rerun captures the same set), or "hinted" (only requests
    # the caller marked). The capture is paid by the request being captured --
    # ~1.2 GB of host copy at a 21.3K-token prompt -- so a deployment whose
    # donors are known in advance can stop paying it on the rest of its
    # traffic. "all" is the default because the phase-0 manifests predict
    # which requests become donors from it.
    capture_policy: str = "all"
    # The fraction captured under capture_policy="sampled". Inert otherwise.
    capture_sample_rate: float = 1.0
    # The key capture_policy="hinted" looks for. Read from the request's
    # sampling-params extra_args (vLLM's OpenAI server forwards `vllm_xargs`
    # into it), from the request metadata/header mappings the routing keys
    # come from -- both the bare key and "x-"-prefixed -- and from an
    # attribute of that name on the request object.
    capture_hint_key: str = "semblend_capture"
    # A request carrying this hint is one stage of a staged prefill: a caller
    # that split a prompt at its edits sends each prefix first so the engine
    # computes the edited text and the next stage finds it in the prefix
    # cache. Such a request keeps the blocks this connector filled in that
    # cache instead of having them evicted, because the next stage is the same
    # prompt and needs them. Read from the same places as the capture hint.
    stage_hint_key: str = "semblend_stage"
    # How many donor writes may be queued for the writer thread before the
    # forward pass has to wait for it. A full queue means the store is slower
    # than the engine produces captures; the submitting thread blocks and the
    # block is counted (capture_write_queue_blocked_total). Nothing is
    # dropped: a dropped layer would leave a donor advertising bytes that were
    # never written.
    capture_write_queue_depth: int = 64
    # Copy captured layers to pinned host memory on a side CUDA stream, so the
    # forward pass launches the copy and moves on instead of waiting for the
    # GPU to reach the layer and for the bytes to cross PCIe.
    capture_async_copy: bool = True
    # How long teardown waits for the writer to drain, in seconds, covering
    # the whole of the stop: a store that has wedged leaves the queue full, so
    # handing the writer its stop signal is part of the wait and not ahead of
    # it. This is what a process exit costs when the store has stopped
    # responding, so it is short by default and the incomplete drain is
    # warned about rather than waited out.
    capture_write_close_timeout_s: float = 5.0
    # How many scheduler steps a finished donor may wait for its own capture
    # record before its registration is declined for good. Steps, not calls:
    # a step that finishes a whole batch of requests spends none of this
    # budget, because the worker has no chance to publish until the next one.
    # vLLM frees a request before its id reaches finished_req_ids, so a
    # capture still open at finish is closed by the worker one step after the
    # registration hook ran: at 0 that donor is always declined although it
    # was about to become readable.
    donor_registration_retry_steps: int = 8
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
        if self.capture_policy == "sampled" and self.capture_sample_rate <= 0.0:
            warnings.append(
                f"capture_policy=sampled with capture_sample_rate="
                f"{self.capture_sample_rate}: no request is ever captured, so the "
                "donor pool stays empty and every lookup misses"
            )
        if self.capture_policy == "sampled" and self.capture_sample_rate >= 1.0:
            warnings.append(
                f"capture_policy=sampled with capture_sample_rate="
                f"{self.capture_sample_rate}: every eligible request is captured, "
                "which is capture_policy=all under another name"
            )
        if self.donor_registration_retry_steps <= 0:
            warnings.append(
                f"donor_registration_retry_steps={self.donor_registration_retry_steps}: "
                "a request that finishes while its capture is still open is declined "
                "at once, and vLLM closes such a capture one step later, so those "
                "donors are lost"
            )
        if self.capture_write_close_timeout_s <= 0.0:
            warnings.append(
                f"capture_write_close_timeout_s={self.capture_write_close_timeout_s}: "
                "teardown does not wait for the capture writer at all, so donor "
                "writes still queued at shutdown are abandoned"
            )
        if self.capture_write_queue_depth < 1:
            warnings.append(
                f"capture_write_queue_depth={self.capture_write_queue_depth} is below "
                "1 and is read as 1: every captured layer then waits for the "
                "previous one to land"
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
            min_mid_prefill_segment=_read_int(
                extra, "min_mid_prefill_segment", "SEMBLEND_VLLM_MIN_MID_PREFILL_SEGMENT", 128
            ),
            multi_donor_spans=_read_bool(
                extra, "multi_donor_spans", "SEMBLEND_VLLM_MULTI_DONOR_SPANS", True
            ),
            lookup_precheck=_read_bool(
                extra, "lookup_precheck", "SEMBLEND_VLLM_LOOKUP_PRECHECK", True
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
            capture_policy=_read_capture_policy(extra),
            capture_sample_rate=_read_float(
                extra, "capture_sample_rate", "SEMBLEND_VLLM_CAPTURE_SAMPLE_RATE", 1.0
            ),
            stage_hint_key=str(
                extra.get(
                    "stage_hint_key",
                    os.environ.get("SEMBLEND_VLLM_STAGE_HINT_KEY", "semblend_stage"),
                )
            )
            .strip()
            .lower(),
            capture_hint_key=str(
                extra.get(
                    "capture_hint_key",
                    os.environ.get("SEMBLEND_VLLM_CAPTURE_HINT_KEY", "semblend_capture"),
                )
            )
            .strip()
            .lower(),
            capture_async_copy=_read_bool(
                extra, "capture_async_copy", "SEMBLEND_VLLM_CAPTURE_ASYNC_COPY", True
            ),
            capture_write_queue_depth=_read_int(
                extra,
                "capture_write_queue_depth",
                "SEMBLEND_VLLM_CAPTURE_WRITE_QUEUE_DEPTH",
                64,
            ),
            capture_write_close_timeout_s=_read_float(
                extra,
                "capture_write_close_timeout_s",
                "SEMBLEND_VLLM_CAPTURE_WRITE_CLOSE_TIMEOUT_S",
                5.0,
            ),
            donor_registration_retry_steps=_read_int(
                extra,
                "donor_registration_retry_steps",
                "SEMBLEND_VLLM_DONOR_REGISTRATION_RETRY_STEPS",
                8,
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

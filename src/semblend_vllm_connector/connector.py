"""vLLM KVConnector implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from semblend_vllm_connector._vllm_compat import (
    KVConnectorBase_V1,
    KVConnectorRole,
    get_virtual_engine,
)
from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.namespace import model_id_from_config, namespace_for_request
from semblend_vllm_connector.provider import SemanticKvProvider, load_provider
from semblend_vllm_connector.semantic_span import (
    block_align_spans,
    supply_at_boundary,
)
from semblend_vllm_connector.types import (
    DonorRegistration,
    MaterializationKind,
    PendingLoad,
    PendingStore,
    ReuseMode,
    SemanticLookupRequest,
    SemBlendConnectorMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


logger = logging.getLogger("semblend_vllm_connector")

# Per-donor length record the scheduler role reads from disk under both
# storage backends; retracted on eviction so it never outlives the tensors.
_DONOR_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class _CaptureState:
    """A donor still in prefill, as the scheduler role carries it across steps.

    The cached-request diff vLLM sends after admission has no token ids and
    only the blocks added that step, so the running table and the prompt are
    kept here. ``captured_end`` is the block-aligned position the last emitted
    store reached; a later chunk only produces a store when it moves it.
    """

    token_ids: list[int]
    block_ids: tuple[list[int], ...]
    namespace: str
    captured_end: int


def _cacheable_prefix_tokens(num_tokens: int, block_size: int) -> int:
    """Return the largest block-aligned prefix vLLM can treat as computed."""
    eligible = max(num_tokens - 1, 0)
    if eligible <= 0:
        return 0
    return ((eligible - 1) // block_size) * block_size


def _normalize_block_ids(block_ids: Any) -> tuple[list[int], ...] | None:
    if block_ids is None:
        return None
    if not block_ids:
        return None
    if isinstance(block_ids[0], int):
        return (list(block_ids),)
    return tuple(list(group) for group in block_ids)


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _rope_keys_are_layer_types(rope: Mapping[str, Any]) -> bool:
    """vLLM's own tell for the nested shape: every key is a layer-type name."""
    try:
        from transformers.configuration_utils import ALLOWED_LAYER_TYPES
    except ImportError:
        return False
    return set(rope.keys()).issubset(set(ALLOWED_LAYER_TYPES))


def _is_nested_rope_parameters(rope: Mapping[str, Any]) -> bool:
    """Whether ``rope_parameters`` is keyed by layer type (transformers v5).

    vLLM decides this with ``is_rope_parameters_nested`` (every key drawn from
    transformers' ``ALLOWED_LAYER_TYPES``); use its answer when vLLM is
    importable so the two never disagree on a real engine.
    """
    if not rope:
        return False
    try:
        from vllm.transformers_utils.config import is_rope_parameters_nested
    except ImportError:
        # Without vLLM, apply both of its tells and treat either as nested.
        # Taking only one of them can classify a nested config as flat, and
        # the rope guards then read a layer-type sub-dict as if it were the
        # rope config itself and pass a model they should decline. A flat
        # config's values are scalars, strings or lists, and its keys are not
        # layer-type names, so neither tell fires on it.
        return all(isinstance(value, Mapping) for value in rope.values()) or (
            _rope_keys_are_layer_types(rope)
        )
    return bool(is_rope_parameters_nested(dict(rope)))


def _rope_parameter_sets(rope: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """The rope dicts to inspect: one per layer type, or the flat dict alone.

    Mirrors vLLM's normalization of the nested shape to ``{layer_type: dict}``
    before iterating. None means the shape could not be classified, which must
    not be read as a plain config.
    """
    if not _is_nested_rope_parameters(rope):
        return [rope]
    sets = [value for value in rope.values() if isinstance(value, Mapping)]
    if len(sets) != len(rope):
        return None
    return sets


class SemBlendVllmConnector(KVConnectorBase_V1):
    """Safe-by-default SemBlend-backed vLLM OOT connector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._config = SemBlendVllmConfig.from_vllm_config(vllm_config)
        self._provider: SemanticKvProvider | None = None
        self._pending_loads: dict[str, PendingLoad] = {}
        # Load decisions taken in get_num_new_matched_tokens, which vLLM
        # documents as side-effect free. The PendingLoad the worker consumes is
        # only built once the scheduler has actually allocated destination
        # blocks, so a request that never reaches update_state_after_alloc
        # cannot leak a destination-less load into connector metadata.
        self._load_plans: dict[str, dict[str, Any]] = {}
        # Mid-request consults re-enter get_num_new_matched_tokens on every
        # scheduling step; the semantic lookup (embedding + alignment) is
        # request-stable, so memoize it per request (cleared on finish).
        self._lookup_cache: dict[str, Any] = {}
        # Requests that have reached the match hook at least once. The early
        # exits above the lookup cannot use `first_lookup` (they return before
        # anything is memoized), so this is what makes them per-request.
        self._seen_requests: set[str] = set()
        # vLLM >= 0.26: the worker registers per-layer KV cache tensors
        # directly; layer objects no longer expose kv_cache. Load/save use
        # this dict when populated, falling back to the forward-context walk.
        self._registered_kv_caches: dict[str, Any] = {}
        # Recipients that received a semantic load this scheduling cycle;
        # consulted by the store builder so served requests are not captured.
        self._served_request_ids: set[str] = set()
        # Worker-side donor layers for kv_storage_backend="memory":
        # storage key -> {layer_name: cpu tensor, "__token_count__": int,
        # "__request_id__": str}, in access order so the cap evicts the least
        # recently used donor. This backend is the only one that evicts, so
        # it is the only way the scheduler role (another process, reading
        # metadata.json) can be left advertising tensors that are gone; the
        # eviction path retracts that file for exactly this reason. A "disk"
        # backend with kv_storage_path on tmpfs (e.g. /dev/shm) keeps the
        # cross-process truth in one place at memory speed and is the
        # recommended setup.
        self._memory_store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        # Worker-side capture progress: request id -> {layer_name: tokens
        # captured so far}. Chunked prefill computes a donor one prefix chunk
        # per step, so a capture is a sequence of appends and each layer has
        # to know where its own copy ends (save_kv_layer is per layer).
        self._capture_progress: dict[str, dict[str, int]] = {}
        # Destination blocks of loads declined on this worker after the
        # scheduler had already credited them; drained by vLLM each forward
        # so the engine recomputes them under kv_load_failure_policy.
        self._load_error_block_ids: set[int] = set()
        # Scheduler-side state for donors still in prefill, so the chunks
        # after admission can be captured too.
        self._capture_state: dict[str, _CaptureState] = {}
        self._last_attn_metadata: Any = None
        # vllm-fork capability gate: re-consult this connector at chunked
        # continuation boundaries (mid-prompt external KV). Only the
        # semantic-span mode benefits; other modes keep stock behavior.
        self.supports_mid_request_matching = (
            self._config.mode == ReuseMode.SEMANTIC_SPAN_EXPERIMENTAL
        )
        # Counter scope convention. The phase-0 reuse metrics join these events
        # per request, so a counter with no scope suffix fires at most once per
        # request: sites below the lookup gate on `first_lookup`, sites above it
        # on `_first_attempt_for`, and advertise sites on `first_advertise`. A
        # counter whose decision genuinely differs between scheduling attempts
        # of the same request (anything derived from num_computed_tokens) keeps
        # firing per attempt and carries a `_per_attempt` suffix, so a reader
        # never mistakes it for a request count. A counter whose name names its
        # own unit (`layer_*`, `*_evictions`) counts that unit.
        self._stats: Counter[str] = Counter()
        # (expected 4-D page content dim, which source answered, K|V halves
        # equal). Engine-static: resolved once so its unresolved case is not
        # counted per layer.
        self._content_dim_expectation: tuple[int | None, str, bool] | None = None
        self._block_size = int(getattr(getattr(vllm_config, "cache_config", None), "block_size", 16))
        self._prompt_tokenizer: Any | None = None
        self._prompt_tokenizer_failed = False
        # Startup checks that could not run (see _note_compat_check_incomplete):
        # a skipped check is not a passed check and must stay visible.
        self._compat_check_incomplete: list[str] = []
        self._compat_decline = self._check_engine_compatibility(vllm_config)

        self._provider = load_provider(self._config)

        # HMA is auto-disabled for connectors that do not declare SupportsHMA,
        # which changes the engine's memory layout: record it so an A/B is not
        # silently comparing two different engine configurations.
        hma_disabled = bool(
            getattr(
                getattr(vllm_config, "scheduler_config", None),
                "disable_hybrid_kv_cache_manager",
                False,
            )
        )
        if self._compat_decline is not None:
            logger.error(
                "SemBlendVllmConnector declining all reuse: %s", self._compat_decline
            )
        if self._config.log_decisions:
            logger.info(
                "SemBlendVllmConnector initialized role=%s mode=%s provider=%s "
                "hybrid_kv_cache_manager_disabled=%s",
                role,
                self._config.mode.value,
                self._config.provider,
                hma_disabled,
            )
        self._audit_event(
            "connector_initialized",
            role=str(role),
            mode=self._config.mode.value,
            provider=self._config.provider,
            block_size=self._block_size,
            hybrid_kv_cache_manager_disabled=hma_disabled,
            compat_declined_reason=self._compat_decline,
            compat_check_incomplete=",".join(self._compat_check_incomplete) or None,
        )

    def _note_compat_check_incomplete(self, reason: str) -> None:
        """Record that part of the startup gate could not run.

        A skipped check must not read like a passed one: without this, a
        construction that never resolved a KV cache config is indistinguishable
        in the audit trail from an engine that was fully verified.
        """
        self._compat_check_incomplete.append(reason)
        self._stats[f"compat_check_incomplete_{reason}"] += 1
        logger.warning(
            "SemBlendVllmConnector engine compatibility check incomplete: %s", reason
        )

    def _check_engine_compatibility(self, vllm_config: "VllmConfig") -> str | None:
        """Why this engine configuration cannot be served; None when it can.

        Every destination this connector computes assumes token ``p`` lives at
        block ``p // block_size`` of a single dense group whose page holds one
        slot per KV head. Context parallelism moves the scheduler's alignment
        unit off ``block_size``, a second KV-cache group makes the group index
        ambiguous, a finer prefix-hash unit re-registers blocks the connector
        filled, a backend that packs head slots puts K and V in cells the write
        path infers wrongly, and a spec that does not keep one slot per token
        at a fixed block index has no such ``p`` to compute. None of those can
        be served correctly, so decline at startup rather than write KV to the
        wrong slots.
        """
        parallel_config = getattr(vllm_config, "parallel_config", None)
        decode_cp = int(getattr(parallel_config, "decode_context_parallel_size", 1) or 1)
        prefill_cp = int(getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
        if decode_cp != 1 or prefill_cp != 1:
            return (
                f"context parallelism is unsupported (decode_cp={decode_cp}, "
                f"prefill_cp={prefill_cp})"
            )

        groups = getattr(self._kv_cache_config, "kv_cache_groups", None)
        if not groups:
            # vLLM 0.29 passes a resolved kv_cache_config to both roles, so
            # this is a construction that never reached the engine; the group
            # and spec invariants below went unchecked either way.
            self._note_compat_check_incomplete("kv_cache_config_unresolved")
            return None
        if len(groups) != 1:
            return f"{len(groups)} KV-cache groups; only a single group is supported"

        block_size_decline = self._check_hash_block_size(vllm_config)
        if block_size_decline is not None:
            return block_size_decline

        try:
            from vllm.v1.kv_cache_interface import is_full_attention_spec, iter_layer_specs
        except ImportError:
            self._note_compat_check_incomplete("spec_import_failed")
            return None
        spec = getattr(groups[0], "kv_cache_spec", None)
        # vLLM publishes this predicate for exactly this question, and it
        # unwraps UniformTypeKVCacheSpecs. An identity check on the class
        # rejected a group that merely carries the wrapper, or a spec promoted
        # to a subclass that keeps the same page layout, while still admitting
        # nothing that is not full attention on every layer.
        if spec is None or not is_full_attention_spec(spec):
            return f"KV-cache spec {type(spec).__name__} is not plain full attention"
        for layer_spec in iter_layer_specs(spec):
            # num_head_slots is applied with dataclasses.replace on the same
            # spec class (FlashInfer-nvfp4, ROCm), so no class check can see
            # it. The page then carries one head_size per slot rather than a
            # packed K|V pair, and the element counts still multiply out: the
            # reshape and the indexed write both succeed with K and V in the
            # wrong cells.
            num_head_slots = getattr(layer_spec, "num_head_slots", None)
            num_kv_heads = getattr(layer_spec, "num_kv_heads", None)
            num_heads = getattr(layer_spec, "num_heads", num_kv_heads)
            if num_head_slots is not None or num_heads != num_kv_heads:
                return (
                    f"KV-cache spec packs {num_heads} head slots for "
                    f"{num_kv_heads} KV heads; the page is not one slot per KV head"
                )
            # The 4-D write path finds K and V by splitting the page's content
            # dim in half, so a spec whose V is narrower than its K has no half
            # to split at; the element counts still multiply out.
            head_size = getattr(layer_spec, "head_size", None)
            head_size_v = getattr(layer_spec, "head_size_v", head_size)
            if head_size is not None and head_size_v != head_size:
                return (
                    f"KV-cache spec packs head_size={head_size} with "
                    f"head_size_v={head_size_v}; K and V are not equal halves "
                    "of the page content dim"
                )
            # Every destination here is one slot per token. A spec that folds
            # several tokens into one state (sparse MLA) or several states into
            # one token (block pooling) makes token p's slot uncomputable, and
            # the predicate above admits both.
            tokens_per_state = getattr(layer_spec, "tokens_per_state", 1)
            if tokens_per_state != 1:
                return (
                    f"KV-cache spec stores {tokens_per_state} token(s) per state; "
                    "only one state per token can be addressed"
                )
            # A recycling sliding window evicts the gap blocks between the
            # prefill tail and the decode window, so block p // block_size
            # stops holding token p for the life of the request.
            if getattr(layer_spec, "rswa_window", None) is not None:
                return (
                    "KV-cache spec recycles sliding-window blocks; the block "
                    "index no longer addresses a fixed token position"
                )
        return None

    def _check_hash_block_size(self, vllm_config: "VllmConfig") -> str | None:
        """Decline unless vLLM hashes prefixes at the block size we index with.

        ``resolve_kv_cache_block_sizes`` is the engine's own answer for both
        the scheduler's alignment unit and the prefix-hash granularity, and it
        is what ``--prefix-match-unit`` actually feeds (only on the multi-group
        branch). Reading the flag directly declined single-group engines where
        vLLM ignores it. The DCP scaling is folded into the same answer.
        """
        try:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            _, hash_block_size = resolve_kv_cache_block_sizes(
                self._kv_cache_config, vllm_config
            )
        except ImportError:
            self._note_compat_check_incomplete("block_size_resolution_unavailable")
            return None
        except Exception:
            logger.warning("SemBlend KV block-size resolution failed", exc_info=True)
            self._note_compat_check_incomplete("block_size_resolution_failed")
            return None
        if int(hash_block_size) != self._block_size:
            return (
                f"prefix hash block size {int(hash_block_size)} differs from "
                f"connector block_size={self._block_size}"
            )
        return None

    @property
    def stats_snapshot(self) -> dict[str, int]:
        return dict(self._stats)

    def _token_ids(self, request: Any) -> list[int]:
        for attr in ("prompt_token_ids", "all_token_ids", "token_ids"):
            token_ids = getattr(request, attr, None)
            if token_ids:
                return list(token_ids)
        return []

    def _prompt_text(self, request: Any) -> str | None:
        if not self._config.enable_prompt_text:
            return None
        for attr in ("prompt", "prompt_text", "text"):
            value = getattr(request, attr, None)
            if isinstance(value, str):
                return value
        prompt_token_ids = getattr(request, "prompt_token_ids", None)
        return self._decode_prompt_tokens(prompt_token_ids or self._token_ids(request))

    def _decode_prompt_tokens(self, token_ids: Any) -> str | None:
        if not token_ids or self._prompt_tokenizer_failed:
            return None
        try:
            tokenizer = self._get_prompt_tokenizer()
            if tokenizer is None:
                return None
            text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
            return text.strip() or None
        except Exception:
            logger.warning("SemBlend prompt token decode failed", exc_info=True)
            self._prompt_tokenizer_failed = True
        return None

    def _get_prompt_tokenizer(self) -> Any | None:
        if self._prompt_tokenizer is not None:
            return self._prompt_tokenizer
        if self._prompt_tokenizer_failed:
            return None
        try:
            from transformers import AutoTokenizer

            model_config = getattr(self._vllm_config, "model_config", None)
            tokenizer_id = (
                getattr(model_config, "tokenizer", None)
                or self._config.model_id
                or getattr(model_config, "model", None)
            )
            if not tokenizer_id:
                self._prompt_tokenizer_failed = True
                return None
            self._prompt_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
            return self._prompt_tokenizer
        except Exception:
            logger.warning("SemBlend prompt tokenizer load failed", exc_info=True)
            self._prompt_tokenizer_failed = True
            return None

    def _request_id(self, request: Any) -> str:
        return str(getattr(request, "request_id", "unknown-request"))

    def _first_attempt_for(self, request_id: str) -> bool:
        """True only on this request's first visit to the match hook.

        The hook is re-entered on every scheduling attempt while a request
        waits for admission. The early exits above the lookup return before
        anything is memoized, so `first_lookup` cannot scope them; this can.
        """
        if request_id in self._seen_requests:
            return False
        self._seen_requests.add(request_id)
        return True

    def _audit_event(self, event: str, **fields: Any) -> None:
        if not self._config.audit_path:
            return
        record = {
            "schema_version": 1,
            "event": event,
            "source": "semblend_vllm_connector",
            "time_unix_s": time.time(),
            "mode": self._config.mode.value,
            **fields,
        }
        try:
            parent = os.path.dirname(self._config.audit_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._config.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            self._stats["audit_errors_total"] += 1
            logger.warning("SemBlend audit event write failed", exc_info=True)

    def _routing_metadata(self, request: Any) -> dict[str, str]:
        """Extract tenant/template routing keys from common vLLM request shapes.

        Gateways differ in where they place request-scoped metadata. Keep this
        conservative and canonicalize only the keys Synapse uses for donor reuse
        policy; unknown metadata stays out of the fleet-routing contract.
        """
        mappings: list[Mapping[str, Any]] = []
        for attr in (
            "metadata",
            "request_metadata",
            "extra_metadata",
            "extra_headers",
            "trace_headers",
        ):
            value = getattr(request, attr, None)
            if isinstance(value, Mapping):
                mappings.append(value)

        normalized: dict[str, Any] = {}
        for mapping in mappings:
            for key, value in mapping.items():
                normalized[str(key).lower().replace("-", "_")] = value

        tenant = _first_non_empty(
            getattr(request, "tenant_id", None),
            getattr(request, "tenant", None),
            normalized.get("tenant_id"),
            normalized.get("tenant"),
            normalized.get("x_synapse_tenant"),
            normalized.get("x_worldflow_tenant"),
        )
        template = _first_non_empty(
            getattr(request, "template_id", None),
            getattr(request, "template", None),
            getattr(request, "chat_template", None),
            normalized.get("template_id"),
            normalized.get("template"),
            normalized.get("chat_template"),
            normalized.get("x_synapse_template"),
            normalized.get("x_worldflow_template"),
        )

        metadata: dict[str, str] = {}
        if tenant:
            metadata["tenant"] = tenant
        if template:
            metadata["template"] = template
        return metadata

    def _storage_key(self, donor_id: str, namespace: str) -> str:
        raw = f"{namespace}:{donor_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _donor_dir_for_key(self, storage_key: str) -> str:
        return os.path.join(self._config.kv_storage_path, storage_key)

    def _donor_dir(self, donor_id: str, namespace: str) -> str:
        return self._donor_dir_for_key(self._storage_key(donor_id, namespace))

    def _has_stored_donor(self, donor_id: str, namespace: str) -> bool:
        if self._storage_key(donor_id, namespace) in self._memory_store:
            return True
        return os.path.isdir(self._donor_dir(donor_id, namespace))

    def _materialization_enabled(self) -> bool:
        if self._compat_decline is not None:
            return False
        return self._config.mode in {
            ReuseMode.EXACT_PREFIX,
            ReuseMode.REQUEST_ONLY_EXPERIMENTAL,
            ReuseMode.SEGMENTED_EXPERIMENTAL,
            ReuseMode.SEMANTIC_SPAN_EXPERIMENTAL,
        }

    def _destination_block_ids(self, load: PendingLoad) -> set[int]:
        """Blocks a load would have written, in the request's own block list."""
        if not load.block_ids or not load.block_ids[0]:
            return set()
        target_start = load.target_start or 0
        start_block = target_start // self._block_size
        end_block = -(-(target_start + load.token_count) // self._block_size)
        return {int(b) for b in load.block_ids[0][start_block:end_block]}

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Sync loading: report in the forward pass the failure was detected.
        failed, self._load_error_block_ids = self._load_error_block_ids, set()
        return failed

    def _slot_mapping(
        self,
        block_ids: tuple[list[int], ...],
        token_count: int,
        device: Any,
        target_start: int = 0,
    ):
        import torch

        if len(block_ids) != 1:
            self._stats["slot_mapping_multi_group_refused"] += 1
            self._audit_event(
                "slot_mapping_multi_group_refused",
                group_count=len(block_ids),
                target_start=int(target_start),
                token_count=int(token_count),
            )
            raise RuntimeError(
                f"SemBlend requires a single KV-cache group, got {len(block_ids)}"
            )
        # The destination is [target_start, target_start + token_count). Block
        # index 0 holds the request's shared, reference-counted exact-prefix
        # region; writing there corrupts every other request holding it and
        # leaves the region the scheduler marked computed uninitialized.
        start_block = target_start // self._block_size
        end_block = -(-(target_start + token_count) // self._block_size)
        group = block_ids[0][start_block:end_block]
        offset = target_start % self._block_size
        # A short block list silently yields a short slot mapping, which writes
        # a prefix of the window and leaves the rest of a region the scheduler
        # already marked computed uninitialized.
        covered = len(group) * self._block_size - offset
        if covered < token_count:
            self._stats["slot_mapping_window_not_covered"] += 1
            self._audit_event(
                "slot_mapping_window_not_covered",
                target_start=int(target_start),
                token_count=int(token_count),
                group_block_count=len(block_ids[0]),
                covered_tokens=int(max(covered, 0)),
            )
            raise RuntimeError(
                "SemBlend destination blocks do not cover the window: "
                f"target_start={target_start}, token_count={token_count}, "
                f"len(block_ids[0])={len(block_ids[0])} covers only "
                f"{max(covered, 0)} token(s) from target_start"
            )
        block_ids_tensor = torch.tensor(group, device=device)
        block_offsets = torch.arange(0, self._block_size, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_tensor.reshape((len(group), 1)) * self._block_size
        )
        return slot_mapping.flatten()[offset : offset + token_count]

    def _model_head_size(self) -> int | None:
        """The model's per-head KV width, or None when the config is silent."""
        model_config = getattr(self._vllm_config, "model_config", None)
        hf = getattr(model_config, "hf_config", None)
        head_dim = getattr(hf, "head_dim", None)
        if head_dim is None:
            hidden = getattr(hf, "hidden_size", None)
            heads = getattr(hf, "num_attention_heads", None)
            if hidden and heads:
                head_dim = hidden // heads
        return int(head_dim) if head_dim else None

    def _spec_kv_content_widths(self) -> tuple[int, int] | None:
        """(head_size, head_size_v) the KV-cache spec publishes for the group.

        The spec is the authority for the page's content dimension: a cell
        holds ``head_size + head_size_v``, and the two are not required to be
        equal (MLA carries no separate V at all). The HF head size answers the
        model's question, not the page's, so it can only stand in.
        """
        groups = getattr(self._kv_cache_config, "kv_cache_groups", None)
        if not groups:
            return None
        spec = getattr(groups[0], "kv_cache_spec", None)
        if spec is None:
            return None
        try:
            from vllm.v1.kv_cache_interface import iter_layer_specs

            layer_specs: Any = iter_layer_specs(spec)
        except ImportError:
            layer_specs = (spec,)
        widths: set[tuple[int, int]] = set()
        for layer_spec in layer_specs:
            head_size = getattr(layer_spec, "head_size", None)
            head_size_v = getattr(layer_spec, "head_size_v", None)
            if head_size is None or head_size_v is None:
                return None
            widths.add((int(head_size), int(head_size_v)))
        # Layers that disagree have no single expectation to check against.
        if len(widths) != 1:
            return None
        return widths.pop()

    def _expected_page_content_dim(self) -> tuple[int | None, str, bool]:
        """(expected content dim, source that answered, K and V halves equal).

        Resolved once: both sources are fixed for the life of the engine, so
        resolving per layer would only make the unresolved case count once per
        layer per load instead of once.
        """
        if self._content_dim_expectation is not None:
            return self._content_dim_expectation
        widths = self._spec_kv_content_widths()
        if widths is not None:
            head_size, head_size_v = widths
            self._content_dim_expectation = (
                head_size + head_size_v,
                "kv_cache_spec",
                head_size == head_size_v,
            )
        else:
            head_size = self._model_head_size()
            if head_size is None:
                self._stats["packed_content_dim_check_skipped_unknown_head_size"] += 1
                self._audit_event("packed_content_dim_check_skipped")
                self._content_dim_expectation = (None, "unresolved", True)
            else:
                self._content_dim_expectation = (2 * head_size, "hf_head_size", True)
        return self._content_dim_expectation

    def _require_packed_content_dim(self, shape: Any) -> None:
        """Refuse a 4-D page that does not pack K|V per KV head.

        The 4-D branches read ``heads = shape[1]`` and ``head_size =
        shape[3] // 2``. Where a backend publishes ``num_head_slots``
        (FlashInfer-nvfp4, ROCm) the page holds a single head_size per slot
        instead, and the element counts still multiply out: the reshape and the
        indexed access both succeed with K and V in the wrong cells. The
        startup gate catches this from the KV-cache spec; this is the same
        check against the tensor actually handed to the worker.
        """
        expected, source, halves_equal = self._expected_page_content_dim()
        if not halves_equal:
            # The width matches but the split does not: ``shape[3] // 2`` cuts
            # a page whose V is narrower than its K in the wrong place, and the
            # element counts still multiply out.
            self._stats["packed_asymmetric_kv_content_refused"] += 1
            self._audit_event(
                "packed_asymmetric_kv_content_refused",
                page_shape=[int(dim) for dim in shape],
                expected_content_dim=int(expected or 0),
            )
            raise RuntimeError(
                "KV-cache spec publishes an uneven K|V content dim "
                f"({expected}) for page shape {tuple(int(dim) for dim in shape)}: "
                "the write path splits the content dim in half, so K and V "
                "cannot be placed safely"
            )
        if expected is None:
            return
        content = int(shape[3])
        if content == expected:
            return
        # The counter names the source so a refusal read off a fallback answer
        # is never mistaken for one the engine's own spec disagreed with.
        self._stats[f"packed_head_slot_layout_refused_vs_{source}"] += 1
        self._audit_event(
            "packed_head_slot_layout_refused",
            page_shape=[int(dim) for dim in shape],
            expected_content_dim=int(expected),
            expected_content_dim_source=source,
        )
        raise RuntimeError(
            f"KV page content dim {content} is not the {expected} the {source} "
            f"publishes, for page shape {tuple(int(dim) for dim in shape)}: "
            "this backend packs head slots, so K and V cannot be placed safely"
        )

    def _is_mla_metadata(self, attn_metadata: Any) -> bool:
        try:
            from vllm.v1.attention.backends.mla.common import MLACommonMetadata
        except Exception:
            return False
        return isinstance(attn_metadata, MLACommonMetadata)

    def _layer_attn_metadata(self, attn_metadata: Any, layer_name: str) -> Any:
        """ForwardContext.attn_metadata is a per-layer-name dict (a list of
        such dicts under microbatching), so the layout checks must resolve the
        layer's own metadata first or they can never match."""
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0] if attn_metadata else None
        if isinstance(attn_metadata, dict):
            return attn_metadata.get(layer_name)
        return attn_metadata

    def _inject_kv_into_layer(
        self,
        dst_kv_cache_layer: Any,
        src_kv_cache: Any,
        slot_mapping: Any,
        attn_metadata: Any,
    ) -> None:
        dst_shape = dst_kv_cache_layer.shape
        if self._is_mla_metadata(attn_metadata):
            num_pages = dst_shape[0]
            page_size = dst_shape[1]
            dst = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
            dst[slot_mapping, ...] = src_kv_cache
            dst.reshape(dst_shape)
            return

        page_size = dst_shape[2]
        pages = slot_mapping // page_size
        offsets = slot_mapping % page_size
        n = int(slot_mapping.numel())
        if len(dst_shape) == 4:
            import torch

            self._require_packed_content_dim(dst_shape)
            heads = dst_shape[1]
            head_size = dst_shape[3] // 2
            kv = src_kv_cache.reshape(2, n, heads, head_size)
            dst_kv_cache_layer[pages, :, offsets, :] = torch.cat(
                (kv[0], kv[1]), dim=-1
            )
            return
        if dst_shape[0] == 2:
            dst_kv_cache_layer[:, pages, offsets, ...] = src_kv_cache.reshape(
                2, n, *dst_kv_cache_layer.shape[3:]
            )
        elif dst_shape[1] == 2:
            dst_kv_cache_layer[pages, :, offsets, ...] = src_kv_cache.reshape(
                2, n, *dst_kv_cache_layer.shape[3:]
            ).transpose(0, 1)
        else:
            raise RuntimeError(
                f"unrecognized paged KV layout {tuple(dst_shape)}"
            )

    def _extract_kv_from_layer(self, kv_layer: Any, slot_mapping: Any, attn_metadata: Any) -> Any:
        # Index pages/offsets directly: reshaping the whole paged layer
        # copies it when strides are not flat-contiguous, and the K/V axis
        # position varies by engine version ([2, blocks, bs, ...] vs
        # [blocks, 2, bs, ...]) — both misreads showed up as multi-GiB
        # allocations during donor capture.
        layer_shape = kv_layer.shape
        if self._is_mla_metadata(attn_metadata):
            page_size = layer_shape[1]
            pages = slot_mapping // page_size
            offsets = slot_mapping % page_size
            gathered = kv_layer[pages, offsets, ...]
            return gathered.reshape(gathered.shape[0], -1)

        page_size = layer_shape[2]
        pages = slot_mapping // page_size
        offsets = slot_mapping % page_size
        import torch

        if len(layer_shape) == 4:
            # flash_attn v1 packed-content layout:
            # (blocks, kv_heads, block_size, 2*head_size), K first half of
            # the content dim, V second (flash_attn.py split convention).
            self._require_packed_content_dim(layer_shape)
            page_size = layer_shape[2]
            pages = slot_mapping // page_size
            offsets = slot_mapping % page_size
            gathered = kv_layer[pages, :, offsets, :]  # [n, H, 2D]
            k, v = gathered.split(layer_shape[3] // 2, dim=-1)
            stacked = torch.stack((k, v))  # [2, n, H, D]
            return stacked.reshape(2, stacked.shape[1], -1)
        if layer_shape[0] == 2:
            gathered = kv_layer[:, pages, offsets, ...]
        elif layer_shape[1] == 2:
            # Separated advanced indices move to the front: result is
            # [n, 2, heads, dim]; put K/V first.
            gathered = kv_layer[pages, :, offsets, ...].transpose(0, 1)
        else:
            raise RuntimeError(
                f"unrecognized paged KV layout {tuple(layer_shape)}"
            )
        return gathered.reshape(2, gathered.shape[1], -1)

    def _layer_filename(self, donor_id: str, namespace: str, layer_name: str) -> str:
        return os.path.join(self._donor_dir(donor_id, namespace), f"{layer_name}.safetensors")

    def _donor_metadata_path(self, donor_id: str, namespace: str) -> str:
        return os.path.join(self._donor_dir(donor_id, namespace), _DONOR_METADATA_FILENAME)

    def _stored_donor_token_count(self, donor_id: str, namespace: str) -> int:
        storage_key = self._storage_key(donor_id, namespace)
        entry = self._memory_store.get(storage_key)
        if entry is not None:
            # A read is a use: without it the cap evicts in arrival order.
            self._memory_store.move_to_end(storage_key)
            return int(entry.get("__token_count__", 0))
        try:
            with open(self._donor_metadata_path(donor_id, namespace), encoding="utf-8") as f:
                metadata = json.load(f)
        except OSError:
            return 0
        try:
            return int(metadata.get("token_count", 0))
        except (TypeError, ValueError):
            return 0

    def _record_load_plan(
        self,
        request_id: str,
        *,
        donor_id: str,
        token_count: int,
        materialization_kind: MaterializationKind,
        namespace: str,
        donor_start: int | None = None,
        target_start: int | None = None,
    ) -> bool:
        """Memoize a load decision for update_state_after_alloc to build on.

        The match hook is re-entered per scheduling attempt and there are many
        scheduler exits between it and the allocation hook, so nothing recorded
        here is visible to the worker until blocks actually exist.

        Returns True only the first time a request records a plan: the
        per-request reuse metrics join on the advertise event, so re-entering
        the hook while a request waits for admission must not count again.
        """
        first_record = request_id not in self._load_plans
        self._load_plans[request_id] = {
            "donor_id": donor_id,
            "token_count": int(token_count),
            "materialization_kind": materialization_kind,
            "namespace": namespace,
            "donor_start": donor_start,
            "target_start": target_start,
        }
        return first_record

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        token_ids = self._token_ids(request)
        request_id = self._request_id(request)
        # Everything above the lookup is a request- or engine-stable decision,
        # so it is counted once per request rather than once per attempt.
        first_attempt = self._first_attempt_for(request_id)
        if first_attempt:
            self._stats["lookups_total"] += 1
        if self._compat_decline is not None:
            if first_attempt:
                self._stats["skipped_incompatible_engine_config"] += 1
            return 0, False

        if not token_ids or len(token_ids) < self._config.min_prompt_tokens:
            if first_attempt:
                self._stats["skipped_short_prompt"] += 1
            if self._config.log_decisions:
                logger.info(
                    "SemBlend lookup skipped request_id=%s reason=short_prompt tokens=%d min=%d",
                    request_id,
                    len(token_ids),
                    self._config.min_prompt_tokens,
                )
            return 0, False

        exact_ratio = num_computed_tokens / max(len(token_ids), 1)
        if exact_ratio >= self._config.skip_when_exact_prefix_ratio_at_least:
            # Attempt-scoped on purpose: the ratio moves with
            # num_computed_tokens, so a request can clear the threshold at one
            # boundary and not at the next, and gating would hide the skip.
            self._stats["skipped_exact_prefix_sufficient_per_attempt"] += 1
            if self._config.log_decisions:
                logger.info(
                    "SemBlend lookup skipped request_id=%s reason=exact_prefix_sufficient "
                    "exact_ratio=%.3f threshold=%.3f",
                    request_id,
                    exact_ratio,
                    self._config.skip_when_exact_prefix_ratio_at_least,
                )
            return 0, False

        if self._provider is None:
            if first_attempt:
                self._stats["skipped_no_provider"] += 1
            if self._config.log_decisions:
                logger.info("SemBlend lookup skipped request_id=%s reason=no_provider", request_id)
            return 0, False

        namespace = namespace_for_request(self._config, self._vllm_config, request)
        lookup = SemanticLookupRequest(
            request_id=request_id,
            token_ids=token_ids,
            prompt_text=self._prompt_text(request),
            model_id=model_id_from_config(self._config, self._vllm_config),
            namespace=namespace,
            already_computed_tokens=num_computed_tokens,
        )

        # A waiting request re-enters this hook on every scheduling step. The
        # lookup itself is memoized; the counters and audit events have to be
        # too, or the per-request reuse metrics they feed scale with queueing
        # depth instead of with traffic.
        first_lookup = request_id not in self._lookup_cache
        if not first_lookup:
            result = self._lookup_cache[request_id]
            elapsed_ms = 0
        else:
            started = time.monotonic()
            try:
                result = self._provider.lookup(lookup)
            except Exception:
                logger.exception(
                    "SemBlend provider lookup failed; falling back to normal prefill"
                )
                self._stats["provider_errors_total"] += 1
                return 0, False
            finally:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                self._stats["lookup_latency_ms_sum"] += elapsed_ms
            self._lookup_cache[request_id] = result

        if result is None:
            if first_lookup:
                self._stats["semantic_misses_total"] += 1
                self._audit_event(
                    "semantic_lookup_miss",
                    request_id=request_id,
                    namespace=namespace,
                    latency_ms=elapsed_ms,
                )
            if self._config.log_decisions and first_lookup:
                logger.info(
                    "SemBlend semantic lookup miss request_id=%s namespace=%s latency_ms=%d",
                    request_id,
                    namespace,
                    elapsed_ms,
                )
            return 0, False

        if first_lookup:
            self._stats["semantic_hits_total"] += 1
            self._audit_event(
                "semantic_lookup_hit",
                request_id=request_id,
                donor_id=result.donor_id,
                namespace=namespace,
                similarity=float(result.similarity),
                materialization_kind=result.materialization_kind.value,
                reusable_tokens=int(result.reusable_token_count),
                reason=result.reason,
                latency_ms=elapsed_ms,
                already_computed_tokens=int(num_computed_tokens),
                confidence_tier=str(
                    (result.quality_signals or {}).get("confidence_tier", "unknown")
                ),
            )
        if self._config.log_decisions and first_lookup:
            logger.info(
                "SemBlend semantic lookup hit request_id=%s donor_id=%s similarity=%.4f "
                "kind=%s reusable_tokens=%d reason=%s latency_ms=%d",
                request_id,
                result.donor_id,
                result.similarity,
                result.materialization_kind.value,
                result.reusable_token_count,
                result.reason,
                elapsed_ms,
            )
        if self._config.mode == ReuseMode.DISCOVERY_ONLY:
            if first_lookup:
                self._stats["materialization_suppressed_by_mode"] += 1
                self._audit_event(
                    "materialization_suppressed_by_mode",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    materialization_kind=result.materialization_kind.value,
                )
                if result.materialization_kind == MaterializationKind.DISCOVERY_ONLY:
                    self._stats["discovery_only_hits_total"] += 1
            return 0, False

        if (
            self._config.mode == ReuseMode.SEMANTIC_SPAN_EXPERIMENTAL
            and result.segments
        ):
            if num_computed_tokens % self._block_size != 0:
                # supply_at_boundary counts from the next block edge, but the
                # scheduler adds what this hook returns to the boundary itself.
                # Off an aligned boundary the two frames differ and the tokens
                # in between are marked computed yet never written.
                #
                # This decline and the two clamps below are decided from
                # num_computed_tokens, which moves between attempts, so they
                # stay attempt-scoped: gating them on first_lookup would
                # silently swallow a decline taken at a later boundary.
                self._stats["semantic_span_declined_unaligned_boundary_per_attempt"] += 1
                self._audit_event(
                    "semantic_span_declined_unaligned_boundary",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    boundary=int(num_computed_tokens),
                    block_size=self._block_size,
                )
                return 0, False
            # The donor KV on disk is a block-aligned prefix of the donor's
            # first scheduled chunk; a span reaching past it would load a
            # short tensor and fail the engine at inject. Trim every span
            # to the captured window (no capture -> nothing servable).
            stored_tokens = self._stored_donor_token_count(result.donor_id, namespace)
            raw_spans = []
            for seg in result.segments:
                if seg.donor_id != result.donor_id:
                    continue
                length = min(seg.token_count, stored_tokens - seg.donor_start)
                if length <= 0:
                    continue
                raw_spans.append(
                    {
                        "target_start": seg.target_start,
                        "length": length,
                        "donor_start": seg.donor_start,
                    }
                )
            spans = block_align_spans(
                raw_spans, self._block_size, self._config.min_semantic_span
            )
            token_count, donor_start = supply_at_boundary(
                spans, num_computed_tokens, self._block_size
            )
            token_count = min(
                token_count,
                (self._config.max_materialized_tokens // self._block_size)
                * self._block_size,
            )
            # The advertised count is added straight into num_computed_tokens
            # and the scheduler then asserts num_new_tokens > 0, so a span
            # reaching the prompt end kills the engine core. Bound it by the
            # same cacheable prefix vLLM's own local lookup respects.
            headroom = max(
                0,
                _cacheable_prefix_tokens(len(token_ids), self._block_size)
                - num_computed_tokens,
            )
            if token_count > headroom:
                self._stats["semantic_span_clamped_to_prompt_headroom_per_attempt"] += 1
                self._audit_event(
                    "semantic_span_supply_clamped",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    boundary=int(num_computed_tokens),
                    prompt_tokens=len(token_ids),
                    requested_tokens=int(token_count),
                    clamped_tokens=int(headroom),
                )
                token_count = headroom
            # block_align_spans enforces the operator's floor, but both clamps
            # above run after it and can cut a span to a single block. Serving
            # a 16-token span under a 512-token policy breaks that policy and
            # fills the boundary-alignment metric with trivial spans.
            if 0 < token_count < self._config.min_semantic_span:
                self._stats["semantic_span_declined_below_min_after_clamp_per_attempt"] += 1
                self._audit_event(
                    "semantic_span_declined_below_min_after_clamp",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    token_count=int(token_count),
                    min_semantic_span=int(self._config.min_semantic_span),
                    boundary=int(num_computed_tokens),
                )
                return 0, False
            if token_count > 0 and donor_start is not None:
                target_start = (
                    (num_computed_tokens + self._block_size - 1) // self._block_size
                ) * self._block_size
                first_advertise = self._record_load_plan(
                    request_id,
                    donor_id=result.donor_id,
                    token_count=token_count,
                    materialization_kind=MaterializationKind.SEMANTIC_SPAN,
                    namespace=namespace,
                    donor_start=donor_start,
                    target_start=target_start,
                )
                if first_advertise:
                    self._stats["semantic_span_loads_advertised_total"] += 1
                    self._audit_event(
                        "semantic_span_load_advertised",
                        request_id=request_id,
                        donor_id=result.donor_id,
                        namespace=namespace,
                        token_count=token_count,
                        donor_start=donor_start,
                        target_start=target_start,
                        boundary=int(num_computed_tokens),
                    )
                return token_count, False
            return 0, False

        if (
            self._config.mode == ReuseMode.REQUEST_ONLY_EXPERIMENTAL
            and result.donor_token_ids
            and self._has_stored_donor(result.donor_id, namespace)
            and num_computed_tokens == 0
        ):
            donor_token_ids = list(result.donor_token_ids)
            common_prefix = _common_prefix_len(donor_token_ids, token_ids)
            max_candidate_tokens = min(
                len(token_ids),
                len(donor_token_ids),
                self._stored_donor_token_count(result.donor_id, namespace),
                self._config.max_materialized_tokens,
            )
            if not self._config.allow_non_identical_request_only:
                max_candidate_tokens = min(max_candidate_tokens, common_prefix)
            token_count = _cacheable_prefix_tokens(max_candidate_tokens, self._block_size)
            if token_count > 0:
                first_advertise = self._record_load_plan(
                    request_id,
                    donor_id=result.donor_id,
                    token_count=token_count,
                    materialization_kind=MaterializationKind.REQUEST_ONLY,
                    namespace=namespace,
                )
                if first_advertise:
                    self._stats["request_only_loads_advertised_total"] += 1
                    self._audit_event(
                        "request_only_load_advertised",
                        request_id=request_id,
                        donor_id=result.donor_id,
                        namespace=namespace,
                        tokens=int(token_count),
                        materialization_kind=MaterializationKind.REQUEST_ONLY.value,
                        common_prefix_tokens=int(common_prefix),
                        allow_non_identical_request_only=bool(
                            self._config.allow_non_identical_request_only
                        ),
                    )
                    if self._config.log_decisions:
                        logger.info(
                            "SemBlend request-local experimental load advertised "
                            "request_id=%s donor_id=%s tokens=%d",
                            request_id,
                            result.donor_id,
                            token_count,
                        )
                return token_count, False
            # The branch only runs at boundary 0 and the common prefix is a
            # property of the two token lists, so this rejection is the same on
            # every attempt: count and audit it once per request.
            if first_lookup:
                self._stats["request_only_loads_rejected_non_identical_prefix"] += int(
                    not self._config.allow_non_identical_request_only
                    and common_prefix < len(token_ids)
                )
                self._audit_event(
                    "request_only_load_rejected",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    common_prefix_tokens=int(common_prefix),
                    candidate_tokens=int(max_candidate_tokens),
                    reason="non_identical_prefix",
                )

        if result.materialization_kind == MaterializationKind.DISCOVERY_ONLY:
            # Twin of the DISCOVERY_ONLY-mode site above: the kind comes from
            # the memoized lookup, so it is one fact about the request.
            if first_lookup:
                self._stats["discovery_only_hits_total"] += 1
            return 0, False

        if (
            self._config.mode == ReuseMode.EXACT_PREFIX
            and result.materialization_kind == MaterializationKind.EXACT_PREFIX
            and result.block_refs
            and result.reusable_token_count > 0
        ):
            # An exact-prefix reusable count is a position in the prompt
            # measured from token 0, and the donor's capture is indexed by that
            # same position. vLLM's local prefix cache has already computed
            # [0, num_computed_tokens) and the scheduler credits exactly
            # [num_computed_tokens, num_computed_tokens + advertised), so the
            # servable window is what the match reaches past that boundary --
            # in both frames at once. Keeping the two ends separate:
            #
            #   donor bound  - how far the exact match runs, in whole blocks.
            #     A plain truncation: this bound says nothing about leaving the
            #     prompt a token to compute, so the cacheable-prefix rule (one
            #     token held back) does not belong on it.
            reusable_end = (
                min(int(result.reusable_token_count), len(token_ids))
                // self._block_size
            ) * self._block_size
            #   prompt bound - the advertised count is added to
            #     num_computed_tokens and the scheduler then asserts there is at
            #     least one token left to compute, so a provider reporting the
            #     whole prompt cannot be taken at its word.
            prompt_end = _cacheable_prefix_tokens(len(token_ids), self._block_size)
            token_count = max(0, min(reusable_end, prompt_end) - num_computed_tokens)
            #   operator bound - a budget of tokens to write, not a position.
            token_count = (
                min(token_count, self._config.max_materialized_tokens)
                // self._block_size
            ) * self._block_size
            if token_count <= 0:
                # Attempt-scoped: the window shrinks as the local prefix grows,
                # so this decline is a fact about the boundary, not the request.
                self._stats["exact_prefix_declined_below_one_block_after_clamp_per_attempt"] += 1
                self._audit_event(
                    "exact_prefix_load_declined",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    reusable_tokens=int(result.reusable_token_count),
                    prompt_tokens=len(token_ids),
                    boundary=int(num_computed_tokens),
                    reason="below_one_block_after_clamp",
                )
                return 0, False
            first_advertise = self._record_load_plan(
                request_id,
                donor_id=result.donor_id,
                token_count=token_count,
                materialization_kind=result.materialization_kind,
                namespace=namespace,
                # Both frames travel with the plan. Without them the worker
                # writes [0, token_count) from donor token 0, which overwrites
                # the reference-counted blocks vLLM's own prefix cache handed
                # this request and leaves the credited window uninitialized.
                donor_start=num_computed_tokens,
                target_start=num_computed_tokens,
            )
            if first_advertise:
                self._stats["exact_prefix_loads_advertised_total"] += 1
                self._audit_event(
                    "exact_prefix_load_advertised",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    tokens=int(token_count),
                    reusable_tokens=int(result.reusable_token_count),
                    materialization_kind=result.materialization_kind.value,
                    donor_start=int(num_computed_tokens),
                    target_start=int(num_computed_tokens),
                    boundary=int(num_computed_tokens),
                )
            return token_count, False

        # Reached from the memoized lookup result and the configured mode, both
        # request-stable, so count and audit the fall-through once.
        if first_lookup:
            self._stats["materialization_rejected_no_safe_plan"] += 1
            self._audit_event(
                "materialization_rejected_no_safe_plan",
                request_id=request_id,
                donor_id=result.donor_id,
                namespace=namespace,
                materialization_kind=result.materialization_kind.value,
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        request_id = self._request_id(request)
        # The scheduler dropped the advertised span (e.g. a partial-tail loss):
        # the plan is stale and must not become a load.
        plan = self._load_plans.pop(request_id, None)
        if num_external_tokens <= 0:
            return
        if plan is None:
            self._stats["alloc_without_pending_load"] += 1
            return
        get_block_ids = getattr(blocks, "get_block_ids", None)
        block_ids = get_block_ids(allow_none=True) if callable(get_block_ids) else None
        if block_ids is None:
            # build_connector_meta would drop this load anyway. Declining here
            # keeps the request out of the served set (so it stays eligible as
            # a donor) and keeps load_allocated out of the reuse join for a
            # request nothing was ever served to.
            self._stats["load_declined_no_blocks"] += 1
            self._audit_event(
                "load_declined_no_blocks",
                request_id=request_id,
                donor_id=plan["donor_id"],
                namespace=plan["namespace"],
                tokens=int(plan["token_count"]),
                materialization_kind=plan["materialization_kind"].value,
            )
            return
        self._pending_loads[request_id] = PendingLoad(
            request_id=request_id,
            donor_id=plan["donor_id"],
            token_count=plan["token_count"],
            materialization_kind=plan["materialization_kind"],
            namespace=plan["namespace"],
            block_ids=block_ids,
            donor_start=plan["donor_start"],
            target_start=plan["target_start"],
        )
        self._served_request_ids.add(request_id)
        self._audit_event(
            "load_allocated",
            request_id=request_id,
            donor_id=plan["donor_id"],
            namespace=plan["namespace"],
            tokens=int(plan["token_count"]),
            materialization_kind=plan["materialization_kind"].value,
            block_id_count=sum(len(group) for group in block_ids),
        )

    def _build_store_metadata(self, scheduler_output: "SchedulerOutput") -> list[PendingStore]:
        if not self._materialization_enabled():
            return []
        for finished_id in getattr(scheduler_output, "finished_req_ids", None) or ():
            self._capture_state.pop(str(finished_id), None)
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
        return [
            *self._capture_admitted(scheduler_output, scheduled_tokens),
            *self._capture_continuations(scheduler_output, scheduled_tokens),
        ]

    def _capture_admitted(
        self, scheduler_output: "SchedulerOutput", scheduled_tokens: Mapping[str, int] | None
    ) -> list[PendingStore]:
        """Open a capture for each request scheduled for the first time."""
        stores: list[PendingStore] = []
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            token_ids = self._token_ids(new_req)
            if len(token_ids) < self._config.min_prompt_tokens:
                continue
            request_id = str(
                getattr(new_req, "req_id", None)
                or getattr(new_req, "request_id", None)
                or "unknown-request"
            )
            if not self._capture_allowed(request_id):
                continue
            block_ids = _normalize_block_ids(getattr(new_req, "block_ids", None))
            if block_ids is None:
                continue
            state = _CaptureState(
                token_ids=token_ids,
                block_ids=block_ids,
                namespace=namespace_for_request(self._config, self._vllm_config, new_req),
                captured_end=0,
            )
            # num_computed_tokens is the local prefix hit (plus anything a
            # connector served), already valid in the shared blocks; the
            # forward pass adds num_scheduled_tokens on top of it.
            computed_end = self._computed_end(
                request_id,
                int(getattr(new_req, "num_computed_tokens", 0) or 0),
                len(token_ids),
                scheduled_tokens,
            )
            store = self._advance_capture(request_id, state, computed_end)
            if store is not None:
                stores.append(store)
        return stores

    def _capture_continuations(
        self, scheduler_output: "SchedulerOutput", scheduled_tokens: Mapping[str, int] | None
    ) -> list[PendingStore]:
        """Extend open captures with the chunks scheduled after admission."""
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None or not self._capture_state:
            return []
        # For ids in resumed_req_ids the diff is the whole block table, not
        # an addition to it (vLLM SchedulerOutput.CachedRequestData).
        resumed = {str(req_id) for req_id in getattr(cached, "resumed_req_ids", None) or ()}
        rows = zip(
            cached.req_ids,
            cached.new_block_ids,
            cached.num_computed_tokens,
            cached.num_output_tokens,
            strict=True,
        )
        stores: list[PendingStore] = []
        for req_id, new_block_ids, num_computed_tokens, num_output_tokens in rows:
            request_id = str(req_id)
            state = self._capture_state.get(request_id)
            if state is None:
                continue
            if not self._capture_allowed(request_id):
                self._capture_state.pop(request_id, None)
                continue
            if int(num_output_tokens or 0) > 0:
                # Decode: the prompt is fully computed, so a capture still
                # open here is short of its cap for good (the block table
                # never covered it); close it so the entry does not linger.
                self._stats["capture_closed_short_at_decode"] += 1
                self._capture_state.pop(request_id, None)
                continue
            block_ids = self._extend_block_table(
                state.block_ids,
                _normalize_block_ids(new_block_ids),
                replace_table=request_id in resumed,
            )
            computed_end = self._computed_end(
                request_id, int(num_computed_tokens or 0), len(state.token_ids), scheduled_tokens
            )
            store = self._advance_capture(
                request_id, replace(state, block_ids=block_ids), computed_end
            )
            if store is not None:
                stores.append(store)
        return stores

    def _capture_allowed(self, request_id: str) -> bool:
        """Whether a request served by this connector may be captured too.

        The served set is consumed here: a request is served once, and the
        decision must be taken on the step the load landed.
        """
        if request_id not in self._served_request_ids:
            return True
        self._served_request_ids.discard(request_id)
        if self._config.capture_served_requests:
            return True
        self._stats["capture_skipped_served"] += 1
        return False

    def _computed_end(
        self,
        request_id: str,
        num_computed_tokens: int,
        prompt_tokens: int,
        scheduled_tokens: Mapping[str, int] | None,
    ) -> int:
        """Prompt position computed once this step's forward pass has run."""
        if scheduled_tokens is None:
            # Stock vLLM always carries num_scheduled_tokens; only a stub
            # engine omits it. Assume the whole prompt, and keep the
            # unclamped capture visible in the stats.
            self._stats["capture_unclamped_no_schedule_info"] += 1
            return prompt_tokens
        return num_computed_tokens + int(scheduled_tokens.get(request_id, 0) or 0)

    @staticmethod
    def _extend_block_table(
        current: tuple[list[int], ...],
        added: tuple[list[int], ...] | None,
        *,
        replace_table: bool,
    ) -> tuple[list[int], ...]:
        if added is None:
            return current
        if replace_table:
            return added
        return tuple([*old, *new] for old, new in zip(current, added, strict=True))

    def _advance_capture(
        self, request_id: str, state: _CaptureState, computed_end: int
    ) -> PendingStore | None:
        """The store that extends this donor to what is now computed, if any.

        The capture is clamped to whole blocks the forward pass has actually
        filled: under chunked prefill the admission step computes at most
        max_num_batched_tokens (shared across co-scheduled requests), and a
        capture sized from the prompt alone publishes uninitialized KV as the
        donor's authoritative length. Later chunks append from the running
        offset. The state is dropped once the prompt's cacheable prefix is
        captured in full.
        """
        prompt_cap = _cacheable_prefix_tokens(len(state.token_ids), self._block_size)
        if prompt_cap <= 0:
            self._capture_state.pop(request_id, None)
            return None
        end = min(
            prompt_cap,
            len(state.block_ids[0]) * self._block_size,
            (computed_end // self._block_size) * self._block_size,
        )
        if end <= state.captured_end:
            # Nothing new is block-complete yet; keep the running table.
            self._capture_state[request_id] = state
            return None
        if end >= prompt_cap:
            self._capture_state.pop(request_id, None)
        else:
            self._capture_state[request_id] = replace(state, captured_end=end)
        return PendingStore(
            request_id=request_id,
            token_ids=state.token_ids,
            token_count=end,
            namespace=state.namespace,
            block_ids=state.block_ids,
        )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> SemBlendConnectorMetadata:
        loads: list[PendingLoad] = []
        for load in self._pending_loads.values():
            if load.block_ids is None:
                self._stats["load_declined_no_blocks"] += 1
                self._audit_event(
                    "load_declined_no_blocks",
                    request_id=load.request_id,
                    donor_id=load.donor_id,
                    namespace=load.namespace,
                    tokens=int(load.token_count),
                    materialization_kind=load.materialization_kind.value,
                )
                continue
            loads.append(load)
        stores = self._build_store_metadata(scheduler_output)
        self._pending_loads.clear()
        return SemBlendConnectorMetadata(loads=loads, stores=stores)

    def _rope_params(self) -> tuple[float, int] | None:
        """(rope_theta, head_dim) from the model config; None when unknown.

        transformers 5.x carries theta in the unified rope_parameters
        dict instead of a flat rope_theta attribute, and may key that dict
        by layer type (interleaved attention) rather than carry one flat
        dict, so every sub-dict has to clear the guard. Scaled rope types
        (yarn, linear, llama3, dynamic) and partial rotary embeddings
        rotate K differently from the plain re-rotation the realizer
        applies, so anything but the default declines the load. The
        scaling dict is inspected unconditionally because models such as
        Llama-3.1/3.3 expose a flat rope_theta alongside it; keying off
        the flat attribute would skip the guard on exactly those models.
        """
        model_config = getattr(self._vllm_config, "model_config", None)
        hf = getattr(model_config, "hf_config", None)
        rope_sets: list[Mapping[str, Any]] | None = None
        for attr in ("rope_parameters", "rope_scaling"):
            value = getattr(hf, attr, None)
            if isinstance(value, Mapping) and value:
                rope_sets = _rope_parameter_sets(value)
                if rope_sets is None:
                    return None
                break
        for rope in rope_sets or ():
            rope_type = str(rope.get("rope_type") or rope.get("type") or "default")
            if rope_type != "default":
                return None
            factor = rope.get("partial_rotary_factor")
            if factor is not None and float(factor) != 1.0:
                return None
        partial_rotary_factor = getattr(hf, "partial_rotary_factor", None)
        if partial_rotary_factor is not None and float(partial_rotary_factor) != 1.0:
            return None
        # Interleaved layer types may rotate at different base frequencies; one
        # constant-delta re-rotation cannot serve two of them, so theta is only
        # resolved from sub-dicts that agree on it.
        thetas = {
            float(rope["rope_theta"])
            for rope in (rope_sets or ())
            if rope.get("rope_theta") is not None
        }
        if len(thetas) > 1:
            return None
        theta = thetas.pop() if thetas else getattr(hf, "rope_theta", None)
        head_dim = self._model_head_size()
        if theta is None or head_dim is None:
            return None
        return float(theta), int(head_dim)

    def _semantic_span_slice(self, src_kv_cache, load, attn_metadata):
        """Donor-offset slice + K re-rotation for a semantic-span load.

        Returns None (declining the layer) for MLA layouts or when rope
        parameters are unavailable; callers fall back to normal compute.
        """
        import torch

        from semblend_vllm_connector.semantic_span import rerotate_k

        if self._is_mla_metadata(attn_metadata):
            self._stats["semantic_span_declined_mla"] += 1
            return None
        params = self._rope_params()
        if params is None or load.donor_start is None or load.target_start is None:
            self._stats["semantic_span_declined_no_rope_params"] += 1
            return None
        theta, head_dim = params
        window = src_kv_cache[
            :, load.donor_start : load.donor_start + load.token_count, ...
        ]
        if window.shape[1] < load.token_count:
            self._stats["semantic_span_declined_short_donor"] += 1
            return None
        # Stored KV is [2, n, H*D] (flattened); rotation operates per head.
        n_tok = window.shape[1]
        k_heads = window[0].reshape(n_tok, -1, head_dim)
        k = rerotate_k(
            k_heads,
            donor_start=load.donor_start,
            target_start=load.target_start,
            head_dim=head_dim,
            rope_theta=theta,
        ).reshape(n_tok, -1)
        return torch.stack((k, window[1]))

    def register_kv_caches(self, kv_caches: dict) -> None:
        self._registered_kv_caches = dict(kv_caches)
        self._audit_event(
            "kv_caches_registered", layer_count=len(self._registered_kv_caches)
        )

    def _iter_kv_layers(self, forward_context):
        """Yield (layer_name, dst_kv_cache_tensor) across vLLM versions."""
        if self._registered_kv_caches:
            yield from self._registered_kv_caches.items()
            return
        for layer_name, layer in getattr(
            forward_context, "no_compile_layers", {}
        ).items():
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue
            yield layer_name, kv_cache_attr[get_virtual_engine(forward_context)]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SemBlendConnectorMetadata) or not metadata.loads:
            return

        from safetensors.torch import load_file

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if attn_metadata is None:
            # vLLM runs a no-forward step (kv_connector.no_forward) when a
            # scheduling step has loads but no tokens to compute, common under
            # concurrent long prefills. The scheduler has already counted the
            # span as computed, so the load must happen now; attn_metadata is
            # only consulted for the MLA layout check, which stays as last seen.
            attn_metadata = self._last_attn_metadata
            self._stats["materialized_without_forward"] += 1
        else:
            self._last_attn_metadata = attn_metadata

        for load in metadata.loads:
            if load.block_ids is None:
                raise RuntimeError(f"SemBlend load missing destination blocks: {load.request_id}")
            self._audit_event(
                "runtime_materialization_started",
                request_id=load.request_id,
                donor_id=load.donor_id,
                namespace=load.namespace,
                tokens=int(load.token_count),
                materialization_kind=load.materialization_kind.value,
            )
            if self._config.log_decisions:
                logger.info(
                    "SemBlend materializing request_id=%s donor_id=%s tokens=%d",
                    load.request_id,
                    load.donor_id,
                    load.token_count,
                )
            layers_materialized = 0
            donor_gone = False
            # The donor capture is indexed by absolute token position, so a
            # load whose destination starts past 0 has to read from the same
            # offset: donor_start and target_start are one frame, and the
            # exact-prefix path sets both to the local prefix-cache boundary.
            donor_start = load.donor_start or 0
            storage_key = self._storage_key(load.donor_id, load.namespace)
            entry = self._memory_store.get(storage_key)
            if entry is not None:
                # A load is a use: without it the cap evicts in arrival order.
                self._memory_store.move_to_end(storage_key)
            for layer_name, dst_kv_cache_layer in self._iter_kv_layers(forward_context):
                layer_metadata = self._layer_attn_metadata(attn_metadata, layer_name)
                if layer_metadata is None:
                    # No per-layer metadata to read the layout from (the
                    # no-forward step, or a dict that does not carry this
                    # layer). The MLA check below then answers "not MLA" by
                    # default, so keep the gap visible rather than silent.
                    self._stats["layer_attn_metadata_missing"] += 1
                if entry is not None and layer_name in entry:
                    src_kv_cache = entry[layer_name].to(dst_kv_cache_layer.device)
                else:
                    filename = self._layer_filename(load.donor_id, load.namespace, layer_name)
                    try:
                        tensors = load_file(filename)
                    except OSError:
                        # Evicted (or never fully captured) between the
                        # scheduler's advertise and this load. Whole donors
                        # go, not layers, so one miss decides the load.
                        donor_gone = True
                        break
                    src_kv_cache = tensors["kv_cache"].to(dst_kv_cache_layer.device)
                if load.materialization_kind == MaterializationKind.SEMANTIC_SPAN:
                    src_kv_cache = self._semantic_span_slice(
                        src_kv_cache, load, layer_metadata
                    )
                    if src_kv_cache is None:
                        continue
                elif self._is_mla_metadata(layer_metadata):
                    src_kv_cache = src_kv_cache[
                        donor_start : donor_start + load.token_count, ...
                    ]
                else:
                    src_kv_cache = src_kv_cache[
                        :, donor_start : donor_start + load.token_count, ...
                    ]
                slot_mapping = self._slot_mapping(
                    load.block_ids,
                    load.token_count,
                    dst_kv_cache_layer.device,
                    target_start=load.target_start or 0,
                )
                self._inject_kv_into_layer(
                    dst_kv_cache_layer,
                    src_kv_cache,
                    slot_mapping,
                    layer_metadata,
                )
                layers_materialized += 1
            if donor_gone:
                self._stats["load_declined_donor_gone"] += 1
                self._audit_event(
                    "runtime_materialization_declined",
                    request_id=load.request_id,
                    donor_id=load.donor_id,
                    namespace=load.namespace,
                    tokens=int(load.token_count),
                    materialization_kind=load.materialization_kind.value,
                    declined_reason="donor_gone",
                    layers_materialized=layers_materialized,
                )
            if load.materialization_kind == MaterializationKind.SEMANTIC_SPAN and (
                donor_gone or layers_materialized <= 0
            ):
                # The scheduler already skipped compute for these tokens;
                # continuing without KV would decode garbage silently. A
                # donor that vanished after some layers were written is the
                # same failure: those layers now disagree with the rest.
                self._audit_event(
                    "runtime_materialization_failed_loud",
                    request_id=load.request_id,
                    donor_id=load.donor_id,
                    reason="donor_gone" if donor_gone else "no_layers",
                )
                failure = (
                    f"donor {load.donor_id} is gone from storage after "
                    f"{layers_materialized} layer(s)"
                    if donor_gone
                    else "materialized 0 layers"
                )
                raise RuntimeError(
                    f"semantic-span load {failure} for {load.request_id}; "
                    "failing loudly instead of decoding over uninitialized KV"
                )
            if donor_gone:
                # The scheduler skipped compute for this window. Hand the
                # blocks back so the engine recomputes them instead of
                # decoding over whatever the pool held.
                failed = self._destination_block_ids(load)
                self._load_error_block_ids.update(failed)
                self._stats["load_error_blocks_reported"] += len(failed)
                self._audit_event(
                    "load_error_blocks_reported",
                    request_id=load.request_id,
                    donor_id=load.donor_id,
                    blocks=len(failed),
                )
                continue
            if layers_materialized <= 0:
                self._stats["loads_rejected_no_kv_layers"] += 1
                self._audit_event(
                    "runtime_materialization_declined",
                    request_id=load.request_id,
                    donor_id=load.donor_id,
                    namespace=load.namespace,
                    tokens=int(load.token_count),
                    materialization_kind=load.materialization_kind.value,
                    declined_reason="no_kv_layers_materialized",
                )
                continue
            self._stats["loads_materialized_total"] += 1
            self._audit_event(
                "runtime_materialized",
                request_id=load.request_id,
                donor_id=load.donor_id,
                namespace=load.namespace,
                tokens=int(load.token_count),
                materialization_kind=load.materialization_kind.value,
                layers_materialized=layers_materialized,
            )
            if self._config.log_decisions:
                logger.info(
                    "SemBlend materialized request_id=%s donor_id=%s tokens=%d",
                    load.request_id,
                    load.donor_id,
                    load.token_count,
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SemBlendConnectorMetadata) or not metadata.stores:
            return

        for store in metadata.stores:
            if store.block_ids is None:
                continue
            self._capture_layer(store, layer_name, kv_layer, attn_metadata)

    def _capture_layer(
        self, store: PendingStore, layer_name: str, kv_layer: Any, attn_metadata: Any
    ) -> None:
        """Copy one layer of the window this store adds, appended to earlier chunks.

        ``store.token_count`` is the donor's computed end, not a chunk length:
        the scheduler role does not know what the worker already holds, so the
        window starts at this layer's own progress. The block table covers the
        whole prefix from token 0, so when the earlier chunks are missing the
        capture simply restarts from 0 rather than storing a tail alone.
        """
        import torch

        progress = self._capture_progress.get(store.request_id, {})
        start = progress.get(layer_name, 0)
        if start >= store.token_count:
            # A resumed request re-admitted as new re-announces an end this
            # layer already reached.
            self._stats["layer_capture_skipped_nothing_new"] += 1
            return
        base = self._captured_layer(store, layer_name) if start > 0 else None
        if start > 0 and base is None:
            self._stats["layer_capture_base_missing"] += 1
            start = 0
        slot_mapping = self._slot_mapping(
            store.block_ids, store.token_count - start, kv_layer.device, target_start=start
        )
        kv_cache = self._extract_kv_from_layer(kv_layer, slot_mapping, attn_metadata)
        host_kv = kv_cache.detach().contiguous().cpu()
        if base is not None:
            # MLA captures are [tokens, C]; every other layout [2, tokens, H*D].
            host_kv = torch.cat((base, host_kv), dim=0 if host_kv.dim() == 2 else 1)
        actual_token_count = start + int(slot_mapping.numel())
        self._write_captured_layer(store, layer_name, host_kv, actual_token_count)
        self._capture_progress[store.request_id] = {**progress, layer_name: actual_token_count}

    def _captured_layer(self, store: PendingStore, layer_name: str) -> Any | None:
        """This donor's stored copy of one layer, or None when there is none."""
        if self._config.kv_storage_backend == "memory":
            entry = self._memory_store.get(self._storage_key(store.request_id, store.namespace))
            return None if entry is None else entry.get(layer_name)
        from safetensors.torch import load_file

        try:
            tensors = load_file(self._layer_filename(store.request_id, store.namespace, layer_name))
        except OSError:
            return None
        return tensors["kv_cache"]

    def _write_captured_layer(
        self, store: PendingStore, layer_name: str, host_kv: Any, token_count: int
    ) -> None:
        donor_dir = self._donor_dir(store.request_id, store.namespace)
        os.makedirs(donor_dir, exist_ok=True)
        # The scheduler-role connector runs in another process and reads
        # the captured length from this file for both backends.
        with open(self._donor_metadata_path(store.request_id, store.namespace), "w", encoding="utf-8") as f:
            json.dump({"token_count": token_count}, f)
        if self._config.kv_storage_backend == "memory":
            key = self._storage_key(store.request_id, store.namespace)
            entry = self._memory_store.get(key)
            if entry is None:
                entry = {"__token_count__": token_count, "__request_id__": store.request_id}
                self._memory_store[key] = entry
                while len(self._memory_store) > max(1, self._config.kv_memory_max_donors):
                    self._evict_memory_donor()
            else:
                entry["__token_count__"] = token_count
                self._memory_store.move_to_end(key)
            entry[layer_name] = host_kv
            return
        from safetensors.torch import save_file

        save_file(
            {"kv_cache": host_kv},
            self._layer_filename(store.request_id, store.namespace, layer_name),
        )

    def _evict_memory_donor(self) -> None:
        """Drop the least recently used donor and retract its advertised length.

        metadata.json is what the scheduler role sizes spans against, so left
        behind it advertises tensors that no longer exist and the worker then
        fails the load. The directory goes too: _has_stored_donor reads its
        presence as "captured".
        """
        storage_key, entry = self._memory_store.popitem(last=False)
        self._stats["memory_store_evictions"] += 1
        self._capture_progress.pop(str(entry.get("__request_id__", "")), None)
        donor_dir = self._donor_dir_for_key(storage_key)
        try:
            os.remove(os.path.join(donor_dir, _DONOR_METADATA_FILENAME))
        except FileNotFoundError:
            pass
        except OSError:
            self._stats["memory_store_eviction_retract_errors"] += 1
            logger.warning("SemBlend evicted-donor metadata removal failed", exc_info=True)
            return
        try:
            os.rmdir(donor_dir)
        except OSError:
            # Not empty or already gone; the length record is what mattered.
            pass

    def wait_for_save(self) -> None:
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # Worker side. A finished request sends no more prefill chunks, so
        # its capture progress can go; the captured donor itself stays.
        for request_id in finished_req_ids or ():
            self._capture_progress.pop(str(request_id), None)
        return None, None

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._lookup_cache.pop(self._request_id(request), None)
        self._load_plans.pop(self._request_id(request), None)
        self._seen_requests.discard(self._request_id(request))
        self._capture_state.pop(self._request_id(request), None)
        if self._compat_decline is not None:
            # A donor captured under a layout this connector cannot address is
            # a landmine for every later recipient that plans against it.
            self._stats["donor_registration_skipped_incompatible"] += 1
            return False, None
        if not self._config.register_donors or self._provider is None:
            return False, None

        token_ids = self._token_ids(request)
        if len(token_ids) < self._config.min_prompt_tokens:
            return False, None

        try:
            donor = DonorRegistration(
                donor_id=self._request_id(request),
                token_ids=token_ids,
                prompt_text=self._prompt_text(request),
                model_id=model_id_from_config(self._config, self._vllm_config),
                namespace=namespace_for_request(self._config, self._vllm_config, request),
                metadata={
                    "num_blocks": len(block_ids),
                    **self._routing_metadata(request),
                },
            )
            self._provider.register_donor(donor)
            self._stats["donors_registered_total"] += 1
            self._audit_event(
                "donor_registered",
                request_id=donor.donor_id,
                namespace=donor.namespace,
                tokens=len(token_ids),
                blocks=len(block_ids),
                metadata=dict(donor.metadata),
            )
            if self._config.log_decisions:
                logger.info(
                    "SemBlend donor registered request_id=%s namespace=%s tokens=%d blocks=%d",
                    donor.donor_id,
                    donor.namespace,
                    len(token_ids),
                    len(block_ids),
                )
        except Exception:
            logger.exception("SemBlend donor registration failed")
            self._stats["donor_registration_errors_total"] += 1
        return False, None

    def take_events(self):
        return ()

    def shutdown(self) -> None:
        return None

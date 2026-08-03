"""vLLM KVConnector implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from semblend_vllm_connector._vllm_compat import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.namespace import model_id_from_config, namespace_for_request
from semblend_vllm_connector.semantic_span import (
    block_align_spans,
    supply_at_boundary,
)
from semblend_vllm_connector.provider import SemanticKvProvider, load_provider
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
        self._stats: Counter[str] = Counter()
        self._block_size = int(getattr(getattr(vllm_config, "cache_config", None), "block_size", 16))
        self._prompt_tokenizer: Any | None = None
        self._prompt_tokenizer_failed = False

        self._provider = load_provider(self._config)

        if self._config.log_decisions:
            logger.info(
                "SemBlendVllmConnector initialized role=%s mode=%s provider=%s",
                role,
                self._config.mode.value,
                self._config.provider,
            )
        self._audit_event(
            "connector_initialized",
            role=str(role),
            mode=self._config.mode.value,
            provider=self._config.provider,
            block_size=self._block_size,
        )

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

    def _donor_dir(self, donor_id: str, namespace: str) -> str:
        return os.path.join(self._config.kv_storage_path, self._storage_key(donor_id, namespace))

    def _has_stored_donor(self, donor_id: str, namespace: str) -> bool:
        return os.path.isdir(self._donor_dir(donor_id, namespace))

    def _materialization_enabled(self) -> bool:
        return self._config.mode in {
            ReuseMode.EXACT_PREFIX,
            ReuseMode.REQUEST_ONLY_EXPERIMENTAL,
            ReuseMode.SEGMENTED_EXPERIMENTAL,
        }

    def _slot_mapping(self, block_ids: tuple[list[int], ...], token_count: int, device: Any):
        import torch

        group = block_ids[0]
        block_ids_tensor = torch.tensor(group, device=device)
        block_offsets = torch.arange(0, self._block_size, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_tensor.reshape((len(group), 1)) * self._block_size
        )
        return slot_mapping.flatten()[:token_count]

    def _is_mla_metadata(self, attn_metadata: Any) -> bool:
        try:
            from vllm.v1.attention.backends.mla.common import MLACommonMetadata
        except Exception:
            return False
        return isinstance(attn_metadata, MLACommonMetadata)

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

        num_pages = dst_shape[1]
        page_size = dst_shape[2]
        dst = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
        dst[:, slot_mapping, ...] = src_kv_cache
        dst.reshape(dst_shape)

    def _extract_kv_from_layer(self, kv_layer: Any, slot_mapping: Any, attn_metadata: Any) -> Any:
        layer_shape = kv_layer.shape
        if self._is_mla_metadata(attn_metadata):
            num_pages = layer_shape[0]
            page_size = layer_shape[1]
            return kv_layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]

        num_pages = layer_shape[1]
        page_size = layer_shape[2]
        return kv_layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

    def _layer_filename(self, donor_id: str, namespace: str, layer_name: str) -> str:
        return os.path.join(self._donor_dir(donor_id, namespace), f"{layer_name}.safetensors")

    def _donor_metadata_path(self, donor_id: str, namespace: str) -> str:
        return os.path.join(self._donor_dir(donor_id, namespace), "metadata.json")

    def _stored_donor_token_count(self, donor_id: str, namespace: str) -> int:
        try:
            with open(self._donor_metadata_path(donor_id, namespace), encoding="utf-8") as f:
                metadata = json.load(f)
        except OSError:
            return 0
        try:
            return int(metadata.get("token_count", 0))
        except (TypeError, ValueError):
            return 0

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        self._stats["lookups_total"] += 1
        token_ids = self._token_ids(request)
        request_id = self._request_id(request)

        if not token_ids or len(token_ids) < self._config.min_prompt_tokens:
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
            self._stats["skipped_exact_prefix_sufficient"] += 1
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

        started = time.monotonic()
        try:
            result = self._provider.lookup(lookup)
        except Exception:
            logger.exception("SemBlend provider lookup failed; falling back to normal prefill")
            self._stats["provider_errors_total"] += 1
            return 0, False
        finally:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self._stats["lookup_latency_ms_sum"] += elapsed_ms

        if result is None:
            self._stats["semantic_misses_total"] += 1
            self._audit_event(
                "semantic_lookup_miss",
                request_id=request_id,
                namespace=namespace,
                latency_ms=elapsed_ms,
            )
            if self._config.log_decisions:
                logger.info(
                    "SemBlend semantic lookup miss request_id=%s namespace=%s latency_ms=%d",
                    request_id,
                    namespace,
                    elapsed_ms,
                )
            return 0, False

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
        )
        if self._config.log_decisions:
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
            raw_spans = [
                {
                    "target_start": seg.target_start,
                    "length": seg.token_count,
                    "donor_start": seg.donor_start,
                }
                for seg in result.segments
                if seg.donor_id == result.donor_id
            ]
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
            if token_count > 0 and donor_start is not None:
                self._pending_loads[request_id] = PendingLoad(
                    request_id=request_id,
                    donor_id=result.donor_id,
                    token_count=token_count,
                    materialization_kind=MaterializationKind.SEMANTIC_SPAN,
                    namespace=namespace,
                    donor_start=donor_start,
                )
                self._stats["semantic_span_loads_advertised_total"] += 1
                self._audit_event(
                    "semantic_span_load_advertised",
                    request_id=request_id,
                    donor_id=result.donor_id,
                    namespace=namespace,
                    token_count=token_count,
                    donor_start=donor_start,
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
                self._pending_loads[request_id] = PendingLoad(
                    request_id=request_id,
                    donor_id=result.donor_id,
                    token_count=token_count,
                    materialization_kind=MaterializationKind.REQUEST_ONLY,
                    namespace=namespace,
                )
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
            self._stats["discovery_only_hits_total"] += 1
            return 0, False

        if (
            self._config.mode == ReuseMode.EXACT_PREFIX
            and result.materialization_kind == MaterializationKind.EXACT_PREFIX
            and result.block_refs
            and result.reusable_token_count > 0
        ):
            self._pending_loads[request_id] = PendingLoad(
                request_id=request_id,
                donor_id=result.donor_id,
                token_count=result.reusable_token_count,
                materialization_kind=result.materialization_kind,
                namespace=namespace,
            )
            self._stats["exact_prefix_loads_advertised_total"] += 1
            self._audit_event(
                "exact_prefix_load_advertised",
                request_id=request_id,
                donor_id=result.donor_id,
                namespace=namespace,
                tokens=int(result.reusable_token_count),
                materialization_kind=result.materialization_kind.value,
            )
            return result.reusable_token_count, False

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
        if num_external_tokens <= 0:
            return
        request_id = self._request_id(request)
        pending = self._pending_loads.get(request_id)
        if pending is None:
            self._stats["alloc_without_pending_load"] += 1
            return
        get_block_ids = getattr(blocks, "get_block_ids", None)
        block_ids = get_block_ids(allow_none=True) if callable(get_block_ids) else None
        self._pending_loads[request_id] = PendingLoad(
            request_id=pending.request_id,
            donor_id=pending.donor_id,
            token_count=pending.token_count,
            materialization_kind=pending.materialization_kind,
            namespace=pending.namespace,
            block_ids=block_ids,
            donor_start=pending.donor_start,
        )
        self._audit_event(
            "load_allocated",
            request_id=request_id,
            donor_id=pending.donor_id,
            namespace=pending.namespace,
            tokens=int(pending.token_count),
            materialization_kind=pending.materialization_kind.value,
            block_id_count=sum(len(group) for group in block_ids or ()),
        )

    def _build_store_metadata(self, scheduler_output: "SchedulerOutput") -> list[PendingStore]:
        if not self._materialization_enabled():
            return []

        stores: list[PendingStore] = []
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            token_ids = self._token_ids(new_req)
            if len(token_ids) < self._config.min_prompt_tokens:
                continue

            block_ids = _normalize_block_ids(getattr(new_req, "block_ids", None))
            if block_ids is None:
                continue

            token_count = _cacheable_prefix_tokens(len(token_ids), self._block_size)
            token_count = min(token_count, len(block_ids[0]) * self._block_size)
            if token_count <= 0:
                continue

            request_id = str(
                getattr(new_req, "req_id", None)
                or getattr(new_req, "request_id", None)
                or "unknown-request"
            )
            stores.append(
                PendingStore(
                    request_id=request_id,
                    token_ids=token_ids,
                    token_count=token_count,
                    namespace=namespace_for_request(self._config, self._vllm_config, new_req),
                    block_ids=block_ids,
                )
            )
        return stores

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> SemBlendConnectorMetadata:
        loads = list(self._pending_loads.values())
        stores = self._build_store_metadata(scheduler_output)
        self._pending_loads.clear()
        return SemBlendConnectorMetadata(loads=loads, stores=stores)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SemBlendConnectorMetadata) or not metadata.loads:
            return

        from safetensors.torch import load_file

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if attn_metadata is None:
            raise RuntimeError("SemBlend materialization requires forward_context.attn_metadata")

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
            for layer_name, layer in getattr(forward_context, "no_compile_layers", {}).items():
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue
                dst_kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                filename = self._layer_filename(load.donor_id, load.namespace, layer_name)
                tensors = load_file(filename)
                src_kv_cache = tensors["kv_cache"].to(dst_kv_cache_layer.device)
                if self._is_mla_metadata(attn_metadata):
                    src_kv_cache = src_kv_cache[: load.token_count, ...]
                else:
                    src_kv_cache = src_kv_cache[:, : load.token_count, ...]
                slot_mapping = self._slot_mapping(
                    load.block_ids,
                    load.token_count,
                    dst_kv_cache_layer.device,
                )
                self._inject_kv_into_layer(
                    dst_kv_cache_layer,
                    src_kv_cache,
                    slot_mapping,
                    attn_metadata,
                )
                layers_materialized += 1
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

        from safetensors.torch import save_file

        for store in metadata.stores:
            if store.block_ids is None:
                continue
            slot_mapping = self._slot_mapping(store.block_ids, store.token_count, kv_layer.device)
            actual_token_count = int(slot_mapping.numel())
            kv_cache = self._extract_kv_from_layer(kv_layer, slot_mapping, attn_metadata)
            donor_dir = self._donor_dir(store.request_id, store.namespace)
            os.makedirs(donor_dir, exist_ok=True)
            with open(self._donor_metadata_path(store.request_id, store.namespace), "w", encoding="utf-8") as f:
                json.dump({"token_count": actual_token_count}, f)
            save_file(
                {"kv_cache": kv_cache.detach().contiguous().cpu()},
                self._layer_filename(store.request_id, store.namespace, layer_name),
            )

    def wait_for_save(self) -> None:
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
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

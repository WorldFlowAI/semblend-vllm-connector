"""Shared connector types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from semblend_vllm_connector._vllm_compat import KVConnectorMetadata


class ReuseMode(str, Enum):
    DISCOVERY_ONLY = "discovery_only"
    EXACT_PREFIX = "exact_prefix"
    REQUEST_ONLY_EXPERIMENTAL = "request_only_experimental"
    SEGMENTED_EXPERIMENTAL = "segmented_experimental"
    # Semantic-span mode: block-aligned donor spans from token-verified
    # alignment, served boundary-anchored with RoPE re-rotation at load.
    SEMANTIC_SPAN_EXPERIMENTAL = "semantic_span_experimental"


class MaterializationKind(str, Enum):
    DISCOVERY_ONLY = "discovery_only"
    EXACT_PREFIX = "exact_prefix"
    REQUEST_ONLY = "request_only"
    SEMANTIC_SPAN = "semantic_span"
    SEGMENTED = "segmented"


@dataclass(frozen=True)
class AuditJoinKey:
    """The identity every audit event for one request carries.

    Phase-0 joins advertise -> allocate -> materialize -> finish by equality on
    ``request_id`` alone, so nothing here may be re-derived from state that
    moves between scheduling attempts. ``request_seq`` is stamped once, at the
    connector's first sight of the request, which makes it an arrival order
    rather than a scheduling order; ``event_seq`` orders the events within one
    request. Both are numbered per connector instance, which is what
    ``connector_id`` disambiguates: the scheduler and worker roles append to
    the same audit file and number the same request independently.
    """

    connector_id: str
    request_id: str
    request_seq: int
    event_seq: int

    def as_fields(self) -> Mapping[str, Any]:
        return {
            "connector_id": self.connector_id,
            "request_id": self.request_id,
            "request_seq": self.request_seq,
            "event_seq": self.event_seq,
        }


@dataclass(frozen=True)
class SemanticBlockRef:
    block_id: str
    start_token: int
    token_count: int


@dataclass(frozen=True)
class SemanticSegment:
    donor_id: str
    donor_start: int
    target_start: int
    token_count: int
    block_refs: Sequence[SemanticBlockRef] | None = None


@dataclass(frozen=True)
class SemanticLookupRequest:
    request_id: str
    token_ids: Sequence[int]
    prompt_text: str | None
    model_id: str
    namespace: str
    # The request's RAW cache salt, carried beside the derived namespace so a
    # provider never has to reach back into vLLM's objects for it. The
    # namespace is engine-local (it digests dtype and block size, which are not
    # request fields); the salt is what a caller holding the request can
    # reproduce. See docs/VLLM_CONNECTOR_CONTRACT.md, "Tenant key".
    cache_salt: str | None = None
    already_computed_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticLookupResult:
    donor_id: str
    similarity: float
    reusable_token_count: int
    materialization_kind: MaterializationKind = MaterializationKind.DISCOVERY_ONLY
    donor_token_ids: Sequence[int] | None = None
    block_refs: Sequence[SemanticBlockRef] | None = None
    segments: Sequence[SemanticSegment] | None = None
    quality_signals: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class DonorRegistration:
    donor_id: str
    token_ids: Sequence[int]
    prompt_text: str | None
    model_id: str
    namespace: str
    #: The registering request's RAW cache salt; see
    #: :class:`SemanticLookupRequest`.
    cache_salt: str | None = None
    #: Whether the registering engine holds captured KV for this donor. A
    #: donor registered with ``False`` can be discovered but can never supply
    #: KV, so a provider ranking candidates must drop it *before* the top-k
    #: cut: otherwise a family of such donors fills the cut and the one donor
    #: that can supply KV never reaches the verification stage behind it. The
    #: connector refuses to register a request whose capture was skipped, so
    #: the flag guards the registration paths it does not own (a donor
    #: announced by another engine, a provider that registers donors itself).
    has_captured_kv: bool = True
    timestamp: float = field(default_factory=time.monotonic)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadPiece:
    """One donor's contiguous contribution to a multi-donor span."""

    donor_id: str
    donor_start: int
    target_start: int
    token_count: int


@dataclass(frozen=True)
class PendingLoad:
    request_id: str
    donor_id: str
    token_count: int
    materialization_kind: MaterializationKind
    namespace: str
    # The request's COMPLETE per-group block list from token 0, exactly as the
    # scheduler supplies it, so absolute token position p is addressed by
    # block_ids[group][p // block_size]. A caller passing a list trimmed to the
    # served window would shift every destination back to the request's shared
    # prefix blocks and overwrite them.
    block_ids: tuple[list[int], ...] | None = None
    # Semantic-span loads: donor- and target-side positions of the first
    # served token, consumed by the re-rotation loader (delta = target -
    # donor).
    donor_start: int | None = None
    target_start: int | None = None
    # A span assembled from several donors, back to back in the target. The
    # scheduler expands it into one load per piece before the worker sees it.
    pieces: tuple[LoadPiece, ...] = ()


@dataclass(frozen=True)
class PendingStore:
    request_id: str
    token_ids: Sequence[int]
    token_count: int
    namespace: str
    block_ids: tuple[list[int], ...] | None = None
    #: Whether this store completes the donor -- the capture has reached the
    #: prompt's cacheable prefix and no further chunk will arrive. The worker
    #: publishes a donor's metadata record when its capture is complete, and
    #: that record is what makes the donor discoverable, so this is the flag
    #: that ends one donor's capture rather than a length the worker has to
    #: guess at.
    final: bool = False


@dataclass
class SemBlendConnectorMetadata(KVConnectorMetadata):
    loads: list[PendingLoad] = field(default_factory=list)
    stores: list[PendingStore] = field(default_factory=list)
    #: Requests whose capture will receive no further chunk although no store
    #: marked ``final`` closed it: decode started mid-capture, or the request
    #: finished. The worker publishes what it holds for them, so a donor
    #: shorter than its prompt is still discoverable.
    finalize: list[str] = field(default_factory=list)

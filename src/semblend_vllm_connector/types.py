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
    timestamp: float = field(default_factory=time.monotonic)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PendingLoad:
    request_id: str
    donor_id: str
    token_count: int
    materialization_kind: MaterializationKind
    namespace: str
    block_ids: tuple[list[int], ...] | None = None
    # Semantic-span loads: donor- and target-side positions of the first
    # served token, consumed by the re-rotation loader (delta = target -
    # donor).
    donor_start: int | None = None
    target_start: int | None = None


@dataclass(frozen=True)
class PendingStore:
    request_id: str
    token_ids: Sequence[int]
    token_count: int
    namespace: str
    block_ids: tuple[list[int], ...] | None = None


@dataclass
class SemBlendConnectorMetadata(KVConnectorMetadata):
    loads: list[PendingLoad] = field(default_factory=list)
    stores: list[PendingStore] = field(default_factory=list)

"""Provider protocol and deterministic local provider."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Iterable
from typing import Protocol

from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.types import (
    DonorRegistration,
    MaterializationKind,
    SemanticLookupRequest,
    SemanticLookupResult,
)


class SemanticKvProvider(Protocol):
    def lookup(self, request: SemanticLookupRequest) -> SemanticLookupResult | None: ...

    def register_donor(self, donor: DonorRegistration) -> None: ...

    def clear_donors(self) -> None: ...


def rank_candidates(
    scored: Iterable[tuple[float, DonorRegistration]], *, top_k: int
) -> list[tuple[float, DonorRegistration]]:
    """Order scored donors by similarity and cut to the top ``top_k``.

    Donors that hold no captured KV are dropped *before* the cut, not after.
    They are perfect matches for anything that looks like them -- a re-issue of
    a served prompt is identical to the last re-issue -- so once a few of them
    exist they take every slot in the cut, and the donor that actually holds KV
    never reaches whatever verification runs on the survivors. Dropping them
    first costs nothing: the cut exists to bound that verification, and a donor
    that can supply nothing has nothing to verify.
    """
    usable = [(similarity, donor) for similarity, donor in scored if donor.has_captured_kv]
    usable.sort(key=lambda item: item[0], reverse=True)
    return usable[:top_k] if top_k > 0 else usable


class LocalSemanticProvider:
    """Small deterministic provider for connector validation.

    This is not the production SemBlend provider. It intentionally returns
    discovery-only matches unless the token sequence is exactly equal.
    """

    def __init__(
        self,
        min_similarity: float = 0.70,
        max_donors: int = 10_000,
        lookup_top_k: int = 5,
    ) -> None:
        self._min_similarity = min_similarity
        self._max_donors = max_donors
        self._lookup_top_k = lookup_top_k
        self._donors: OrderedDict[str, DonorRegistration] = OrderedDict()

    def lookup(self, request: SemanticLookupRequest) -> SemanticLookupResult | None:
        candidates = rank_candidates(self._scored_donors(request), top_k=self._lookup_top_k)
        if not candidates:
            return None

        similarity, donor = candidates[0]
        exact = list(donor.token_ids) == list(request.token_ids)
        return SemanticLookupResult(
            donor_id=donor.donor_id,
            similarity=similarity,
            reusable_token_count=len(request.token_ids) if exact else 0,
            materialization_kind=(
                MaterializationKind.EXACT_PREFIX if exact else MaterializationKind.DISCOVERY_ONLY
            ),
            donor_token_ids=list(donor.token_ids),
            quality_signals={"token_jaccard": similarity, "exact_tokens": exact},
            reason="local_exact" if exact else "local_semantic_discovery",
        )

    def _scored_donors(
        self, request: SemanticLookupRequest
    ) -> list[tuple[float, DonorRegistration]]:
        """Every donor this request may see, with its token-Jaccard score."""
        query_set = set(request.token_ids)
        if not query_set:
            return []
        scored: list[tuple[float, DonorRegistration]] = []
        for donor in self._donors.values():
            if donor.namespace != request.namespace or donor.model_id != request.model_id:
                continue
            donor_set = set(donor.token_ids)
            if not donor_set:
                continue
            similarity = len(query_set & donor_set) / len(query_set | donor_set)
            if similarity >= self._min_similarity:
                scored.append((similarity, donor))
        return scored

    def register_donor(self, donor: DonorRegistration) -> None:
        if donor.donor_id in self._donors:
            self._donors.move_to_end(donor.donor_id)
            self._donors[donor.donor_id] = donor
            return
        while len(self._donors) >= self._max_donors:
            self._donors.popitem(last=False)
        self._donors[donor.donor_id] = donor

    def clear_donors(self) -> None:
        self._donors.clear()


def load_provider(config: SemBlendVllmConfig) -> SemanticKvProvider:
    if config.provider_module and config.provider_class:
        module = importlib.import_module(config.provider_module)
        cls = getattr(module, config.provider_class)
        return cls(config)

    if config.provider == "local":
        return LocalSemanticProvider(
            min_similarity=config.min_similarity,
            max_donors=config.max_donors,
            lookup_top_k=config.lookup_top_k,
        )

    if config.provider == "semblend":
        from semblend_vllm_connector.providers.semblend import SemBlendPipelineProvider

        return SemBlendPipelineProvider(config)

    raise ValueError(f"Unsupported SemBlend vLLM provider: {config.provider}")

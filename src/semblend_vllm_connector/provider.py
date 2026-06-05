"""Provider protocol and deterministic local provider."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from typing import Protocol

from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.types import (
    DonorRegistration,
    MaterializationKind,
    SemanticLookupRequest,
    SemanticLookupResult,
)


class SemanticKvProvider(Protocol):
    def lookup(self, request: SemanticLookupRequest) -> SemanticLookupResult | None:
        ...

    def register_donor(self, donor: DonorRegistration) -> None:
        ...

    def clear_donors(self) -> None:
        ...


class LocalSemanticProvider:
    """Small deterministic provider for connector validation.

    This is not the production SemBlend provider. It intentionally returns
    discovery-only matches unless the token sequence is exactly equal.
    """

    def __init__(self, min_similarity: float = 0.70, max_donors: int = 10_000) -> None:
        self._min_similarity = min_similarity
        self._max_donors = max_donors
        self._donors: OrderedDict[str, DonorRegistration] = OrderedDict()

    def lookup(self, request: SemanticLookupRequest) -> SemanticLookupResult | None:
        best: tuple[float, DonorRegistration] | None = None
        query_set = set(request.token_ids)
        if not query_set:
            return None

        for donor in self._donors.values():
            if donor.namespace != request.namespace or donor.model_id != request.model_id:
                continue
            donor_set = set(donor.token_ids)
            if not donor_set:
                continue
            similarity = len(query_set & donor_set) / len(query_set | donor_set)
            if similarity >= self._min_similarity and (
                best is None or similarity > best[0]
            ):
                best = (similarity, donor)

        if best is None:
            return None

        similarity, donor = best
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
        )

    if config.provider == "semblend":
        from semblend_vllm_connector.providers.semblend import SemBlendPipelineProvider

        return SemBlendPipelineProvider(config)

    raise ValueError(f"Unsupported SemBlend vLLM provider: {config.provider}")


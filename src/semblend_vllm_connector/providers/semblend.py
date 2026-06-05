"""Lazy SemBlendPipeline provider adapter."""

from __future__ import annotations

from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.types import (
    DonorRegistration,
    MaterializationKind,
    SemanticLookupRequest,
    SemanticLookupResult,
)


class SemBlendPipelineProvider:
    """Adapter from the connector provider protocol to SemBlendPipeline."""

    def __init__(self, config: SemBlendVllmConfig) -> None:
        try:
            from semblend import SemBlendPipeline
        except Exception as exc:  # pragma: no cover - depends on optional package.
            raise RuntimeError(
                "provider='semblend' requires the optional semblend package. "
                "Install with `pip install '.[semblend]'`."
            ) from exc

        self._config = config
        self._pipeline = self._create_pipeline(SemBlendPipeline, config)

    def _create_pipeline(self, pipeline_cls, config: SemBlendVllmConfig):
        try:
            from semblend_core.donor_store import DonorStore
            from semblend_core.embedder import create_embedder
        except Exception:
            return pipeline_cls(
                max_donors=config.max_donors,
                min_similarity=config.min_similarity,
                min_reuse_ratio=config.min_reuse_ratio,
                embedder_type=config.embedder_type,
                model_name=config.model_id,
                chunk_size=config.chunk_size,
            )

        embedder = create_embedder(config.embedder_type)
        donor_store = DonorStore(
            max_entries=config.max_donors,
            embedding_dim=embedder.dimension,
            min_similarity=config.min_similarity,
            chunk_size=config.chunk_size or 32,
        )
        return pipeline_cls(
            max_donors=config.max_donors,
            min_similarity=config.min_similarity,
            min_reuse_ratio=config.min_reuse_ratio,
            embedder_type=config.embedder_type,
            model_name=config.model_id,
            chunk_size=config.chunk_size,
            donor_store=donor_store,
        )

    def lookup(self, request: SemanticLookupRequest) -> SemanticLookupResult | None:
        result = self._pipeline.find_donor(
            token_ids=list(request.token_ids),
            prompt_text=request.prompt_text or "",
            top_k=self._config.lookup_top_k,
            extra_key=request.namespace,
        )
        if not result or not result.found or not result.donor_id:
            return None

        return SemanticLookupResult(
            donor_id=result.donor_id,
            similarity=float(result.similarity),
            reusable_token_count=0,
            materialization_kind=MaterializationKind.DISCOVERY_ONLY,
            donor_token_ids=list(result.donor_tokens or []),
            quality_signals={
                "reuse_ratio": float(getattr(result, "reuse_ratio", 0.0)),
                "confidence_tier": getattr(result, "confidence_tier", "unknown"),
                "fuzzy_confidence": float(getattr(result, "fuzzy_confidence", 0.0)),
                "rejection_reason": getattr(result, "rejection_reason", None),
            },
            metadata={
                "timings": getattr(result, "timings", None).__dict__
                if getattr(result, "timings", None)
                else {}
            },
            reason="semblend_discovery",
        )

    def register_donor(self, donor: DonorRegistration) -> None:
        self._pipeline.register_donor(
            request_id=donor.donor_id,
            token_ids=list(donor.token_ids),
            prompt_text=donor.prompt_text or "",
            extra_key=donor.namespace,
        )

    def clear_donors(self) -> None:
        self._pipeline.clear_donors()

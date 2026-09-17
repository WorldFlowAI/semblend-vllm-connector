"""Lazy SemBlendPipeline provider adapter."""

from __future__ import annotations

from inspect import signature

from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.types import (
    DonorRegistration,
    MaterializationKind,
    SemanticLookupRequest,
    SemanticLookupResult,
)


def _segments_from_position_map(result) -> list | None:
    """Group aligned (donor, target) position pairs into contiguous runs.

    Runs advance +1/+1 on both sides; each becomes a SemanticSegment with
    prompt-absolute target positions (the connector aligns and serves them
    boundary-anchored).
    """
    pmap = getattr(result, "position_map", None)
    if pmap is None:
        return None
    donors = list(getattr(pmap, "donor_positions", []) or [])
    targets = list(getattr(pmap, "target_positions", []) or [])
    if not donors or len(donors) != len(targets):
        return None

    from semblend_vllm_connector.types import SemanticSegment

    segments = []
    run_start = 0
    for i in range(1, len(donors) + 1):
        if i == len(donors) or donors[i] - donors[i - 1] != 1 or targets[i] - targets[i - 1] != 1:
            length = i - run_start
            if length >= 1:
                segments.append(
                    SemanticSegment(
                        donor_id=result.donor_id,
                        donor_start=donors[run_start],
                        target_start=targets[run_start],
                        token_count=length,
                    )
                )
            run_start = i
    return segments or None


MISS_DIAGNOSTIC_KEYS = (
    "rejection_reason",
    "donor_count",
    "donor_count_scope",
    "best_similarity",
    "reuse_ratio",
)


def _donor_count(pipeline, namespace: str | None) -> tuple[int | None, str | None]:
    """How many donors THIS lookup could see, and which population that is.

    The store filters by namespace before it searches, so the whole-store size
    answers a question nobody asked: a tenant whose namespace holds no donors
    misses every request while the store reports a thousand. Prefer the
    namespace-scoped count when the store offers one; otherwise say plainly
    that the number is the whole store.
    """
    store = getattr(pipeline, "_donor_store", None)
    if store is None:
        return None, None
    scoped = getattr(store, "visible_donors", None)
    if callable(scoped) and namespace is not None:
        try:
            return int(scoped(namespace)), "namespace"
        except Exception:
            pass
    size = getattr(store, "size", None)
    if isinstance(size, bool):
        return None, None
    if isinstance(size, int):
        return size, "store"
    return None, None


def _bare_reason(reason: object) -> str | None:
    """The rejection reason without anything a provider message might carry.

    The pipeline reports an internal failure as ``pipeline_error: <message>``,
    and a provider message can contain prompt text. Only the class of failure
    travels, matching what the connector does with exceptions on the sibling
    path.
    """
    if not isinstance(reason, str) or not reason:
        return None
    if reason.startswith("pipeline_error"):
        return "pipeline_error"
    return reason


def _miss_diagnostics(result, pipeline, namespace: str | None) -> dict:
    """The declining pipeline's own numbers, in a fixed shape for an audit row.

    Every key in MISS_DIAGNOSTIC_KEYS is present on every miss; a value the
    pipeline did not actually compute is None, never a dataclass default. On
    a ``no_donor_match`` the pipeline returns a fresh result carrying only the
    reason, so similarity and reuse read as their defaults there and MUST NOT
    be reported as measurements -- an earlier draft did exactly that and
    audited ``0.0`` as if the best candidate had scored zero.
    """
    count, scope = _donor_count(pipeline, namespace)
    diagnostics: dict = {
        "rejection_reason": _bare_reason(getattr(result, "rejection_reason", None))
        if result is not None
        else "no_result",
        "donor_count": count,
        "donor_count_scope": scope,
        "best_similarity": None,
        "reuse_ratio": None,
    }
    if result is None:
        return diagnostics
    # Only the decline paths that held a candidate know these. The pipeline
    # sets them on those results (semblend >= 0.3.24); an older pipeline
    # leaves the defaults, which are excluded by the > 0 test.
    similarity = getattr(result, "similarity", None)
    if isinstance(similarity, (int, float)) and not isinstance(similarity, bool) and similarity > 0:
        diagnostics["best_similarity"] = float(similarity)
    reuse = getattr(result, "reuse_ratio", None)
    if isinstance(reuse, (int, float)) and not isinstance(reuse, bool) and reuse > 0:
        diagnostics["reuse_ratio"] = float(reuse)
    return diagnostics


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
        self._last_miss: dict | None = None

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
            # The pipeline knows why it declined; the protocol can only say
            # None. Keep the reason so the caller can record it -- without it
            # a miss cannot be told apart from "nothing scored high enough".
            # Collected under a guard: diagnostics are never worth turning a
            # miss into a provider error.
            try:
                self._last_miss = _miss_diagnostics(result, self._pipeline, request.namespace)
            except Exception as exc:
                self._last_miss = {
                    **{key: None for key in MISS_DIAGNOSTIC_KEYS},
                    "rejection_reason": _bare_reason(getattr(result, "rejection_reason", None)),
                    "diagnostics_error": type(exc).__name__,
                }
            return None
        self._last_miss = None

        segments = _segments_from_position_map(result)
        return SemanticLookupResult(
            donor_id=result.donor_id,
            similarity=float(result.similarity),
            reusable_token_count=(sum(seg.token_count for seg in segments) if segments else 0),
            materialization_kind=MaterializationKind.DISCOVERY_ONLY,
            donor_token_ids=list(result.donor_tokens or []),
            segments=segments,
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

    def last_lookup_diagnostics(self) -> dict | None:
        """Why the most recent lookup missed, or None if it hit."""
        return self._last_miss

    def register_donor(self, donor: DonorRegistration) -> None:
        """Register a completed request as a donor.

        Two different keys travel with it. ``extra_key`` is the connector's
        per-request namespace, which isolates the donor inside this engine.
        ``cache_salt`` is the request's raw salt, from which SemBlend derives
        the tenant key it publishes: the namespace digests engine-private
        inputs and no router can reproduce it, so without the salt the donor
        is announced with no tenant identity anyone can match. Neither value
        is logged.

        A donor that holds no captured KV is not forwarded at all. Ranking
        inside the pipeline is the pipeline's -- this adapter passes a top_k
        and sees only the winner -- so an unusable donor put into that index
        can only take a slot in the cut away from a usable one.
        """
        if not donor.has_captured_kv:
            return
        kwargs = {
            "request_id": donor.donor_id,
            "token_ids": list(donor.token_ids),
            "prompt_text": donor.prompt_text or "",
            "extra_key": donor.namespace,
        }
        tenant = donor.metadata.get("tenant") or donor.metadata.get("tenant_id")
        template = donor.metadata.get("template") or donor.metadata.get("template_id")
        params = signature(self._pipeline.register_donor).parameters
        if tenant and "tenant" in params:
            kwargs["tenant"] = str(tenant)
        if template and "template" in params:
            kwargs["template"] = str(template)
        # Older SemBlend releases have no tenant-key argument; passing it
        # would raise rather than degrade, and the donor would be lost.
        if "cache_salt" in params:
            kwargs["cache_salt"] = donor.cache_salt
        self._pipeline.register_donor(**kwargs)

    def clear_donors(self) -> None:
        self._pipeline.clear_donors()

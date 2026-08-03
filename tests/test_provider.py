from __future__ import annotations

from semblend_vllm_connector.provider import LocalSemanticProvider
from semblend_vllm_connector.providers.semblend import SemBlendPipelineProvider
from semblend_vllm_connector.types import (
    DonorRegistration,
    MaterializationKind,
    SemanticLookupRequest,
)


def test_local_provider_returns_discovery_hit_for_overlap() -> None:
    provider = LocalSemanticProvider(min_similarity=0.5)
    provider.register_donor(
        DonorRegistration(
            donor_id="d1",
            token_ids=[1, 2, 3, 4],
            prompt_text=None,
            model_id="m",
            namespace="n",
        )
    )

    result = provider.lookup(
        SemanticLookupRequest(
            request_id="r1",
            token_ids=[1, 2, 3, 9],
            prompt_text=None,
            model_id="m",
            namespace="n",
        )
    )

    assert result is not None
    assert result.materialization_kind == MaterializationKind.DISCOVERY_ONLY
    assert result.reusable_token_count == 0


def test_local_provider_is_namespace_isolated() -> None:
    provider = LocalSemanticProvider(min_similarity=0.1)
    provider.register_donor(
        DonorRegistration(
            donor_id="d1",
            token_ids=[1, 2, 3, 4],
            prompt_text=None,
            model_id="m",
            namespace="n1",
        )
    )

    result = provider.lookup(
        SemanticLookupRequest(
            request_id="r1",
            token_ids=[1, 2, 3, 4],
            prompt_text=None,
            model_id="m",
            namespace="n2",
        )
    )

    assert result is None


def test_semblend_provider_forwards_routing_metadata_to_pipeline() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.calls = []

        def register_donor(
            self,
            *,
            request_id,
            token_ids,
            prompt_text,
            extra_key,
            tenant=None,
            template=None,
        ) -> None:
            self.calls.append(
                {
                    "request_id": request_id,
                    "token_ids": token_ids,
                    "prompt_text": prompt_text,
                    "extra_key": extra_key,
                    "tenant": tenant,
                    "template": template,
                }
            )

    pipeline = FakePipeline()
    provider = SemBlendPipelineProvider.__new__(SemBlendPipelineProvider)
    provider._pipeline = pipeline  # noqa: SLF001

    provider.register_donor(
        DonorRegistration(
            donor_id="d1",
            token_ids=[1, 2, 3, 4],
            prompt_text="policy text",
            model_id="m",
            namespace="n",
            metadata={"tenant": "wf-commercial", "template": "wf-rag-v1"},
        )
    )

    assert pipeline.calls == [
        {
            "request_id": "d1",
            "token_ids": [1, 2, 3, 4],
            "prompt_text": "policy text",
            "extra_key": "n",
            "tenant": "wf-commercial",
            "template": "wf-rag-v1",
        }
    ]


def test_semblend_provider_emits_segments_from_position_map() -> None:
    from types import SimpleNamespace

    from semblend_vllm_connector.types import SemanticLookupRequest

    class FakePipeline:
        def find_donor(self, **kwargs):
            return SimpleNamespace(
                found=True,
                donor_id="d1",
                similarity=0.98,
                donor_tokens=list(range(50)),
                reuse_ratio=0.9,
                confidence_tier="exact",
                fuzzy_confidence=0.98,
                rejection_reason=None,
                timings=None,
                # Two contiguous runs: target 10..14 <- donor 110..114 and
                # target 30..32 <- donor 200..202 (with a break between).
                position_map=SimpleNamespace(
                    donor_positions=[110, 111, 112, 113, 200, 201],
                    target_positions=[10, 11, 12, 13, 30, 31],
                ),
            )

    provider = SemBlendPipelineProvider.__new__(SemBlendPipelineProvider)
    provider._pipeline = FakePipeline()  # noqa: SLF001
    provider._config = SimpleNamespace(lookup_top_k=1)  # noqa: SLF001

    result = provider.lookup(
        SemanticLookupRequest(
            request_id="r1",
            token_ids=list(range(40)),
            prompt_text="p",
            model_id="m",
            namespace="n",
            already_computed_tokens=0,
        )
    )

    assert result is not None
    assert result.segments is not None and len(result.segments) == 2
    s0, s1 = result.segments
    assert (s0.target_start, s0.donor_start, s0.token_count) == (10, 110, 4)
    assert (s1.target_start, s1.donor_start, s1.token_count) == (30, 200, 2)
    assert result.reusable_token_count == 6


def test_semblend_provider_no_position_map_keeps_segments_none() -> None:
    from types import SimpleNamespace

    from semblend_vllm_connector.types import SemanticLookupRequest

    class FakePipeline:
        def find_donor(self, **kwargs):
            return SimpleNamespace(
                found=True,
                donor_id="d1",
                similarity=0.9,
                donor_tokens=[1, 2, 3],
                reuse_ratio=0.5,
                confidence_tier="fuzzy",
                fuzzy_confidence=0.6,
                rejection_reason=None,
                timings=None,
                position_map=None,
            )

    provider = SemBlendPipelineProvider.__new__(SemBlendPipelineProvider)
    provider._pipeline = FakePipeline()  # noqa: SLF001
    provider._config = SimpleNamespace(lookup_top_k=1)  # noqa: SLF001

    result = provider.lookup(
        SemanticLookupRequest(
            request_id="r1",
            token_ids=[1, 2, 3],
            prompt_text="p",
            model_id="m",
            namespace="n",
            already_computed_tokens=0,
        )
    )
    assert result is not None and result.segments is None

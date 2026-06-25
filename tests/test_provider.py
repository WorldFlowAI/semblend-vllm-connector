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

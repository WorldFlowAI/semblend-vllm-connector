"""SEMANTIC_SPAN mode: boundary-anchored block-aligned span advertisement.

The connector consumes token-verified alignment segments from the provider,
snaps them to block edges, and answers get_num_new_matched_tokens with the
contiguous run servable from the scheduler's computed boundary. Mid-span
boundaries advance the donor offset; boundaries in novel regions advertise
zero. The pending load carries the donor start for the re-rotation loader.
"""

from __future__ import annotations

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import (
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
)

from test_connector_discovery import (
    FakeBlocks,
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeSchedulerOutput,
    FakeVllmConfig,
)


def _connector(tmp_path, block_size=4, min_span=8):
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "semantic_span_experimental",
                "provider": "local",
                "min_prompt_tokens": 4,
                "min_similarity": 0.3,
                "kv_storage_path": str(tmp_path),
                "max_materialized_tokens": 4096,
                "min_semantic_span": min_span,
            }
        ),
        cache_config=FakeCacheConfig(block_size=block_size),
    )
    return SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)


def _spanned_result():
    # Donor span: target [10..90) from donor position 210 (delta 200).
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.99,
        reusable_token_count=80,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[
            SemanticSegment(
                donor_id="d1", donor_start=210, target_start=10, token_count=80
            )
        ],
    )


class _ScriptedProvider:
    def __init__(self, result):
        self.result = result
        self.lookups = []

    def lookup(self, req):
        self.lookups.append(req)
        return self.result

    def register(self, reg):
        return True


def test_first_schedule_advertises_zero_when_span_is_interior(tmp_path) -> None:
    connector = _connector(tmp_path)
    connector._provider = _ScriptedProvider(_spanned_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))

    matched, async_load = connector.get_num_new_matched_tokens(recipient, 0)

    # Boundary 0 sits in the novel head [0..10): nothing servable yet.
    assert (matched, async_load) == (0, False)


def test_boundary_inside_span_advertises_block_aligned_tail(tmp_path) -> None:
    connector = _connector(tmp_path)
    connector._provider = _ScriptedProvider(_spanned_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))

    # Simulate a continuation boundary at 12 (block-aligned, inside span
    # [12..88) after snapping 10->12, 90->88).
    matched, async_load = connector.get_num_new_matched_tokens(recipient, 12)

    assert (matched, async_load) == (76, False)

    connector.update_state_after_alloc(recipient, FakeBlocks(([3, 4, 5],)), matched)
    metadata = connector.build_connector_meta(FakeSchedulerOutput())
    assert len(metadata.loads) == 1
    load = metadata.loads[0]
    assert load.materialization_kind == MaterializationKind.SEMANTIC_SPAN
    assert load.token_count == 76
    # Donor start advanced by the snap (10->12): 210 + 2.
    assert load.donor_start == 212


def test_mid_span_boundary_advances_donor_offset(tmp_path) -> None:
    connector = _connector(tmp_path)
    connector._provider = _ScriptedProvider(_spanned_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))

    matched, _ = connector.get_num_new_matched_tokens(recipient, 40)

    assert matched == 48  # [40..88)
    load = connector._pending_loads["r1"]  # noqa: SLF001
    assert load.donor_start == 240  # 210 + (40 - 10)

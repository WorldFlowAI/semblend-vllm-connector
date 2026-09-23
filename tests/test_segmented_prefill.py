"""Connector side of segmented prefill: later segments from the admission lookup.

A scheduler that supports segmented prefill admits the request with the first
span, computes up to where the next reusable run begins
(get_prefill_compute_limit), then asks for that run
(get_num_new_matched_tokens_mid_prefill) and loads it -- one standard causal
chunk per gap, no second lookup. Stock vLLM never calls either hook.
"""

from __future__ import annotations

from test_b6_contamination import FakeBlockPool, _connector
from test_connector_discovery import FakeBlocks, FakeRequest

from semblend_vllm_connector.types import (
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
)

# Block size 4, span floor 8. Target: run A [0, 60), edit [60, 63), run B
# [63, 160) shifted by +3 against the donor, edit, run C [170, 240).
SEGMENTS = [
    SemanticSegment(donor_id="d1", donor_start=0, target_start=0, token_count=60),
    SemanticSegment(donor_id="d1", donor_start=60, target_start=63, token_count=97),
    SemanticSegment(donor_id="d1", donor_start=160, target_start=170, token_count=70),
]


class _Provider:
    def __init__(self):
        self.lookups = 0

    def lookup(self, request):
        self.lookups += 1
        return SemanticLookupResult(
            donor_id="d1",
            similarity=0.9,
            reusable_token_count=227,
            materialization_kind=MaterializationKind.SEMANTIC_SPAN,
            segments=SEGMENTS,
        )

    def register_donor(self, donor):
        return None

    def clear_donors(self):
        return None


def _setup(tmp_path):
    connector = _connector(tmp_path, min_mid_prefill_segment=16)
    provider = _Provider()
    connector._provider = provider  # noqa: SLF001
    connector._stored_donor_token_count = lambda donor_id, namespace: 4096  # noqa: SLF001
    connector.bind_gpu_block_pool(FakeBlockPool())
    request = FakeRequest("r1", list(range(260)))
    return connector, provider, request


def test_a_request_walks_its_segments_without_a_second_lookup(tmp_path):
    connector, provider, request = _setup(tmp_path)
    table = FakeBlocks((list(range(100, 170)),))

    # Admission at boundary 4: run A served to 60 (snapped to 60).
    first, _ = connector.get_num_new_matched_tokens(request, 4)
    assert first == 56
    connector.update_state_after_alloc(request, table, first)
    # Compute from 60 only to the first block edge inside run B.
    assert connector.get_prefill_compute_limit(request, 60) == 4

    # Next step at 64: run B from 64 to 160.
    second = connector.get_num_new_matched_tokens_mid_prefill(request, 64)
    assert second == 96
    plan = connector._load_plans["r1"]  # noqa: SLF001
    (piece,) = plan["pieces"]
    # Donor side shifted by the edit: target 64 is donor 61.
    assert (piece.donor_start, piece.target_start, piece.token_count) == (61, 64, 96)
    connector.update_state_after_alloc(request, table, second)
    assert connector.get_prefill_compute_limit(request, 160) == 12  # to 172

    assert connector.get_num_new_matched_tokens_mid_prefill(request, 172) == 68
    assert connector.get_prefill_compute_limit(request, 240) is None
    assert provider.lookups == 1
    assert connector.stats_snapshot["segmented_prefill_segments_total"] == 2


def test_an_unaligned_position_or_a_short_run_loads_nothing(tmp_path):
    connector, _, request = _setup(tmp_path)
    connector.get_num_new_matched_tokens(request, 4)
    assert connector.get_num_new_matched_tokens_mid_prefill(request, 66) == 0
    assert connector.get_num_new_matched_tokens_mid_prefill(request, 232) == 0  # 8 < 16


def test_a_request_the_connector_never_looked_up_is_untouched(tmp_path):
    connector, _, request = _setup(tmp_path)
    assert connector.get_prefill_compute_limit(request, 0) is None
    assert connector.get_num_new_matched_tokens_mid_prefill(request, 64) == 0

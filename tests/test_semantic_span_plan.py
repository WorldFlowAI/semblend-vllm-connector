"""Block-aligned semantic-span planning for the SEMANTIC_SPAN reuse mode.

vLLM allocates and loads external KV at block granularity, so donor spans
from token-level alignment must be snapped inward to block boundaries in
the TARGET frame, with the donor-side start advanced by the same amount so
token identity is preserved. Spans that snap below the minimum length fold
away. The supply decision is boundary-anchored: given the scheduler's
computed-token boundary, the planner reports how many tokens it can serve
contiguously from that boundary (zero when the boundary sits in a novel
region), which is exactly the get_num_new_matched_tokens contract.
"""

from semblend_vllm_connector.semantic_span import (
    BlockSpan,
    block_align_spans,
    supply_at_boundary,
)


def _span(t_start, length, d_start):
    return {"target_start": t_start, "length": length, "donor_start": d_start}


class TestBlockAlignSpans:
    def test_snaps_inward_and_advances_donor(self):
        spans = block_align_spans([_span(100, 1000, 700)], block_size=16, min_span=64)
        assert len(spans) == 1
        s = spans[0]
        # 100 -> 112 (next block edge), end 1100 -> 1088 (previous edge)
        assert (s.target_start, s.target_end) == (112, 1088)
        assert s.donor_start == 712  # advanced by the same 12 tokens

    def test_already_aligned_untouched(self):
        spans = block_align_spans([_span(128, 512, 700)], block_size=16, min_span=64)
        s = spans[0]
        assert (s.target_start, s.target_end, s.donor_start) == (128, 640, 700)

    def test_below_min_after_snap_folds(self):
        assert block_align_spans([_span(10, 60, 0)], block_size=16, min_span=64) == []

    def test_overlapping_later_span_dropped(self):
        spans = block_align_spans(
            [_span(100, 200, 0), _span(200, 200, 500)], block_size=16, min_span=64
        )
        assert len(spans) == 1
        assert spans[0].target_start == 112


class TestSupplyAtBoundary:
    SPANS = [
        BlockSpan(target_start=112, target_end=1088, donor_start=712),
        BlockSpan(target_start=2048, target_end=4096, donor_start=5000),
    ]

    def test_boundary_at_span_start_supplies_whole_span(self):
        n, donor_start = supply_at_boundary(self.SPANS, boundary=112, block_size=16)
        assert n == 976 and donor_start == 712

    def test_boundary_mid_span_supplies_block_aligned_tail(self):
        # Boundary 240 is inside span 1; tail = 1088-240 = 848 (block multiple),
        # donor advanced by the 128 tokens already covered.
        n, donor_start = supply_at_boundary(self.SPANS, boundary=240, block_size=16)
        assert n == 848 and donor_start == 840

    def test_boundary_in_novel_region_supplies_zero(self):
        n, donor_start = supply_at_boundary(self.SPANS, boundary=1088, block_size=16)
        assert n == 0 and donor_start is None

    def test_unaligned_boundary_snaps_supply_down(self):
        # Boundary 250 (not a block edge): usable tail starts at next block
        # edge 256; supply = 1088-256 = 832, donor advanced 144.
        n, donor_start = supply_at_boundary(self.SPANS, boundary=250, block_size=16)
        assert n == 832 and donor_start == 856

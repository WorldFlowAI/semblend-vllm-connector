"""One contiguous span assembled from several donors.

The stock connector interface serves one contiguous run per request from the
engine's cached prefix, but nothing requires that run to come from a single
donor. A prompt that joins passages first seen in different requests -- two
retrieved documents, say -- is served as one span whose pieces are read from
their own donors and re-rotated by their own deltas. Only the span's outer
ends are block-aligned; the pieces meet at whatever token the passages do.
"""

from __future__ import annotations

import types

from test_b6_contamination import FakeBlockPool, _connector
from test_connector_discovery import FakeRequest

from semblend_vllm_connector.providers.semblend import _segments_from_multi_donor_map
from semblend_vllm_connector.semantic_span import chain_at_boundary, trim_pieces
from semblend_vllm_connector.types import (
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
    SemBlendConnectorMetadata,
)

BLOCK = 4


def _run(donor, target_start, length, donor_start):
    return {
        "donor_id": donor,
        "target_start": target_start,
        "length": length,
        "donor_start": donor_start,
    }


class TestChain:
    def test_two_donors_meet_mid_block(self):
        runs = [_run("a", 0, 50, 100), _run("b", 50, 40, 7)]
        pieces = chain_at_boundary(runs, 12, BLOCK)
        assert pieces == [
            {"donor_id": "a", "donor_start": 112, "target_start": 12, "token_count": 38},
            {"donor_id": "b", "donor_start": 7, "target_start": 50, "token_count": 38},
        ]
        # Outer end snapped down to a block edge: 90 -> 88.
        assert pieces[-1]["target_start"] + pieces[-1]["token_count"] == 88

    def test_the_furthest_reaching_run_wins_at_each_step(self):
        runs = [_run("a", 0, 30, 0), _run("b", 8, 60, 0), _run("c", 30, 10, 0)]
        pieces = chain_at_boundary(runs, 8, BLOCK)
        assert [p["donor_id"] for p in pieces] == ["b"]
        assert pieces[0]["token_count"] == 60

    def test_a_gap_ends_the_chain(self):
        runs = [_run("a", 0, 20, 0), _run("b", 21, 40, 0)]
        pieces = chain_at_boundary(runs, 0, BLOCK)
        assert [p["donor_id"] for p in pieces] == ["a"]
        assert pieces[0]["token_count"] == 20

    def test_nothing_covers_the_boundary(self):
        assert chain_at_boundary([_run("a", 20, 40, 0)], 4, BLOCK) == []

    def test_trim(self):
        pieces = [
            {"donor_id": "a", "donor_start": 0, "target_start": 0, "token_count": 10},
            {"donor_id": "b", "donor_start": 0, "target_start": 10, "token_count": 10},
        ]
        assert trim_pieces(pieces, 8) == [
            {"donor_id": "a", "donor_start": 0, "target_start": 0, "token_count": 8}
        ]


def _two_donor_result():
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.9,
        reusable_token_count=160,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[
            SemanticSegment(donor_id="d1", donor_start=300, target_start=12, token_count=70),
            SemanticSegment(donor_id="d2", donor_start=5, target_start=82, token_count=90),
        ],
    )


class _Provider:
    def lookup(self, request):
        return _two_donor_result()

    def register_donor(self, donor):
        return None

    def clear_donors(self):
        return None


def _serving_connector(tmp_path, **overrides):
    connector = _connector(tmp_path, **overrides)
    connector._provider = _Provider()  # noqa: SLF001
    connector._stored_donor_token_count = lambda donor_id, namespace: 4096  # noqa: SLF001
    connector.bind_gpu_block_pool(FakeBlockPool())
    return connector


def test_the_connector_advertises_the_assembled_span(tmp_path):
    connector = _serving_connector(tmp_path)
    request = FakeRequest("r1", list(range(200)))
    matched, _ = connector.get_num_new_matched_tokens(request, 12)
    # 12 -> 172, snapped down to 172 (a block edge): 160 tokens, two donors.
    assert matched == 160
    plan = connector._load_plans["r1"]  # noqa: SLF001
    assert [(p.donor_id, p.target_start, p.token_count) for p in plan["pieces"]] == [
        ("d1", 12, 70),
        ("d2", 82, 90),
    ]
    assert connector.stats_snapshot["semantic_span_multi_donor_loads_total"] == 1


def test_each_piece_becomes_its_own_load(tmp_path):
    from test_b6_contamination import _admission_step
    from test_connector_discovery import FakeBlocks

    connector = _serving_connector(tmp_path)
    request = FakeRequest("r1", list(range(200)))
    matched, _ = connector.get_num_new_matched_tokens(request, 12)
    table = (list(range(100, 150)),)
    connector.update_state_after_alloc(request, FakeBlocks(table), matched)
    meta = connector.build_connector_meta(_admission_step("r1", request.all_token_ids, table, 12))
    assert isinstance(meta, SemBlendConnectorMetadata)
    assert [
        (load.donor_id, load.donor_start, load.target_start, load.token_count)
        for load in meta.loads
    ] == [
        ("d1", 300, 12, 70),
        ("d2", 5, 82, 90),
    ]
    assert all(load.pieces == () for load in meta.loads)


def test_a_short_chain_is_declined(tmp_path):
    connector = _serving_connector(tmp_path, min_semantic_span=256)
    # Long enough that the remaining-span gate lets the lookup run.
    matched, _ = connector.get_num_new_matched_tokens(FakeRequest("r1", list(range(400))), 12)
    assert matched == 0
    assert connector.stats_snapshot["semantic_span_multi_donor_declined_per_attempt"] == 1


def test_the_switch_restores_single_donor_behaviour(tmp_path):
    connector = _serving_connector(tmp_path, multi_donor_spans=False)
    matched, _ = connector.get_num_new_matched_tokens(FakeRequest("r1", list(range(200))), 12)
    # Only d1's run is used: 12..80 snapped to 12..80 = 68 tokens (>= 8 floor).
    assert matched == 68


def test_the_provider_keeps_only_identical_tokens():
    target = list(range(1000, 1020))
    donors = {"a": list(range(1000, 1010)), "b": [0] * 3 + list(range(1010, 1020))}
    mapping = types.SimpleNamespace(
        donor_ids=("a",) * 10 + ("b",) * 10,
        donor_positions=tuple(range(10)) + tuple(range(3, 13)),
        target_positions=tuple(range(20)),
    )
    # Corrupt one of b's tokens: that position must split b's run.
    donors["b"][7] = -1
    store = types.SimpleNamespace(get_donor_tokens=lambda d: donors[d])
    pipeline = types.SimpleNamespace(_donor_store=store)
    result = types.SimpleNamespace(multi_donor_position_map=mapping)
    segments = _segments_from_multi_donor_map(result, pipeline, target)
    assert [(s.donor_id, s.donor_start, s.target_start, s.token_count) for s in segments] == [
        ("a", 0, 0, 10),
        ("b", 3, 10, 4),
        ("b", 8, 15, 5),
    ]

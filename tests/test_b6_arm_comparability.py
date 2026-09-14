"""B6: the A/B arms have to count the same thing.

``evict_filled_blocks_from_prefix_cache`` exists so the eviction's effect can
be measured: the normal arm evicts every block the connector fills, the control
arm leaves them cached, and the difference between the two runs is what the
mitigation costs or saves. That comparison is only meaningful if both arms
report their contaminated blocks in the same unit.

They do not report in the same unit by accident. A pass over a request's block
table acts on the blocks that are cached *and not yet settled*, and both arms
settle what they acted on, so while the request keeps one block table the two
arms' per-pass counts are identical. They part when that table is replaced:
readmission clears what was settled, and then

* with eviction off the blocks were never removed, so the scan re-finds all of
  them;
* with eviction on it re-finds only the ones vLLM re-hashed on the way back in
  -- which is all of them when the request is handed its own just-freed blocks
  back, and none of them when it lands somewhere else.

Per-pass totals therefore differ between the arms. The request-scoped distinct
totals do not, and these tests pin that -- over the readmission case that makes
the difference visible: a preempted request handed its own just-freed blocks
back, which is the common case, because memory pressure is the reason it was
preempted at all.
"""

from __future__ import annotations

from test_b6_contamination import (
    PROMPT_TOKENS,
    FakeRequest,
    _admission_step,
    _audit_events,
    _bound_connector,
    _cached_step,
    _SchedulerOutput,
    _serve,
    _write_donor_capture,
)
from test_connector_discovery import FakeBlocks

# The fill starts at block index 3 (boundary 12, block size 4), so 103-199 of
# the first table are connector-filled and 200 is the block decode appends.
FILLED = set(range(103, 200)) | {200}
SAME_TABLE = (list(range(100, 200)),)


def _run_same_table_readmit(connector, pool) -> None:
    """Serve, decode a block, then be readmitted onto the very same blocks.

    A preempted request's blocks go back on the free list, so the readmitted
    request is routinely handed its own blocks straight back. vLLM re-hashes
    them under the recipient's token ids on the way in, exactly as it did on
    the first admission, so the connector sees its own filled blocks cached
    again -- ids it has already acted on once.
    """
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)
    pool.cache_blocks(SAME_TABLE[0])
    _serve(connector, request, 12, SAME_TABLE)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, SAME_TABLE, 12))
    pool.cache_blocks([200])
    connector.build_connector_meta(_cached_step("r1", ([200],), 400, num_output_tokens=1))

    # Preempted and readmitted at the same boundary onto the same table.
    pool.cache_blocks(SAME_TABLE[0])
    _serve(connector, request, 12, SAME_TABLE)
    connector.build_connector_meta(_cached_step("r1", SAME_TABLE, 12, resumed=True))


def _events(root, name: str) -> list[dict]:
    return [event for event in _audit_events(root) if event["event"] == name]


def test_both_arms_report_the_same_distinct_block_total(tmp_path) -> None:
    """The headline A/B number is the same unit on both arms.

    Identical contamination -- the same request, the same span, the same
    physical blocks -- has to produce the same count of contaminated blocks
    whichever arm measured it, or the two arms cannot be subtracted.
    """
    on, on_pool = _bound_connector(tmp_path / "on")
    off, off_pool = _bound_connector(tmp_path / "off", evict_filled_blocks_from_prefix_cache=False)

    _run_same_table_readmit(on, on_pool)
    _run_same_table_readmit(off, off_pool)

    on_stats = on.stats_snapshot
    off_stats = off.stats_snapshot
    assert on_stats["prefix_cache_distinct_blocks_evicted"] == len(FILLED)
    assert off_stats["prefix_cache_distinct_blocks_left_cached"] == len(FILLED)
    assert (
        on_stats["prefix_cache_distinct_blocks_evicted"]
        == off_stats["prefix_cache_distinct_blocks_left_cached"]
    )
    # Guard against a vacuous comparison: the arms must still differ in the
    # behaviour the knob controls.
    assert off_pool.calls == []
    assert off_pool.hashed >= FILLED
    assert not on_pool.hashed & FILLED


def test_per_pass_totals_differ_between_the_arms_by_construction(tmp_path) -> None:
    """And the counters say which unit they are, so the difference is legible.

    The first two passes match: both arms act on 97 blocks and then on the one
    decode block. The readmission's allocate pass matches too -- the blocks were
    re-hashed on the way in, so both arms find all 97. The running step behind
    it is where they part: it replaces the table, so both arms scan again, and
    the eviction arm has just emptied those entries while the control arm has
    not. Summing per-pass counts across arms would compare eviction work against
    residency, which is why the distinct counters exist.
    """
    on, on_pool = _bound_connector(tmp_path / "on")
    off, off_pool = _bound_connector(tmp_path / "off", evict_filled_blocks_from_prefix_cache=False)

    _run_same_table_readmit(on, on_pool)
    _run_same_table_readmit(off, off_pool)

    assert [len(call) for call in on_pool.calls] == [97, 1, 97]
    assert on.stats_snapshot["prefix_cache_blocks_evicted"] == 195
    assert off.stats_snapshot["prefix_cache_blocks_left_cached"] == 292
    # Same blocks either way, which is the point of the distinct pair.
    assert on.stats_snapshot["prefix_cache_distinct_blocks_evicted"] == 98
    assert off.stats_snapshot["prefix_cache_distinct_blocks_left_cached"] == 98


def test_left_cached_events_carry_both_units(tmp_path) -> None:
    """Every pass that finds filled blocks cached writes one event.

    A pass that re-finds only blocks it has already reported still says so --
    silence there would read as "the request stopped being contaminated" -- and
    it contributes nothing to the distinct total.
    """
    root = tmp_path / "off"
    connector, pool = _bound_connector(root, evict_filled_blocks_from_prefix_cache=False)

    _run_same_table_readmit(connector, pool)

    events = _events(root, "prefix_cache_blocks_left_cached")
    assert [event["phase"] for event in events] == [
        "load_allocated",
        "running_step",
        "load_allocated",
        "running_step",
    ]
    assert [event["blocks_left_cached"] for event in events] == [97, 1, 97, 97]
    assert [event["distinct_blocks_left_cached"] for event in events] == [97, 1, 0, 0]
    assert [event["distinct_blocks_left_cached_cumulative"] for event in events] == [
        97,
        98,
        98,
        98,
    ]
    # The ids are the pass's own view, so the union over a run is the
    # contaminated set and no arithmetic on the ids is needed to get it.
    ids = {block_id for event in events for block_id in event["block_ids_left_cached"]}
    assert ids == FILLED


def test_eviction_events_carry_both_units(tmp_path) -> None:
    """The eviction arm reports the same two units, under the same field names.

    ``blocks_evicted_cumulative`` is the running total for the current
    admission and restarts when the request is readmitted, because the blocks
    genuinely have to be evicted again. The distinct total does not restart, so
    it is the one that joins to the control arm.
    """
    connector, pool = _bound_connector(tmp_path)

    _run_same_table_readmit(connector, pool)

    events = _events(tmp_path, "prefix_cache_blocks_evicted")
    assert [event["phase"] for event in events] == [
        "load_allocated",
        "running_step",
        "load_allocated",
    ]
    assert [event["blocks_evicted"] for event in events] == [97, 1, 97]
    assert [event["blocks_evicted_cumulative"] for event in events] == [97, 98, 97]
    assert [event["distinct_blocks_evicted"] for event in events] == [97, 1, 0]
    assert [event["distinct_blocks_evicted_cumulative"] for event in events] == [
        97,
        98,
        98,
    ]


def _serve_drop_serve(connector, pool) -> None:
    """Serve, be readmitted without the span, then be readmitted and served.

    ``update_state_after_alloc`` with no external tokens is the documented
    partial-tail loss: the connector advertised a span and the scheduler
    admitted the request without it (``load_dropped_before_alloc``). The
    request goes on running and is served on a later admission, onto the very
    same physical blocks -- so the blocks it is counted for the second time are
    the ones it was already counted for.
    """
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)
    pool.cache_blocks(SAME_TABLE[0])
    _serve(connector, request, 12, SAME_TABLE)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, SAME_TABLE, 12))

    # Readmitted, advertised, then not credited with the span.
    matched, _ = connector.get_num_new_matched_tokens(request, 12)
    assert matched > 0, "fixture must advertise before the scheduler drops it"
    connector.update_state_after_alloc(request, FakeBlocks(SAME_TABLE), 0)

    # Readmitted once more onto the same blocks, this time served.
    pool.cache_blocks(SAME_TABLE[0])
    _serve(connector, request, 12, SAME_TABLE)
    connector.build_connector_meta(_cached_step("r1", SAME_TABLE, 12, resumed=True))


def test_a_dropped_readmission_does_not_restart_the_distinct_total(tmp_path) -> None:
    """The distinct total is per request, including readmissions nothing served.

    A readmission the scheduler drops before the span is credited still reaches
    ``update_state_after_alloc``, and the block-table state tracked for the
    previous admission has to go there -- it describes blocks this request may
    no longer hold. What must not go with it is the record of which physical
    blocks the request has already been counted for: the contract offers the
    per-request maximum of ``distinct_*_cumulative`` as the contamination
    number, and restarting it would report those 97 blocks twice, once per
    dropped readmission, on whichever arm measured it.
    """
    on, on_pool = _bound_connector(tmp_path / "on")
    off, off_pool = _bound_connector(tmp_path / "off", evict_filled_blocks_from_prefix_cache=False)

    _serve_drop_serve(on, on_pool)
    _serve_drop_serve(off, off_pool)

    distinct = len(range(103, 200))
    on_stats = on.stats_snapshot
    off_stats = off.stats_snapshot
    assert on_stats["load_dropped_before_alloc"] == 1
    assert off_stats["load_dropped_before_alloc"] == 1
    assert on_stats["prefix_cache_distinct_blocks_evicted"] == distinct
    assert off_stats["prefix_cache_distinct_blocks_left_cached"] == distinct

    evicted = _events(tmp_path / "on", "prefix_cache_blocks_evicted")
    left = _events(tmp_path / "off", "prefix_cache_blocks_left_cached")
    assert max(e["distinct_blocks_evicted_cumulative"] for e in evicted) == distinct
    assert max(e["distinct_blocks_left_cached_cumulative"] for e in left) == distinct
    # Guard against a vacuous pass: the second admission really did act on the
    # same 97 blocks again, it just did not count them again.
    assert [e["blocks_evicted"] for e in evicted] == [97, 97]
    assert on_pool.calls == [tuple(range(103, 200))] * 2


def test_the_distinct_ledger_is_retired_with_the_request(tmp_path) -> None:
    """Per request also means gone when the request is, on both exits.

    The ledger outlives the block-table state on purpose, so the two exits a
    request can leave by -- this step's finished ids, and ``request_finished``
    -- are the only things that drop it. If neither did, a long-running engine
    would keep a frozenset per request it has ever served.
    """
    finished_ids, finished_pool = _bound_connector(tmp_path / "ids")
    finished_call, call_pool = _bound_connector(tmp_path / "call")

    for connector, pool in ((finished_ids, finished_pool), (finished_call, call_pool)):
        request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
        _write_donor_capture(connector, request)
        pool.cache_blocks(SAME_TABLE[0])
        _serve(connector, request, 12, SAME_TABLE)
        connector.build_connector_meta(_admission_step("r1", request.all_token_ids, SAME_TABLE, 12))
        assert connector._reported_fill_blocks  # noqa: SLF001

    finished_ids.build_connector_meta(_SchedulerOutput(finished_req_ids={"r1"}))
    finished_call.request_finished(FakeRequest("r1", list(range(PROMPT_TOKENS))), [])

    assert finished_ids._reported_fill_blocks == {}  # noqa: SLF001
    assert finished_call._reported_fill_blocks == {}  # noqa: SLF001

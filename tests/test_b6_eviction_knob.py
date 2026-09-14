"""B6: the ``evict_filled_blocks_from_prefix_cache`` knob.

The eviction in tests/test_b6_contamination.py is the shipped behaviour. This
module covers the one supported way to switch it off: a contaminated control
arm, where the connector still tracks the blocks it filled and audits what it
would have evicted, so a measurement can say what the eviction removes instead
of asserting it.

What the knob must not change is everything else -- the same span served, the
same load metadata handed to the worker, the same counters apart from the ones
that name the mitigation. Which counter to compare between the arms is pinned
in tests/test_b6_arm_comparability.py.

Scheduler-side only: nothing here touches KV tensors, so no torch.
"""

from __future__ import annotations

from typing import Any

from test_b6_contamination import (
    NO_POOL_MARKERS,
    PROMPT_TOKENS,
    FakeRequest,
    _admission_step,
    _audit_events,
    _bound_connector,
    _cached_step,
    _connector,
    _serve,
    _write_donor_capture,
)


def _run_fill_script(connector, pool, request) -> list[Any]:
    """Serve one span, then run the steps a served request goes through.

    Admission, two chunked-prefill continuations that allocate nothing, and a
    decode step that appends a block: the whole window in which vLLM keeps
    hashing this request's blocks into its exact prefix cache.
    """
    _write_donor_capture(connector, request)
    table = (list(range(100, 200)),)
    pool.cache_blocks(table[0])
    _serve(connector, request, 12, table)
    metadata = [
        connector.build_connector_meta(_admission_step("r1", request.all_token_ids, table, 12))
    ]
    for computed in (200, 400):
        metadata.append(connector.build_connector_meta(_cached_step("r1", None, computed)))
    pool.cache_blocks([200])
    metadata.append(
        connector.build_connector_meta(_cached_step("r1", ([200],), 400, num_output_tokens=1))
    )
    return metadata


def _non_prefix_cache_stats(connector) -> dict[str, int]:
    """Counters that say nothing about the prefix-cache mitigation itself."""
    return {
        name: value
        for name, value in connector.stats_snapshot.items()
        if "prefix_cache" not in name
    }


def test_eviction_off_leaves_every_filled_block_cached(tmp_path) -> None:
    """The A6 control arm: the knob switches the mitigation off and nothing else.

    Turning it off must not call ``evict_blocks`` at all -- not with an empty
    set, not once -- because the point of the arm is a prefix cache that still
    holds the donor KV, and any call at all would also touch the pool's
    residency metrics.
    """
    connector, pool = _bound_connector(
        tmp_path / "off", evict_filled_blocks_from_prefix_cache=False
    )
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))

    _run_fill_script(connector, pool, request)

    assert pool.calls == [], f"eviction ran with the knob off: {pool.calls}"
    # Everything the connector filled is still exact-matchable by a later
    # request, which is exactly the contamination the A4 arm removes.
    assert pool.hashed == set(range(100, 201))
    stats = connector.stats_snapshot
    assert stats.get("prefix_cache_blocks_evicted", 0) == 0
    # The blocks it would have evicted are counted instead: 103-199 at
    # admission, then the decode block.
    assert stats["prefix_cache_blocks_left_cached"] == 98


def test_eviction_off_changes_nothing_but_the_eviction(tmp_path) -> None:
    """A6 differs from A4 in one behaviour, so the arms stay comparable.

    Same span served, same load metadata handed to the worker on every step,
    same counters apart from the ones that name the mitigation.
    """
    on, on_pool = _bound_connector(tmp_path / "on")
    off, off_pool = _bound_connector(tmp_path / "off", evict_filled_blocks_from_prefix_cache=False)

    on_meta = _run_fill_script(on, on_pool, FakeRequest("r1", list(range(PROMPT_TOKENS))))
    off_meta = _run_fill_script(off, off_pool, FakeRequest("r1", list(range(PROMPT_TOKENS))))

    # Guard against a vacuous comparison: the two arms must actually differ in
    # the one behaviour the knob controls.
    assert off_pool.hashed != on_pool.hashed
    assert [m.loads for m in off_meta] == [m.loads for m in on_meta]
    assert [len(m.stores) for m in off_meta] == [len(m.stores) for m in on_meta]
    assert on_meta[0].loads[0].token_count == 188
    assert _non_prefix_cache_stats(off) == _non_prefix_cache_stats(on)


def test_default_config_still_evicts(tmp_path) -> None:
    """The default is the mitigation: nobody has to know the knob exists."""
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))

    _run_fill_script(connector, pool, request)

    assert connector._config.evict_filled_blocks_from_prefix_cache is True  # noqa: SLF001
    assert pool.evicted_since() == set(range(103, 201))
    stats = connector.stats_snapshot
    assert stats["prefix_cache_blocks_evicted"] == 98
    assert stats.get("prefix_cache_blocks_left_cached", 0) == 0
    # Only the recipient's own computed prefix stays exact-matchable.
    assert pool.hashed == {100, 101, 102}


def test_blocks_left_cached_audit_event_carries_the_join_key_and_a_count(tmp_path) -> None:
    """sembench counts the contaminated blocks off this event on the A6 arm.

    It needs its own event name (the eviction event means the opposite), the
    same join key every other event carries, and a count.
    """
    root = tmp_path / "off"
    connector, pool = _bound_connector(root, evict_filled_blocks_from_prefix_cache=False)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))

    _run_fill_script(connector, pool, request)

    events = _audit_events(root)
    left = [e for e in events if e["event"] == "prefix_cache_blocks_left_cached"]
    assert [e["phase"] for e in left] == ["load_allocated", "running_step"]
    assert not [e for e in events if "evicted" in e["event"]]

    admission = left[0]
    assert admission["blocks_left_cached"] == 97
    assert admission["blocks_left_cached_cumulative"] == 97
    assert admission["block_ids_left_cached"] == list(range(103, 200))
    assert admission["first_approx_block"] == 3
    assert admission["block_table_blocks"] == 100
    assert admission["donor_id"] == "d1"

    # The decode step reports only what it newly left cached, so summing the
    # events over a run counts each block once.
    assert left[1]["blocks_left_cached"] == 1
    assert left[1]["block_ids_left_cached"] == [200]
    assert left[1]["blocks_left_cached_cumulative"] == 98

    # Same join key as every other event for this request, so the arm's
    # contaminated blocks join to the load that produced them.
    allocated = next(e for e in events if e["event"] == "load_allocated")
    for field_name in ("connector_id", "request_id", "request_seq"):
        assert admission[field_name] == allocated[field_name], field_name
    assert admission["namespace"] == allocated["namespace"]
    # event_seq orders events within the request and must not repeat.
    seqs = [e["event_seq"] for e in events if e.get("request_id") == "r1"]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    assert admission["schema_version"] == 2


def test_eviction_off_without_a_block_pool_is_counted_too(tmp_path) -> None:
    """No pool, no way to name the blocks -- and the gap is still counted.

    With the knob off nothing would be evicted anyway, but which filled blocks
    the engine actually hashed is the pool's answer to give, so a run without
    one must not report zero contaminated blocks as if it had checked.
    """
    root = tmp_path / "off"
    connector = _connector(root, evict_filled_blocks_from_prefix_cache=False)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)
    table = (list(range(100, 200)),)

    assert _serve(connector, request, 12, table) == 188
    metadata = connector.build_connector_meta(
        _admission_step("r1", request.all_token_ids, table, 12)
    )

    assert metadata.loads[0].token_count == 188
    stats = connector.stats_snapshot
    assert stats.get("prefix_cache_blocks_left_cached", 0) == 0
    gap = {
        name: value
        for name, value in stats.items()
        if "left_cached" in name and any(marker in name for marker in NO_POOL_MARKERS)
    }
    assert sum(gap.values()) >= 1, f"the unreported-blocks gap must be counted: {sorted(stats)}"
    assert not [e for e in _audit_events(root) if e["event"] == "prefix_cache_blocks_left_cached"]


def _left_cached_events(root) -> list[dict]:
    return [e for e in _audit_events(root) if e["event"] == "prefix_cache_blocks_left_cached"]


def _assert_each_block_counted_once(connector, root, expected: set[int]) -> None:
    """The distinct counter has to count blocks, not passes over the same blocks.

    With the knob off nothing is evicted, so a filled block stays in the pool's
    hash table for the life of the request: a pass that scans it again -- which
    is what a replaced block table forces, since settling is scoped to the table
    it was observed on -- finds every one of them still there. The distinct
    counters are what the contaminated arm is measured by, so a block counted
    twice there inflates the arm it is compared against. (The plain per-pass
    counts beside them do repeat, on purpose: see
    tests/test_b6_arm_comparability.py.)
    """
    events = _left_cached_events(root)
    seen = {block_id for event in events for block_id in event["block_ids_left_cached"]}
    assert seen == expected
    counted = sum(event["distinct_blocks_left_cached"] for event in events)
    assert counted == len(expected), f"a block was counted twice: {events}"
    stats = connector.stats_snapshot
    assert stats["prefix_cache_distinct_blocks_left_cached"] == len(expected)
    # Per request, not per admission: a scorer taking the max per request_id
    # must see every block this request left cached.
    totals = [event["distinct_blocks_left_cached_cumulative"] for event in events]
    assert max(totals) == len(expected)


def test_eviction_off_reports_a_replaced_block_table_once(tmp_path) -> None:
    """Knob off, across the preemption/readmission the eviction path also takes.

    Readmission re-hashes the prefix -- here onto a different set of physical
    blocks -- and clears what was settled on the old table, and the running
    step that carries the replacement scans those same blocks again. With
    nothing evicted they are all still cached, so only a request-scoped record
    of what was already counted keeps the second pass from counting them a
    second time.
    """
    root = tmp_path / "off"
    connector, pool = _bound_connector(root, evict_filled_blocks_from_prefix_cache=False)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)

    first_table = (list(range(100, 200)),)
    pool.cache_blocks(first_table[0])
    _serve(connector, request, 12, first_table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, first_table, 12))

    second_table = (list(range(300, 400)),)
    pool.cache_blocks(second_table[0])
    _serve(connector, request, 12, second_table)
    connector.build_connector_meta(_cached_step("r1", second_table, 12, resumed=True))

    assert pool.calls == [], f"eviction ran with the knob off: {pool.calls}"
    _assert_each_block_counted_once(connector, root, set(range(103, 200)) | set(range(303, 400)))


def test_evicted_cumulative_restarts_on_readmission(tmp_path) -> None:
    """``blocks_evicted_cumulative`` is scoped to the admission, and says so.

    Readmission re-hashes the prefix, and those blocks have to be evicted
    again, so the per-admission total restarts. The distinct total beside it
    does not -- it is the one that can be compared with the contaminated arm
    (tests/test_b6_arm_comparability.py).
    """
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)

    first_table = (list(range(100, 200)),)
    pool.cache_blocks(first_table[0])
    _serve(connector, request, 12, first_table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, first_table, 12))

    second_table = (list(range(300, 400)),)
    pool.cache_blocks(second_table[0])
    _serve(connector, request, 12, second_table)

    evictions = [
        event
        for event in _audit_events(tmp_path)
        if event["event"] == "prefix_cache_blocks_evicted"
    ]
    assert [event["blocks_evicted_cumulative"] for event in evictions] == [97, 97]
    assert [event["distinct_blocks_evicted_cumulative"] for event in evictions] == [97, 194]
    assert connector.stats_snapshot["prefix_cache_blocks_evicted"] == 194
    assert connector.stats_snapshot["prefix_cache_distinct_blocks_evicted"] == 194


def test_eviction_off_reports_a_re_derived_boundary_once(tmp_path) -> None:
    """Knob off, readmitted at a different boundary.

    The second admission starts at block index 10, so 300-309 are the
    recipient's own computed prefix and must never be reported as contaminated,
    and the 90 blocks above it must be reported exactly once between the
    allocation pass and the running step that replaces the table.
    """
    root = tmp_path / "off"
    connector, pool = _bound_connector(root, evict_filled_blocks_from_prefix_cache=False)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)

    first_table = (list(range(100, 200)),)
    pool.cache_blocks(first_table[0])
    _serve(connector, request, 12, first_table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, first_table, 12))

    second_table = (list(range(300, 400)),)
    pool.cache_blocks(second_table[0])
    assert _serve(connector, request, 40, second_table) == 160
    connector.build_connector_meta(_cached_step("r1", second_table, 40, resumed=True))

    assert pool.calls == [], f"eviction ran with the knob off: {pool.calls}"
    _assert_each_block_counted_once(connector, root, set(range(103, 200)) | set(range(310, 400)))
    assert set(range(300, 310)) <= pool.hashed

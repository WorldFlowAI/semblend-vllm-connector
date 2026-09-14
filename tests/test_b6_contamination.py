"""B6: connector-filled blocks must not stay in vLLM's own prefix cache.

vLLM hashes the blocks this connector fills under the *recipient's* token ids
and inserts them into the exact prefix cache in the same ``allocate_slots``
call that allocated them -- before any donor KV has been written. A later
request then hits those blocks through the local prefix cache, which sits
upstream of the connector, so no gate, no TTL and no audit event ever fires.
The damage is transitive: a poisoned block's hash is the parent hash of
everything after it, so the contaminated set is the served span plus the whole
rest of the request, and it keeps growing.

The mitigation is to drop the connector-filled tail of each served request's
block table from ``BlockPool`` (``$V/vllm/v1/core/block_pool.py``), whose
``evict_blocks`` removes the hash entry without freeing the block and without
touching its ref count. These tests pin what makes that safe:

* only the connector-filled tail goes -- blocks below the boundary hold the
  recipient's own computed prefix and are vLLM's to cache;
* each block reaches ``evict_blocks`` at most once, because
  ``_maybe_evict_cached_block`` reports a block to the pool's metrics
  collector before it checks whether the block has a hash at all;
* a resumed request's block table is replaced, not appended to;
* the null block is never passed in;
* ``min_boundary_tokens`` keeps boundary-0 serves from starving the shared
  prefix this whole scheme depends on;
* the first connector-filled block index is re-derived after a preemption;
* and a connector with no block pool bound says so in its counters rather than
  leaving the cache quietly poisoned.

Scheduler-side only: nothing here touches KV tensors, so no torch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from test_connector_discovery import (
    FakeBlocks,
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.namespace import namespace_for_request
from semblend_vllm_connector.types import (
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
)

BLOCK_SIZE = 4
PROMPT_TOKENS = 400
# Donor span in the target frame: [0, 200), served from donor position 1000.
SPAN_TOKENS = 200
DONOR_START = 1000

# A connector with no block pool bound must name that gap in its counters. The
# counter's exact name is the implementation's to pick; what a test can pin is
# that the missing pool is what the name says.
NO_POOL_MARKERS = ("without_pool", "no_pool", "unbound", "unavailable")


class _FakeBlock:
    """``KVCacheBlock``: ``block_hash`` is set while the pool's table names it."""

    def __init__(self, pool: "FakeBlockPool", block_id: int, *, is_null: bool = False) -> None:
        self._pool = pool
        self.block_id = block_id
        self.is_null = is_null

    @property
    def block_hash(self) -> str | None:
        return f"hash-{self.block_id}" if self.block_id in self._pool.hashed else None


class FakeBlockPool:
    """The slice of vLLM's ``BlockPool`` a connector is allowed to touch.

    Mirrors ``$V/vllm/v1/core/block_pool.py``: block 0 is the null placeholder
    popped off the free queue in ``__init__`` and never ref-counted; a block is
    cached exactly while the hash table names it; and ``evict_blocks`` drops
    that entry (``_maybe_evict_cached_block``) without freeing the block or
    moving its ref count. ``cache_blocks`` stands in for the engine's own
    ``cache_full_blocks``, which is what puts a connector's fill into the exact
    prefix cache in the first place. Calls are recorded one by one so a test
    can tell a second eviction of a block from the first.
    """

    def __init__(self, num_gpu_blocks: int = 1024, hash_block_size: int = BLOCK_SIZE) -> None:
        self.hashed: frozenset[int] = frozenset()
        self.blocks = [
            _FakeBlock(self, idx, is_null=(idx == 0)) for idx in range(num_gpu_blocks)
        ]
        self.null_block = self.blocks[0]
        self.hash_block_size = hash_block_size
        self.enable_caching = True
        self.calls: list[tuple[int, ...]] = []

    @property
    def cached_block_hashes_by_block(self) -> dict[int, set[str]]:
        return {block_id: {f"hash-{block_id}"} for block_id in self.hashed}

    def cache_blocks(self, block_ids: Iterable[int]) -> None:
        """What the engine does to the blocks allocate_slots just handed out."""
        self.hashed = self.hashed | {int(block_id) for block_id in block_ids}

    def evict_blocks(self, block_ids: Iterable[int]) -> None:
        call = tuple(int(block_id) for block_id in block_ids)
        for block_id in call:
            # The real pool asserts this and names the connector in the message.
            assert 0 <= block_id < len(self.blocks), f"block id {block_id} out of range"
        self.calls.append(call)
        self.hashed = self.hashed - set(call)

    def mark(self) -> int:
        """Cursor into the call log, so a test can scope one scheduling step."""
        return len(self.calls)

    def evicted_since(self, mark: int = 0) -> set[int]:
        return {block_id for call in self.calls[mark:] for block_id in call}

    def eviction_sequence(self, mark: int = 0) -> list[int]:
        """Every id handed to ``evict_blocks`` since ``mark``, repeats kept."""
        return [block_id for call in self.calls[mark:] for block_id in call]


@dataclass
class _NewReq:
    """vLLM ``NewRequestData``, trimmed to what the connector reads."""

    req_id: str
    prompt_token_ids: list[int]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int = 0


@dataclass
class _CachedReqs:
    """vLLM ``CachedRequestData`` (``$V/vllm/v1/core/sched/output.py``).

    For ids in ``resumed_req_ids`` ``new_block_ids`` *replaces* the request's
    block table; for everyone else it is appended to it.
    """

    req_ids: list[str] = field(default_factory=list)
    resumed_req_ids: set[str] = field(default_factory=set)
    new_token_ids: list[list[int]] = field(default_factory=list)
    all_token_ids: dict[str, list[int]] = field(default_factory=dict)
    new_block_ids: list[Any] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)


@dataclass
class _SchedulerOutput:
    scheduled_new_reqs: list[_NewReq] = field(default_factory=list)
    scheduled_cached_reqs: _CachedReqs | None = None
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)


class _ScriptedProvider:
    def __init__(self, result: SemanticLookupResult) -> None:
        self.result = result

    def lookup(self, request):  # noqa: ARG002 - the fixture answers every lookup alike
        return self.result

    def register(self, registration):  # noqa: ARG002
        return True


def _span_result() -> SemanticLookupResult:
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.99,
        reusable_token_count=SPAN_TOKENS,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[
            SemanticSegment(
                donor_id="d1",
                donor_start=DONOR_START,
                target_start=0,
                token_count=SPAN_TOKENS,
            )
        ],
    )


def _connector(tmp_path, **overrides) -> SemBlendVllmConnector:
    extra = {
        "mode": "semantic_span_experimental",
        "provider": "local",
        "min_prompt_tokens": 4,
        "min_similarity": 0.3,
        "min_semantic_span": 8,
        "kv_storage_path": str(tmp_path / "kv"),
        "max_materialized_tokens": 4096,
        "audit_path": str(tmp_path / "audit.jsonl"),
        **overrides,
    }
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(extra),
        cache_config=FakeCacheConfig(block_size=BLOCK_SIZE),
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)
    connector._provider = _ScriptedProvider(_span_result())  # noqa: SLF001
    return connector


def _bound_connector(tmp_path, **overrides) -> tuple[SemBlendVllmConnector, FakeBlockPool]:
    connector = _connector(tmp_path, **overrides)
    pool = FakeBlockPool()
    connector.bind_gpu_block_pool(pool)
    return connector, pool


def _write_donor_capture(connector, request, donor_id: str = "d1", tokens: int = 4096) -> None:
    """Record a donor KV capture on disk; the advertise path trims spans to it."""
    namespace = namespace_for_request(
        connector._config, connector._vllm_config, request  # noqa: SLF001
    )
    os.makedirs(connector._donor_dir(donor_id, namespace), exist_ok=True)  # noqa: SLF001
    path = connector._donor_metadata_path(donor_id, namespace)  # noqa: SLF001
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"token_count": tokens}, f)


def _serve(connector, request, boundary: int, block_ids: tuple[list[int], ...]) -> int:
    """Advertise, allocate and publish one span load, as the scheduler does."""
    matched, _ = connector.get_num_new_matched_tokens(request, boundary)
    assert matched > 0, "fixture must actually serve a span"
    connector.update_state_after_alloc(request, FakeBlocks(block_ids), matched)
    return matched


def _admission_step(
    request_id: str,
    token_ids: list[int],
    block_ids: tuple[list[int], ...],
    num_computed_tokens: int,
) -> _SchedulerOutput:
    return _SchedulerOutput(
        scheduled_new_reqs=[
            _NewReq(
                req_id=request_id,
                prompt_token_ids=list(token_ids),
                block_ids=block_ids,
                num_computed_tokens=num_computed_tokens,
            )
        ],
        num_scheduled_tokens={request_id: len(token_ids) - num_computed_tokens},
    )


def _cached_step(
    request_id: str,
    new_block_ids: tuple[list[int], ...] | None,
    num_computed_tokens: int,
    *,
    resumed: bool = False,
    num_output_tokens: int = 0,
) -> _SchedulerOutput:
    return _SchedulerOutput(
        scheduled_cached_reqs=_CachedReqs(
            req_ids=[request_id],
            resumed_req_ids={request_id} if resumed else set(),
            new_token_ids=[[]],
            new_block_ids=[new_block_ids],
            num_computed_tokens=[num_computed_tokens],
            num_output_tokens=[num_output_tokens],
        ),
        num_scheduled_tokens={request_id: 1},
    )


def _stats_matching(stats: dict[str, int], *needles: str) -> dict[str, int]:
    """Counters whose name contains every needle.

    The exact counter names are the implementation's to pick; what a test can
    legitimately pin is that the decision is counted at all, and that it
    carries the repo's ``_per_attempt`` suffix when the decision differs
    between scheduling attempts of the same request.
    """
    return {name: value for name, value in stats.items() if all(n in name for n in needles)}


def _audit_events(tmp_path) -> list[dict]:
    path = tmp_path / "audit.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_eviction_starts_at_the_first_connector_filled_block(tmp_path) -> None:
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)
    table = (list(range(100, 200)),)
    # allocate_slots hashes and caches the blocks it hands out, this fill
    # among them, before the connector is told the allocation happened.
    pool.cache_blocks(table[0])

    assert _serve(connector, request, 12, table) == 188
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, table, 12))

    # Boundary 12 at block size 4: the connector writes from block index 3 on.
    # 100-102 hold the recipient's own computed prefix, which vLLM cached
    # legitimately and other requests share -- evicting those would be a
    # self-inflicted prefix-cache regression. Everything from index 3 up is
    # donor KV living under the recipient's hash, decode tail included,
    # because a poisoned block's hash parents every block hashed after it.
    assert pool.evicted_since() == set(range(103, 200))
    assert not pool.evicted_since() & {100, 101, 102}
    assert connector.stats_snapshot["prefix_cache_blocks_evicted"] == 97
    # End state: the only thing left exact-matchable is the prefix the
    # recipient actually computed.
    assert pool.hashed == {100, 101, 102}


def test_eviction_emits_a_per_load_audit_event(tmp_path) -> None:
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)
    table = (list(range(100, 200)),)
    pool.cache_blocks(table[0])

    _serve(connector, request, 12, table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, table, 12))

    # Without a record per load there is no way to tell "the mitigation ran and
    # found nothing to evict" from "the mitigation never ran".
    evictions = [
        event
        for event in _audit_events(tmp_path)
        if "evict" in event["event"] and event.get("request_id") == "r1"
    ]
    assert evictions, "expected an eviction audit event for the served request"
    # Field naming is the implementation's; the count has to be in there.
    assert any(97 in event.values() for event in evictions)


def test_each_block_is_evicted_at_most_once_across_steps(tmp_path) -> None:
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)
    table = (list(range(100, 200)),)
    pool.cache_blocks(table[0])

    _serve(connector, request, 12, table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, table, 12))
    # Chunked-prefill continuations: nothing new is allocated, but the engine
    # runs its caching pass over the request on every one of them.
    for computed in (200, 300, 400):
        connector.build_connector_meta(_cached_step("r1", None, computed))
    # Decode appends a block, which is new, gets hashed, and does have to go.
    pool.cache_blocks([200])
    connector.build_connector_meta(_cached_step("r1", ([200],), 400, num_output_tokens=1))
    # An already-evicted block reappearing in the hash table is the
    # re-registration path (_cache_partial_tail_block); it is barred at startup
    # by the hash_block_size gate, not handled here. Evicting it a second time
    # would call metrics_collector.on_block_evicted again and destroy that
    # block's residency tracking, so the connector must not.
    pool.cache_blocks([150])
    connector.build_connector_meta(_cached_step("r1", None, 400, num_output_tokens=2))

    sequence = pool.eviction_sequence()
    assert len(sequence) == len(set(sequence)), f"block evicted twice: {sequence}"
    assert set(sequence) == set(range(103, 201))
    assert connector.stats_snapshot["prefix_cache_blocks_evicted"] == 98


def test_resumed_request_replaces_its_block_table_instead_of_appending(tmp_path) -> None:
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)

    first_table = (list(range(100, 200)),)
    pool.cache_blocks(first_table[0])
    _serve(connector, request, 12, first_table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, first_table, 12))
    pool.cache_blocks([200])
    connector.build_connector_meta(_cached_step("r1", ([200],), 400, num_output_tokens=1))
    assert pool.evicted_since() == set(range(103, 201))

    # Preempted, then readmitted. For ids in resumed_req_ids new_block_ids is
    # the whole block table, not an addition to it. An implementation that
    # appends keeps the freed 100-200 on the request -- by now handed to other
    # requests -- and shifts every index, so the readmitted request's own
    # exact-prefix blocks (300-302, shared and ref-counted by everything else
    # on that prefix) are evicted along with them.
    second_table = (list(range(300, 400)),)
    pool.cache_blocks(second_table[0])
    mark = pool.mark()
    _serve(connector, request, 12, second_table)
    connector.build_connector_meta(_cached_step("r1", second_table, 12, resumed=True))

    assert pool.evicted_since(mark) == set(range(303, 400))
    assert {300, 301, 302} <= pool.hashed


def test_null_block_is_never_evicted(tmp_path) -> None:
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)

    # vLLM pads a block table with the shared null placeholder (block id 0,
    # popped off the free queue in BlockPool.__init__) instead of a real
    # allocation. Handing it to evict_blocks would strip hash metadata from a
    # block every request in the engine shares. The fixture puts the null block
    # in the hash table on purpose, so only an explicit null filter keeps it
    # out of the eviction set.
    ids = [0 if idx in (3, 50) else 100 + idx for idx in range(100)]
    table = (ids,)
    pool.cache_blocks(ids)

    _serve(connector, request, 12, table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, table, 12))

    assert 0 not in pool.evicted_since()
    assert pool.evicted_since() == {100 + idx for idx in range(3, 100)} - {103, 150}
    assert connector.stats_snapshot["prefix_cache_blocks_evicted"] == 95


def test_min_boundary_tokens_declines_boundary_zero_per_attempt(tmp_path) -> None:
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))

    # Control: with the gate off (the default) boundary 0 is servable, so the
    # decline below is attributable to the knob and to nothing else.
    ungated = _connector(tmp_path)
    _write_donor_capture(ungated, request)
    assert ungated.get_num_new_matched_tokens(request, 0)[0] == SPAN_TOKENS

    # Serving at boundary 0 evicts the request's entire block table, so the
    # shared wrapper never enters the prefix cache, every boundary stays 0 and
    # the mid-prompt lane this connector exists for never materializes.
    # Declining there lets unserved requests prime the wrapper cleanly.
    connector = _connector(tmp_path, min_boundary_tokens=BLOCK_SIZE)
    _write_donor_capture(connector, request)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

    declines = _stats_matching(connector.stats_snapshot, "boundary", "min")
    assert sum(declines.values()) == 2, f"expected a per-attempt decline, got {declines}"
    # The decision moves with num_computed_tokens, so it is attempt-scoped and
    # has to say so in its name (see the counter convention in connector.py).
    assert all(name.endswith("_per_attempt") for name in declines), declines

    # Same request one block further on: the gate is a property of the
    # boundary, not of the request, so it must not latch.
    assert connector.get_num_new_matched_tokens(request, BLOCK_SIZE)[0] == 196


def test_first_connector_filled_block_is_re_derived_after_preemption(tmp_path) -> None:
    connector, pool = _bound_connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)

    first_table = (list(range(100, 200)),)
    pool.cache_blocks(first_table[0])
    _serve(connector, request, 12, first_table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, first_table, 12))
    assert pool.evicted_since() == set(range(103, 200))

    # _preempt_request resets num_computed_tokens to 0 and drops
    # num_cached_block, so the prefix is re-hashed onto new physical blocks and
    # the connector is re-queried -- at a different boundary. Holding on to the
    # first answer (block index 3) would evict from the middle of a prefix vLLM
    # has every right to keep: 300-309 are computed, not connector-filled.
    second_table = (list(range(300, 400)),)
    pool.cache_blocks(second_table[0])
    mark = pool.mark()
    assert _serve(connector, request, 40, second_table) == 160
    connector.build_connector_meta(_cached_step("r1", second_table, 40, resumed=True))

    assert pool.evicted_since(mark) == set(range(310, 400))
    assert set(range(300, 310)) <= pool.hashed


def test_unbound_block_pool_is_counted_not_silently_ignored(tmp_path) -> None:
    # bind_gpu_block_pool is called unconditionally by the stock scheduler, but
    # a harness, a fork or a future refactor can leave it uncalled. Loads must
    # keep working, and the fact that nothing is being evicted has to be
    # visible in the counters rather than surfacing later as an unexplained
    # prefix-cache hit rate in the contaminated arm.
    connector = _connector(tmp_path)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    _write_donor_capture(connector, request)
    table = (list(range(100, 200)),)

    assert _serve(connector, request, 12, table) == 188
    metadata = connector.build_connector_meta(
        _admission_step("r1", request.all_token_ids, table, 12)
    )
    assert len(metadata.loads) == 1
    assert metadata.loads[0].token_count == 188

    stats = connector.stats_snapshot
    assert stats.get("prefix_cache_blocks_evicted", 0) == 0
    gap = {
        name: value
        for name, value in stats.items()
        if "evict" in name and any(marker in name for marker in NO_POOL_MARKERS)
    }
    assert sum(gap.values()) >= 1, f"the eviction gap must be counted: {sorted(stats)}"

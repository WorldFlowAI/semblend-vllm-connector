"""E2 — one regression test per Phase-0 connector blocker (B1..B5).

Each test encodes the contract the fix must satisfy, not the numbers the
current implementation happens to produce. All five are live: a failure here
is a blocker reopening, not a known gap.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest
from test_connector_discovery import (
    FakeBlocks,
    FakeCacheConfig,
    FakeForwardContext,
    FakeKvTransferConfig,
    FakeRequest,
    FakeSchedulerOutput,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.namespace import namespace_for_request
from semblend_vllm_connector.types import (
    MaterializationKind,
    PendingLoad,
    PendingStore,
    SemanticLookupResult,
    SemanticSegment,
    SemBlendConnectorMetadata,
)


def _connector(tmp_path, role, block_size=4, **extra):
    settings = {
        "mode": "semantic_span_experimental",
        "provider": "local",
        # At least min_semantic_span: the config warns about the gap otherwise.
        "min_prompt_tokens": 8,
        "min_similarity": 0.3,
        "min_semantic_span": 8,
        "max_materialized_tokens": 4096,
        "kv_storage_path": str(tmp_path),
        "log_decisions": False,
    }
    settings.update(extra)
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(settings),
        cache_config=FakeCacheConfig(block_size=block_size),
    )
    return SemBlendVllmConnector(vllm_config, role)


class _ScriptedProvider:
    """Provider stub: hands back one fixed alignment result per lookup."""

    def __init__(self, result):
        self.result = result

    def lookup(self, request):
        return self.result

    def register_donor(self, registration):
        return True


def _span_result(donor_start, target_start, token_count):
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.99,
        reusable_token_count=token_count,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[
            SemanticSegment(
                donor_id="d1",
                donor_start=donor_start,
                target_start=target_start,
                token_count=token_count,
            )
        ],
    )


def _write_donor_capture(connector, request, donor_id, token_count) -> None:
    """Record the donor's captured length where the advertise path reads it."""
    namespace = namespace_for_request(
        connector._config,
        connector._vllm_config,
        request,  # noqa: SLF001
    )
    os.makedirs(connector._donor_dir(donor_id, namespace), exist_ok=True)  # noqa: SLF001
    with open(connector._donor_metadata_path(donor_id, namespace), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": token_count}, f)


def _pending_store(request_id, token_count, block_ids, final=True):
    # final: this store completes the donor, which is what tells the worker to
    # publish its metadata record. A capture the scheduler has not closed is
    # deliberately not publishable.
    return PendingStore(
        request_id=request_id,
        token_ids=list(range(token_count)),
        token_count=token_count,
        namespace="ns",
        block_ids=block_ids,
        final=final,
    )


def _stub_safetensors(monkeypatch, load_result):
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda filename: load_result  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)


# --------------------------------------------------------------------------
# B1 — destination offset
# --------------------------------------------------------------------------


def test_b1_boundary_load_never_writes_the_local_hit_prefix(tmp_path, monkeypatch) -> None:
    """A span served at boundary 12 must land in the blocks covering
    [12, 20) and leave the blocks covering [0, 12) byte-identical.

    Today `_slot_mapping` builds the destination from `block_ids[0]` at
    block index 0 and takes the first `token_count` slots, so the 8 served
    tokens are written into the first two blocks of the group — here
    physical blocks 7 and 2, which hold the recipient's *local* prefix-cache
    hit and are ref-counted into every other request sharing that prefix.
    The prefix assertion below catches that write; the destination
    assertion catches the region [12, 20) being left uninitialized.
    """
    torch = pytest.importorskip("torch")

    connector = _connector(tmp_path, KVConnectorRole.WORKER)

    class _HF:
        rope_theta = 10000.0
        head_dim = 16

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001

    donor_kv = torch.randn(2, 64, 32)  # [2, tokens, H*D], the extract contract
    _stub_safetensors(monkeypatch, {"kv_cache": donor_kv})

    # Physical blocks for target tokens [0, 20) at block_size 4. The first
    # three cover the local-hit prefix [0, 12) and are deliberately not the
    # identity mapping, so a destination built from block index 0 is
    # distinguishable from one built from the span's own blocks.
    group = [7, 2, 9, 4, 5]
    prefix_blocks = torch.tensor([7, 2, 9])
    dst_layer = torch.randn(2, 10, 4, 2, 16)  # [2, pages, block_size, H, D]
    prefix_before = dst_layer[:, prefix_blocks, ...].clone()

    load = PendingLoad(
        request_id="r1",
        donor_id="d1",
        token_count=8,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns",
        block_ids=(group,),
        donor_start=40,
        target_start=12,
    )
    connector.register_kv_caches({"model.layers.0.self_attn.attn": dst_layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[load]))

    attn_metadata = object()
    connector.start_load_kv(FakeForwardContext(attn_metadata=attn_metadata))

    torch.testing.assert_close(
        dst_layer[:, prefix_blocks, ...],
        prefix_before,
        msg="load at boundary 12 overwrote the shared local-hit prefix blocks",
    )

    # Blocks 4 and 5 cover target tokens [12, 16) and [16, 20): slots 16..23.
    span_slots = torch.arange(16, 24)
    injected = connector._extract_kv_from_layer(dst_layer, span_slots, attn_metadata)  # noqa: SLF001
    expected = connector._semantic_span_slice(donor_kv, load, attn_metadata)  # noqa: SLF001
    torch.testing.assert_close(
        injected, expected, msg="served span did not land at [target_start, +token_count)"
    )


# --------------------------------------------------------------------------
# B2 — advertised count cap
# --------------------------------------------------------------------------


def test_b2_advertised_count_leaves_a_token_for_the_engine(tmp_path) -> None:
    """A 64-token prompt (exact block multiple) whose span runs to the end
    must never be advertised all the way to `num_tokens`.

    Today the semantic-span path clamps only by `max_materialized_tokens`,
    so the block-aligned span [0, 64) served from boundary 16 advertises 48
    and the scheduler is left with `num_new_tokens == 0`, tripping
    `assert num_new_tokens > 0` in vLLM's sync scheduler path. The
    `_cacheable_prefix_tokens` cap already applied on the REQUEST_ONLY path
    is missing here; the assertion below is that cap's observable contract.
    """
    connector = _connector(tmp_path, KVConnectorRole.SCHEDULER)
    prompt = list(range(64))
    recipient = FakeRequest("r1", prompt)
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        _span_result(donor_start=0, target_start=0, token_count=len(prompt))
    )
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    boundary = 16
    matched, async_load = connector.get_num_new_matched_tokens(recipient, boundary)

    assert async_load is False
    assert matched > 0, "span covering the boundary must still be servable"
    assert len(prompt) - (boundary + matched) >= 1, (
        f"advertised {matched} from boundary {boundary} of {len(prompt)} tokens; "
        "the engine is left with zero tokens to compute"
    )


# --------------------------------------------------------------------------
# B3 — PendingLoad registration point
# --------------------------------------------------------------------------


def test_b3_failed_allocation_leaves_no_pending_load(tmp_path) -> None:
    """A lookup that advertises tokens but never gets blocks must not reach
    `build_connector_meta`.

    Today the PendingLoad is constructed inside `get_num_new_matched_tokens`,
    which vLLM documents as side-effect-free. There are several exits between
    that hook and `update_state_after_alloc` (allocate_slots returning None,
    deferred prefills, budget exhaustion, a partial-tail loss that zeroes
    `num_external_tokens`), and on every one of them the load keeps
    `block_ids=None`, is emitted by `build_connector_meta`, and raises
    "load missing destination blocks" inside `start_load_kv` on the worker.
    Both scenarios below assert the emitted metadata is empty.
    """
    connector = _connector(tmp_path, KVConnectorRole.SCHEDULER)
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        _span_result(donor_start=0, target_start=0, token_count=48)
    )

    # allocate_slots returned None: update_state_after_alloc is never called.
    dropped = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, dropped, "d1", token_count=4096)
    matched, _ = connector.get_num_new_matched_tokens(dropped, 16)
    assert matched > 0, "test needs an advertised load to be meaningful"

    metadata = connector.build_connector_meta(FakeSchedulerOutput())
    assert metadata.loads == [], "unallocated load survived into connector metadata"

    # Partial-tail loss: the scheduler zeroes num_external_computed_tokens and
    # calls the hook with 0, so no block ids are ever recorded.
    zeroed = FakeRequest("r2", list(range(100)))
    matched, _ = connector.get_num_new_matched_tokens(zeroed, 16)
    assert matched > 0
    connector.update_state_after_alloc(zeroed, FakeBlocks(([1, 2, 3],)), 0)

    metadata = connector.build_connector_meta(FakeSchedulerOutput())
    assert metadata.loads == [], "zero-token allocation left a load behind"


# --------------------------------------------------------------------------
# B4 — donor storage lifecycle
# --------------------------------------------------------------------------


def _save_memory_donor(connector, donor_id, kv_layer, token_count=4) -> None:
    """One-layer capture of `donor_id` into a memory-backend worker."""
    connector.bind_connector_metadata(
        SemBlendConnectorMetadata(
            stores=[_pending_store(request_id=donor_id, token_count=token_count, block_ids=([0],))]
        )
    )
    connector.save_kv_layer("layer0", kv_layer, attn_metadata=object())
    # The end of the step vLLM saved from: the writer is joined here, so the
    # donor is on the store rather than in the queue.
    connector.wait_for_save()


def test_b4_eviction_clears_metadata_and_missing_donor_declines(tmp_path) -> None:
    """Evicting a donor must retract its advertised length, and a load
    against a donor whose tensors are gone must decline, not raise.

    `save_kv_layer` writes metadata.json unconditionally before the backend
    branch, and the memory backend's eviction only drops the in-RAM entry, so
    the scheduler-role connector keeps reading a stale token_count off disk
    for a donor with no tensors. The worker then falls through to
    `load_file` on a path that does not exist and dies with FileNotFoundError
    inside `start_load_kv`.
    """
    torch = pytest.importorskip("torch")

    evicting = _connector(
        tmp_path / "mem",
        KVConnectorRole.WORKER,
        kv_storage_backend="memory",
        kv_memory_max_donors=1,
    )
    kv_layer = torch.randn(2, 6, 4, 2, 8)
    for donor_id in ("d1", "d2"):
        _save_memory_donor(evicting, donor_id, kv_layer)

    assert evicting.stats_snapshot.get("memory_store_evictions", 0) == 1
    assert not os.path.exists(evicting._donor_metadata_path("d1", "ns"))  # noqa: SLF001
    assert evicting._stored_donor_token_count("d1", "ns") == 0  # noqa: SLF001
    # Eviction took the right donor: the survivor is still fully advertised.
    assert os.path.exists(evicting._donor_metadata_path("d2", "ns"))  # noqa: SLF001
    assert evicting._stored_donor_token_count("d2", "ns") == 4  # noqa: SLF001

    # A load whose donor is gone: the layers exist, so the load path reaches
    # storage and must decline there instead of raising. Two layers, so the
    # counter is checked as a per-load count (its name carries no unit
    # suffix): a donor missing for one layer is missing for all of them, and
    # a per-layer tally would read as two lost loads.
    loading = _connector(tmp_path / "gone", KVConnectorRole.WORKER)
    dst_layers = {"layer0": torch.zeros(2, 6, 4, 2, 8), "layer1": torch.zeros(2, 6, 4, 2, 8)}
    loading.register_kv_caches(dst_layers)
    loading.bind_connector_metadata(
        SemBlendConnectorMetadata(
            loads=[
                PendingLoad(
                    request_id="r1",
                    donor_id="evicted",
                    token_count=4,
                    materialization_kind=MaterializationKind.REQUEST_ONLY,
                    namespace="ns",
                    block_ids=([0],),
                )
            ]
        )
    )

    loading.start_load_kv(FakeForwardContext(attn_metadata=object()))

    stats = loading.stats_snapshot
    assert stats.get("load_declined_donor_gone", 0) == 1
    assert stats.get("loads_materialized_total", 0) == 0
    for layer_name, dst in dst_layers.items():
        assert not dst.any(), f"declined load wrote into {layer_name}"


def test_b4_semantic_span_load_against_missing_donor_fails_loud(tmp_path) -> None:
    """For a semantic-span load the scheduler has already skipped compute on
    the span, so a missing donor cannot be a quiet decline: the connector
    counts it and then raises its own loud invariant (0 layers materialized)
    rather than decoding over uninitialized KV — and rather than leaking a
    FileNotFoundError out of storage, which is what happens today.
    """
    torch = pytest.importorskip("torch")

    loading = _connector(tmp_path, KVConnectorRole.WORKER)
    loading.register_kv_caches({"layer0": torch.zeros(2, 6, 4, 2, 8)})
    loading.bind_connector_metadata(
        SemBlendConnectorMetadata(
            loads=[
                PendingLoad(
                    request_id="r1",
                    donor_id="evicted",
                    token_count=8,
                    materialization_kind=MaterializationKind.SEMANTIC_SPAN,
                    namespace="ns",
                    block_ids=([0, 1],),
                    donor_start=0,
                    target_start=0,
                )
            ]
        )
    )

    # FileNotFoundError is an OSError, so it fails this clause rather than
    # satisfying it.
    with pytest.raises(RuntimeError):
        loading.start_load_kv(FakeForwardContext(attn_metadata=object()))

    stats = loading.stats_snapshot
    assert stats.get("load_declined_donor_gone", 0) == 1
    assert stats.get("loads_materialized_total", 0) == 0


def test_b4_a_layer_left_unreadable_declines_like_a_missing_one(tmp_path) -> None:
    """A truncated layer file is what a write that failed part-way leaves.

    The store filling up mid-``save_file`` leaves a file safetensors refuses
    to parse, and it raises its own error type for that -- not an OSError. A
    load site that catches only OSError lets it out of ``start_load_kv`` and
    into the engine's forward pass. Unreadable and absent are the same donor
    from here, and both are declines.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    loading = _connector(tmp_path, KVConnectorRole.WORKER)
    donor_dir = loading._donor_dir("truncated", "ns")  # noqa: SLF001
    os.makedirs(donor_dir, exist_ok=True)
    # A record, so the donor is advertised, and a layer file that stops in
    # the middle of its own header.
    with open(os.path.join(donor_dir, "metadata.json"), "w") as f:
        json.dump({"token_count": 4}, f)
    with open(os.path.join(donor_dir, "layer0.safetensors"), "wb") as f:
        f.write(b'\x40\x00\x00\x00\x00\x00\x00\x00{"kv_cache"')

    loading.register_kv_caches({"layer0": torch.zeros(2, 6, 4, 2, 8)})
    loading.bind_connector_metadata(
        SemBlendConnectorMetadata(
            loads=[
                PendingLoad(
                    request_id="r1",
                    donor_id="truncated",
                    token_count=4,
                    materialization_kind=MaterializationKind.REQUEST_ONLY,
                    namespace="ns",
                    block_ids=([0],),
                )
            ]
        )
    )

    loading.start_load_kv(FakeForwardContext(attn_metadata=object()))

    stats = loading.stats_snapshot
    assert stats.get("load_declined_donor_gone", 0) == 1
    assert stats.get("loads_materialized_total", 0) == 0


def test_b4_memory_backend_evicts_least_recently_used_donor(tmp_path) -> None:
    """Eviction order must follow use, not arrival.

    Today the memory backend evicts with `popitem(last=False)` and never
    `move_to_end`s on a read, so a donor that is being served every step is
    the first one dropped once the cap is reached — the hottest donor is the
    one the store forgets. The length read is one of the two read paths;
    touching d1 through it must move d1 behind d2 in the eviction order.
    """
    torch = pytest.importorskip("torch")

    connector = _connector(
        tmp_path,
        KVConnectorRole.WORKER,
        kv_storage_backend="memory",
        kv_memory_max_donors=2,
    )
    kv_layer = torch.randn(2, 6, 4, 2, 8)
    _save_memory_donor(connector, "d1", kv_layer)
    _save_memory_donor(connector, "d2", kv_layer)

    assert connector._stored_donor_token_count("d1", "ns") == 4  # noqa: SLF001

    _save_memory_donor(connector, "d3", kv_layer)

    assert connector.stats_snapshot.get("memory_store_evictions", 0) == 1
    assert connector._stored_donor_token_count("d2", "ns") == 0, (  # noqa: SLF001
        "arrival-order eviction kept the idle donor and dropped the one in use"
    )
    assert connector._stored_donor_token_count("d1", "ns") == 4  # noqa: SLF001
    assert connector._stored_donor_token_count("d3", "ns") == 4  # noqa: SLF001


# --------------------------------------------------------------------------
# B5 — donor capture clamp
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_tokens", "expected_capture"),
    [
        pytest.param(2048, 2048, id="block-aligned-chunk"),
        # The batched-token budget is shared across co-scheduled requests, so
        # a chunk is not block-aligned in general; the capture floors to the
        # last whole block the chunk computed.
        pytest.param(2050, 2048, id="ragged-chunk-floors-to-block"),
        # A chunk covering the whole prompt must not lift the existing cap
        # that leaves the engine at least one token: 4000 -> 3984.
        pytest.param(4000, 3984, id="whole-prompt-keeps-engine-token"),
    ],
)
def test_b5_donor_capture_clamped_to_the_first_chunk(
    tmp_path, chunk_tokens, expected_capture
) -> None:
    """Under chunked prefill only the first chunk has been computed when the
    donor is captured, so the stored length must be that chunk, not the
    prompt.

    `_build_store_metadata` derives token_count from `len(token_ids)` alone
    and ignores `SchedulerOutput.num_scheduled_tokens`, so a 4,000-token
    prompt scheduled in a 2,048-token first chunk is captured as 3,984
    tokens — roughly half of them uninitialized — and that inflated length is
    published as the donor's authoritative length for every later recipient.
    """
    torch = pytest.importorskip("torch")

    block_size = 16
    scheduler = _connector(tmp_path, KVConnectorRole.SCHEDULER, block_size=block_size)

    prompt = list(range(4000))
    new_req = FakeRequest("d1", prompt)
    new_req.block_ids = ([*range(len(prompt) // block_size)],)
    scheduler_output = FakeSchedulerOutput(scheduled_new_reqs=[new_req])
    # vLLM 0.29 SchedulerOutput.num_scheduled_tokens: dict[req_id, int].
    scheduler_output.num_scheduled_tokens = {"d1": chunk_tokens}

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert len(metadata.stores) == 1
    assert metadata.stores[0].token_count == expected_capture, (
        f"chunk of {chunk_tokens} scheduled tokens captured as "
        f"{metadata.stores[0].token_count}; tokens past the chunk were never computed"
    )

    # The length the worker publishes is what recipients plan against.
    worker = _connector(tmp_path, KVConnectorRole.WORKER, block_size=block_size)
    worker.bind_connector_metadata(metadata)
    worker.save_kv_layer(
        "layer0",
        torch.zeros(2, len(new_req.block_ids[0]), block_size, 2, 8),
        attn_metadata=object(),
    )
    worker.wait_for_save()

    stored = metadata.stores[0]
    if not stored.final:
        # Mid-prefill this donor is deliberately not discoverable. The record
        # is what publishes it, and it is written when the capture closes --
        # here, because the request finished before the rest of the prompt.
        assert not os.path.exists(worker._donor_metadata_path("d1", stored.namespace))  # noqa: SLF001
    worker.get_finished({"d1"})

    with open(worker._donor_metadata_path("d1", stored.namespace), encoding="utf-8") as f:  # noqa: SLF001
        published = json.load(f)
    assert published["token_count"] == expected_capture


def test_gone_donor_on_non_span_load_reports_its_blocks_for_recompute(tmp_path) -> None:
    """A declined load must hand its destination blocks back to the engine.

    The scheduler already credited those tokens as computed, so a silent
    decline leaves the request decoding over whatever the pool held. vLLM
    drains `get_block_ids_with_load_errors` every forward and, under the
    recompute failure policy the quickstart ships, recomputes exactly those
    blocks. The span kind keeps its loud raise; this covers the other kinds.
    """
    torch = pytest.importorskip("torch")

    loading = _connector(tmp_path / "gone", KVConnectorRole.WORKER)
    loading.register_kv_caches({"layer0": torch.zeros(2, 12, 4, 2, 8)})
    # target_start=4 with token_count=8 over block_size 4 covers list
    # entries [1, 3): blocks 2 and 9. Block 7 is the untouched prefix.
    loading.bind_connector_metadata(
        SemBlendConnectorMetadata(
            loads=[
                PendingLoad(
                    request_id="r1",
                    donor_id="evicted",
                    token_count=8,
                    materialization_kind=MaterializationKind.REQUEST_ONLY,
                    namespace="ns",
                    block_ids=([7, 2, 9, 4],),
                    target_start=4,
                    donor_start=4,
                )
            ]
        )
    )

    loading.start_load_kv(FakeForwardContext(attn_metadata=object()))

    assert loading.stats_snapshot.get("load_declined_donor_gone", 0) == 1
    assert loading.get_block_ids_with_load_errors() == {2, 9}
    # Drained once reported, as the sync-loading contract requires.
    assert loading.get_block_ids_with_load_errors() == set()
    assert loading.stats_snapshot.get("load_error_blocks_reported", 0) == 2

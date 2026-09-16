"""The donor write is off the prefill critical path, and still ordered.

Measured in phase-0 (2026-09-14, stock vLLM 0.29, A10G): per captured request
the medians were capture_ms 14,535, copy_ms 119, write_ms 6,726 and copy_bytes
1,254 MB, and engine TTFT went from 6.1 s to 14.8 s at 32K-token prompts. The
device-to-host copy is not the cost; the safetensors write and the per-layer
metadata rewrite on the forward-pass thread are, so those move to a writer
thread and the metadata record is written once, when the donor is complete.

Deferring a write creates one hazard, and it is the whole point of these
tests: a donor must never be discoverable before its bytes are readable.
Registration with the provider happens only after the writer has landed every
layer and the metadata record for that donor, and a write that fails leaves a
decline with a reason rather than a silent absence.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from dataclasses import replace

import pytest
from test_connector_discovery import (
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeSchedulerOutput,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.capture_writer import LAYER, METADATA, CaptureWriter, WriteJob
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.namespace import namespace_for_request
from semblend_vllm_connector.types import PendingStore, SemBlendConnectorMetadata

LAYER_NAME = "model.layers.0.self_attn.attn"


def _config(tmp_path, **extra):
    cfg = {
        "mode": "semantic_span_experimental",
        "provider": "local",
        "min_prompt_tokens": 4,
        "min_semantic_span": 8,
        "kv_storage_path": str(tmp_path / "kv"),
        "audit_path": str(tmp_path / "audit.jsonl"),
    }
    cfg.update(extra)
    return FakeVllmConfig(FakeKvTransferConfig(cfg), cache_config=FakeCacheConfig(block_size=4))


def _events(tmp_path, name: str) -> list[dict]:
    path = tmp_path / "audit.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row["event"] == name]


def _store(request_id="d1", token_count=8, *, final=True, token_ids=None):
    return PendingStore(
        request_id=request_id,
        token_ids=list(range(token_count)) if token_ids is None else list(token_ids),
        token_count=token_count,
        namespace="ns",
        block_ids=([0, 1],),
        final=final,
    )


def _scheduled(request_id: str, prompt_tokens: int = 64, blocks: int | None = None, **attrs):
    """A request as the store builder sees it, with blocks already allocated.

    Shaped like vLLM's ``NewRequestData``, which is to say without a
    ``cache_salt`` field: the scheduler's capture hook never sees one.

    ``blocks`` short of the prompt leaves the capture open, which is what a
    request that finishes mid-prefill looks like.
    """
    allocated = prompt_tokens // 4 if blocks is None else blocks
    return types.SimpleNamespace(
        req_id=request_id,
        prompt_token_ids=list(range(prompt_tokens)),
        block_ids=([*range(allocated)],),
        **attrs,
    )


def _capture_namespace(connector, scheduled):
    """The namespace a capture opened for this scheduled request runs under."""
    return namespace_for_request(connector._config, connector._vllm_config, scheduled)  # noqa: SLF001


def _step(connector, *, new_reqs=(), finished=()):
    """One scheduler step, carrying whatever vLLM would carry in it."""
    return connector.build_connector_meta(
        types.SimpleNamespace(
            scheduled_new_reqs=list(new_reqs),
            finished_req_ids=list(finished),
        )
    )


def _publish_record(connector, request_id, namespace, token_count=8) -> None:
    """The record the worker writes once a donor's layers are durable.

    Under the namespace the *capture* was opened with, which is what the
    worker writes to and need not be the one a finished Request digests to.
    """
    os.makedirs(connector._donor_dir(request_id, namespace), exist_ok=True)  # noqa: SLF001
    with open(connector._donor_metadata_path(request_id, namespace), "w") as f:  # noqa: SLF001
        json.dump({"token_count": int(token_count)}, f)


def _layer_file(connector, request_id, namespace, name="layers.0.safetensors") -> str:
    """One layer's file, as a capture that never got published leaves it."""
    donor_dir = connector._donor_dir(request_id, namespace)  # noqa: SLF001
    os.makedirs(donor_dir, exist_ok=True)
    path = os.path.join(donor_dir, name)
    with open(path, "wb") as f:
        f.write(b"kv")
    return path


# ---------------------------------------------------------------------------
# The writer thread, and the join wait_for_save is documented to be
# ---------------------------------------------------------------------------


def test_a_donor_is_readable_after_wait_for_save_and_not_before(tmp_path, monkeypatch) -> None:
    """The layer write leaves the forward pass; the join is where it lands.

    ``save_kv_layer`` returns while the store is still writing, so the donor
    is not readable yet and must not be discoverable yet either -- its
    metadata record is what publishes it, and that record is submitted behind
    the layer job at the join.
    """
    torch = pytest.importorskip("torch")
    writing = threading.Event()
    release = threading.Event()

    def _save(tensors, filename):
        writing.set()
        assert release.wait(10), "writer never released"
        with open(filename, "wb") as f:
            f.write(b"kv")

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))

    connector.save_kv_layer(LAYER_NAME, layer, object())

    assert writing.wait(10), "the write never reached the writer thread"
    # The forward pass is back and the bytes are not on the store yet: this is
    # the window the ordering invariant exists for.
    assert not os.path.exists(connector._layer_filename("d1", "ns", LAYER_NAME))  # noqa: SLF001
    assert not connector._has_stored_donor("d1", "ns")  # noqa: SLF001
    assert connector._stored_donor_token_count("d1", "ns") == 0  # noqa: SLF001

    release.set()
    connector.wait_for_save()

    assert os.path.exists(connector._layer_filename("d1", "ns", LAYER_NAME))  # noqa: SLF001
    assert connector._has_stored_donor("d1", "ns")  # noqa: SLF001
    assert connector._stored_donor_token_count("d1", "ns") == 8  # noqa: SLF001
    connector.shutdown()


def test_shutdown_stops_the_writer_thread() -> None:
    """A connector's teardown drains the queue and joins the thread."""
    written: list[str] = []
    writer = CaptureWriter(
        write=lambda job: written.append(str(job.layer_name)),
        on_error=lambda job, exc: None,
        max_queue_depth=4,
    )
    writer.submit(WriteJob(request_id="d1", namespace="ns", kind=LAYER, layer_name="l0"))

    writer.close()

    assert written == ["l0"], "close dropped queued work instead of draining it"
    assert not writer.running
    # Idempotent: a connector shut down twice must not raise.
    writer.close()


def test_a_full_queue_blocks_the_submitter_and_drops_nothing() -> None:
    """A queue at its bound makes the caller wait; it never loses a layer.

    Dropping one would leave a donor whose record promises layers nobody
    wrote, which is the outcome the whole ordering exists to prevent.
    """
    started = threading.Event()
    release = threading.Event()
    written: list[str] = []
    blocked: list[str] = []

    def _write(job):
        if job.layer_name == "hold":
            started.set()
            assert release.wait(10), "writer never released"
        written.append(str(job.layer_name))

    writer = CaptureWriter(
        write=_write,
        on_error=lambda job, exc: None,
        max_queue_depth=1,
        on_blocked=lambda job, ms: blocked.append(str(job.layer_name)),
    )

    writer.submit(WriteJob(request_id="d1", namespace="ns", kind=LAYER, layer_name="hold"))
    assert started.wait(10), "the writer never picked up the first job"
    # One slot, and it is taken: the next submit has to wait for the writer.
    writer.submit(WriteJob(request_id="d1", namespace="ns", kind=LAYER, layer_name="l1"))
    threading.Timer(0.05, release.set).start()

    writer.submit(WriteJob(request_id="d1", namespace="ns", kind=LAYER, layer_name="l2"))

    writer.join()
    assert written == ["hold", "l1", "l2"]
    assert blocked == ["l2"], "the queue-full wait was not reported to the caller"
    writer.close()


def test_a_queue_full_wait_is_counted_against_the_request_that_paid_it(
    tmp_path, monkeypatch
) -> None:
    """The connector's own path: one row per request, and the cost on it."""
    torch = pytest.importorskip("torch")
    release = threading.Event()
    started = threading.Event()

    def _save(tensors, filename):
        started.set()
        release.wait(10)

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(
        _config(tmp_path, capture_write_queue_depth=1), KVConnectorRole.WORKER
    )
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store(final=False)]))

    connector.save_kv_layer("layers.0", layer, object())
    assert started.wait(10)
    connector.save_kv_layer("layers.1", layer, object())
    threading.Timer(0.05, release.set).start()
    connector.save_kv_layer("layers.2", layer, object())

    release.set()
    connector.wait_for_save()

    assert connector.stats_snapshot["capture_write_queue_blocked_total"] >= 1
    (row,) = _events(tmp_path, "capture_write_queue_full")
    assert row["request_id"] == "d1"
    assert row["queue_depth"] == 1
    connector.get_finished({"d1"})
    (cost,) = _events(tmp_path, "donor_capture_cost")
    assert cost["write_blocked_ms"] <= cost["write_ms"]
    connector.shutdown()


def test_the_metadata_record_is_written_once_per_donor_not_once_per_layer(tmp_path) -> None:
    """Two chunks of two layers used to rewrite the record four times.

    It is now written when the capture completes, which is also the point the
    donor becomes discoverable.
    """
    torch = pytest.importorskip("torch")
    connector = SemBlendVllmConnector(
        _config(tmp_path, kv_storage_backend="memory", kv_memory_max_donors=4),
        KVConnectorRole.WORKER,
    )
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})

    for token_count, final in ((4, False), (8, True)):
        connector.bind_connector_metadata(
            SemBlendConnectorMetadata(
                stores=[_store(token_count=token_count, final=final, token_ids=range(8))]
            )
        )
        connector.save_kv_layer("layers.0", layer, object())
        connector.save_kv_layer("layers.1", layer, object())
        connector.wait_for_save()
        if not final:
            # Half a donor is not a donor: nothing can match it yet.
            assert not connector._has_stored_donor("d1", "ns")  # noqa: SLF001

    stats = connector.stats_snapshot
    assert stats["capture_layer_writes_total"] == 4
    assert stats["capture_metadata_writes_total"] == 1, (
        "the record was rewritten per layer, which is the write this change removed"
    )
    assert connector._has_stored_donor("d1", "ns")  # noqa: SLF001
    assert connector._stored_donor_token_count("d1", "ns") == 8  # noqa: SLF001
    connector.shutdown()


def test_a_write_that_fails_after_the_request_finished_is_a_named_decline(
    tmp_path, monkeypatch
) -> None:
    """A donor that never landed leaves a row, not a silence.

    The failure happens on the writer thread, after the forward pass has moved
    on and possibly after the request has finished, so the row is written at
    the next point the forward thread owns -- here the worker's finish hook --
    and the donor is never published.
    """
    torch = pytest.importorskip("torch")

    def _save(tensors, filename):
        raise OSError("no space left on device")

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))

    connector.save_kv_layer(LAYER_NAME, layer, object())
    # No wait_for_save: the request finishes first, and the worker's own
    # finish hook is the drain.
    connector.get_finished({"d1"})

    (failure,) = _events(tmp_path, "donor_capture_write_failed")
    assert failure["request_id"] == "d1"
    assert failure["error_type"] == "OSError"
    assert connector.stats_snapshot["capture_write_errors_total"] == 1
    assert not connector._has_stored_donor("d1", "ns"), (  # noqa: SLF001
        "a donor whose write failed was still published"
    )
    connector.shutdown()


def test_a_layer_left_unreadable_restarts_the_capture_instead_of_raising(
    tmp_path, monkeypatch
) -> None:
    """The next chunk reads its own earlier chunk back. That read can fail.

    A write that failed part-way -- the store filling up inside ``save_file``
    -- leaves a file safetensors refuses to parse, and it raises its own error
    type for that rather than an OSError. This read happens inside
    ``save_kv_layer``, on the forward-pass thread, so an error escaping it
    takes the engine step with it. A base that cannot be read is a base that
    is not there: the capture restarts from 0.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    class _SafetensorError(Exception):
        """Stands in for safetensors' own error type, which is not an OSError."""

    def _load(filename):
        raise _SafetensorError("header too small")

    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})

    # The first chunk, landed and then left unreadable behind the connector's
    # back; the staging copy goes with it, as a later step would have dropped
    # it once the write completed.
    connector.bind_connector_metadata(
        SemBlendConnectorMetadata(stores=[_store(token_count=4, final=False, token_ids=range(8))])
    )
    connector.save_kv_layer("layers.0", layer, object())
    connector.wait_for_save()
    connector._inflight_layers.clear()  # noqa: SLF001

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.load_file = _load
    fake_st_torch.save_file = lambda tensors, filename: open(filename, "wb").write(b"kv")
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector.bind_connector_metadata(
        SemBlendConnectorMetadata(stores=[_store(token_count=8, final=True, token_ids=range(8))])
    )
    connector.save_kv_layer("layers.0", layer, object())
    connector.wait_for_save()

    assert connector.stats_snapshot["layer_capture_base_missing"] == 1, (
        "the unreadable base was not treated as a missing one"
    )
    assert connector._capture_progress["d1"] == {"layers.0": 8}  # noqa: SLF001
    connector.shutdown()


def test_a_capture_whose_write_failed_does_not_read_its_own_base_back(tmp_path) -> None:
    """A failed write is a reason not to trust what is on the store.

    ``save_file`` failing part-way leaves a file with this donor's name on it
    and not this donor's bytes in it, so the next chunk must not append to it.
    The restart costs the earlier chunk; reading it back risks a capture that
    is silently a mixture.
    """
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    store = _store()
    storage_key = connector._storage_key("d1", "ns")  # noqa: SLF001
    connector._inflight_layers[(storage_key, "layers.0")] = "staged"  # noqa: SLF001
    assert connector._captured_layer(store, "layers.0") == "staged"  # noqa: SLF001

    connector._capture_write_failed.add("d1")  # noqa: SLF001

    assert connector._captured_layer(store, "layers.0") is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# The ordering invariant, at the registration end
# ---------------------------------------------------------------------------


def test_a_donor_is_not_registered_before_its_capture_is_durable(tmp_path) -> None:
    """Registration is what makes a donor matchable, so it goes last.

    The scheduler role finishes the request in another process from the one
    writing its KV. It reads the same record the writer publishes, and until
    that record exists the donor is declined with a reason rather than indexed
    and hoped for.
    """
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.SCHEDULER)

    metadata = connector.build_connector_meta(
        FakeSchedulerOutput(scheduled_new_reqs=[_scheduled("d1")])
    )
    assert [store.request_id for store in metadata.stores] == ["d1"]

    finished = FakeRequest("d1", list(range(64)))
    connector.request_finished(finished, [0])

    assert "d1" not in connector._provider._donors  # noqa: SLF001
    (held,) = _events(tmp_path, "donor_registration_deferred")
    assert held["reason"] == "capture_not_durable"
    assert connector.stats_snapshot["donor_registration_deferred_total"] == 1

    # The same request with its capture landed: registered, no decline.
    second = connector.build_connector_meta(
        FakeSchedulerOutput(scheduled_new_reqs=[_scheduled("d2")])
    )
    _publish_record(connector, "d2", second.stores[0].namespace)
    connector.request_finished(FakeRequest("d2", list(range(64))), [0])

    assert "d2" in connector._provider._donors  # noqa: SLF001
    assert len(_events(tmp_path, "donor_registered")) == 1


# ---------------------------------------------------------------------------
# Capture policy
# ---------------------------------------------------------------------------


def _captured_ids(connector, requests) -> list[str]:
    metadata = connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=requests))
    return [store.request_id for store in metadata.stores]


def test_capture_policy_all_is_the_default_and_is_todays_selection(tmp_path) -> None:
    """ "all" decides nothing, so the donors a run produces do not move.

    The phase-0 manifests predict which requests become donors from this
    behaviour; a default that changed the set would invalidate them.
    """
    from semblend_vllm_connector.config import SemBlendVllmConfig

    assert SemBlendVllmConfig().capture_policy == "all"
    assert SemBlendVllmConfig().capture_sample_rate == 1.0

    requests = [_scheduled(f"r{i}") for i in range(8)]
    requests.append(types.SimpleNamespace(req_id="short", prompt_token_ids=[1], block_ids=([0],)))

    default_arm = SemBlendVllmConnector(_config(tmp_path / "a"), KVConnectorRole.SCHEDULER)
    explicit_arm = SemBlendVllmConnector(
        _config(tmp_path / "b", capture_policy="all"), KVConnectorRole.SCHEDULER
    )

    captured_by_default = _captured_ids(default_arm, requests)
    captured_explicitly = _captured_ids(explicit_arm, requests)

    assert captured_by_default == [f"r{i}" for i in range(8)]
    assert captured_explicitly == captured_by_default
    # The only skip is the length gate that predates the policy.
    reasons = {row["reason"] for row in _events(tmp_path / "a", "capture_skipped")}
    assert reasons == {"short_prompt"}


def test_sampled_captures_the_same_fraction_on_a_rerun(tmp_path) -> None:
    """The sampled set is keyed on the request id, not on arrival order."""
    requests = [_scheduled(f"r{i}") for i in range(40)]

    first = SemBlendVllmConnector(
        _config(tmp_path / "a", capture_policy="sampled", capture_sample_rate=0.5),
        KVConnectorRole.SCHEDULER,
    )
    second = SemBlendVllmConnector(
        _config(tmp_path / "b", capture_policy="sampled", capture_sample_rate=0.5),
        KVConnectorRole.SCHEDULER,
    )

    captured = _captured_ids(first, requests)
    rerun = _captured_ids(second, list(reversed(requests)))

    assert set(captured) == set(rerun), "the sampled set moved between two runs"
    assert 0 < len(captured) < len(requests)
    skipped = _events(tmp_path / "a", "capture_skipped")
    assert {row["reason"] for row in skipped} == {"not_sampled"}
    assert len(skipped) == len(requests) - len(captured)
    assert skipped[0]["capture_policy"] == "sampled"
    assert skipped[0]["capture_sample_rate"] == 0.5


def test_hinted_captures_only_the_requests_the_caller_marked(tmp_path) -> None:
    """A router marks the donors; everything else pays nothing for capture."""
    marked_extra_args = _scheduled(
        "hinted-xargs",
        sampling_params=types.SimpleNamespace(extra_args={"semblend_capture": True}),
    )
    marked_header = _scheduled("hinted-header", metadata={"x-semblend-capture": "1"})
    unmarked = _scheduled("plain")
    explicitly_off = _scheduled(
        "off", sampling_params=types.SimpleNamespace(extra_args={"semblend_capture": False})
    )

    connector = SemBlendVllmConnector(
        _config(tmp_path, capture_policy="hinted"), KVConnectorRole.SCHEDULER
    )

    captured = _captured_ids(
        connector, [marked_extra_args, marked_header, unmarked, explicitly_off]
    )

    assert captured == ["hinted-xargs", "hinted-header"]
    skipped = _events(tmp_path, "capture_skipped")
    assert [row["request_id"] for row in skipped] == ["plain", "off"]
    assert {row["reason"] for row in skipped} == {"not_hinted"}


# ---------------------------------------------------------------------------
# Teardown is bounded: a wedged store must not be able to hang a process
# ---------------------------------------------------------------------------


def test_close_returns_within_its_timeout_when_the_queue_is_full() -> None:
    """The stop signal is inside the deadline, not ahead of it.

    A store that has stopped draining leaves the writer blocked in a write and
    the queue at its bound, which is exactly the case where the sentinel
    cannot be enqueued either. Putting it unbounded ahead of the join means
    ``close`` never reaches its own timeout: the connector's ``shutdown``, and
    the interpreter's exit hook behind it, hang on a store nobody can fix.
    """
    wedged = threading.Event()
    release = threading.Event()

    def _write(job):
        wedged.set()
        assert release.wait(30), "the writer was never released"

    writer = CaptureWriter(
        write=_write,
        on_error=lambda job, exc: None,
        max_queue_depth=1,
    )
    writer.submit(WriteJob(request_id="d1", namespace="ns", kind=LAYER, layer_name="wedge"))
    assert wedged.wait(10), "the writer never picked the first job up"
    # One slot, and it is taken: the queue is now full and stays full.
    writer.submit(WriteJob(request_id="d1", namespace="ns", kind=LAYER, layer_name="l1"))

    returned = threading.Event()

    def _close():
        writer.close(timeout=0.5)
        returned.set()

    closer = threading.Thread(target=_close, daemon=True)
    closer.start()
    try:
        assert returned.wait(10), "close() did not return inside its own timeout"
    finally:
        release.set()
        closer.join(10)


def test_shutdown_returns_when_the_store_is_wedged_and_the_queue_is_full(
    tmp_path, monkeypatch
) -> None:
    """The connector's own teardown inherits that bound, from config."""
    torch = pytest.importorskip("torch")
    wedged = threading.Event()
    release = threading.Event()

    def _save(tensors, filename):
        wedged.set()
        assert release.wait(30), "the writer was never released"

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(
        _config(tmp_path, capture_write_queue_depth=1, capture_write_close_timeout_s=0.5),
        KVConnectorRole.WORKER,
    )
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store(final=False)]))

    connector.save_kv_layer("layers.0", layer, object())
    assert wedged.wait(10), "the writer never picked the first layer up"
    connector.save_kv_layer("layers.1", layer, object())

    returned = threading.Event()
    done = threading.Thread(target=lambda: (connector.shutdown(), returned.set()), daemon=True)
    done.start()
    try:
        assert returned.wait(10), "shutdown() hung on a wedged store"
    finally:
        release.set()
        done.join(10)


def test_wait_for_save_after_a_wedged_shutdown_returns(tmp_path, monkeypatch) -> None:
    """Teardown abandons jobs. The join that follows must not wait on them.

    vLLM calls ``wait_for_save`` at the end of every step that called
    ``save_kv_layer``, shutdown or no shutdown, and the queue's own counter
    still holds whatever a wedged store made ``close`` abandon. Joining that
    hangs the engine step -- a worse outcome than the write failure it is
    standing in for, and one this file's earlier teardown tests all stopped
    just short of.
    """
    torch = pytest.importorskip("torch")
    wedged = threading.Event()
    release = threading.Event()

    def _save(tensors, filename):
        wedged.set()
        assert release.wait(30), "the writer was never released"

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(
        _config(tmp_path, capture_write_queue_depth=1, capture_write_close_timeout_s=0.5),
        KVConnectorRole.WORKER,
    )
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store(final=False)]))

    try:
        connector.save_kv_layer("layers.0", layer, object())
        assert wedged.wait(10), "the writer never picked the first layer up"
        # Queued, and abandoned by the teardown below: nothing will run it.
        connector.save_kv_layer("layers.1", layer, object())
        connector.shutdown()

        connector.bind_connector_metadata(
            SemBlendConnectorMetadata(stores=[_store(request_id="d2", token_ids=range(8))])
        )
        connector.save_kv_layer("layers.0", layer, object())
        returned = threading.Event()
        step = threading.Thread(
            target=lambda: (connector.wait_for_save(), returned.set()), daemon=True
        )
        step.start()
        assert returned.wait(10), "wait_for_save hung on writes teardown had abandoned"
        step.join(10)
    finally:
        release.set()

    # The abandoned write is a counted failure, not a silence.
    assert connector.stats_snapshot["capture_write_errors_total"] >= 1
    assert "d1" in {row["request_id"] for row in _events(tmp_path, "donor_capture_write_failed")}


def test_teardown_keeps_drain_time_for_the_writer_when_a_record_is_pending(
    tmp_path, monkeypatch
) -> None:
    """Queueing the records must not eat the whole close timeout.

    The two halves of teardown are different jobs and only the second one is
    what the timeout is named after. A record queued onto a full queue can
    spend the entire budget waiting for room, leaving the drain none of it --
    so a store that is merely slow gets abandoned as though it had wedged,
    which is the opposite of the documented behaviour.
    """
    wedged = threading.Event()
    release = threading.Event()

    def _save(tensors, filename):
        wedged.set()
        assert release.wait(30), "the writer was never released"

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    budget = 1.0
    connector = SemBlendVllmConnector(
        _config(tmp_path, capture_write_queue_depth=1, capture_write_close_timeout_s=budget),
        KVConnectorRole.WORKER,
    )
    job = WriteJob(
        request_id="d1", namespace="ns", kind=LAYER, layer_name="layers.0", token_count=8
    )
    try:
        connector._submit_write(job)  # noqa: SLF001
        assert wedged.wait(10), "the writer never picked the first job up"
        # The one queue slot, taken: every later put waits for the deadline.
        connector._submit_write(replace(job, layer_name="layers.1"))  # noqa: SLF001

        # A donor whose record teardown will try to queue onto that full queue.
        connector._capture_namespaces["d1"] = "ns"  # noqa: SLF001
        connector._capture_progress["d1"] = {"layers.0": 8}  # noqa: SLF001
        connector._finalize_pending.add("d1")  # noqa: SLF001

        given: list[float | None] = []
        writer = connector._writer  # noqa: SLF001
        closing = writer.close

        def _close(timeout=None):
            given.append(timeout)
            closing(timeout=timeout)

        monkeypatch.setattr(writer, "close", _close)

        started = time.monotonic()
        connector.shutdown()
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert given and given[0] >= budget / 2.0, (
        f"the record's submission left the drain {given}s of a {budget}s budget"
    )
    assert elapsed < 4 * budget, "teardown spent more than its own budget"


def test_shutdown_publishes_the_record_of_a_donor_whose_layers_all_landed(tmp_path) -> None:
    """Teardown submits the record, rather than assuming it is already queued.

    Every tensor durable and no record is the worst of both: the bytes are on
    a tier with no eviction and nothing can ever find them.
    """
    torch = pytest.importorskip("torch")
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))

    connector.save_kv_layer("layers.0", layer, object())
    connector.save_kv_layer("layers.1", layer, object())
    # No wait_for_save and no get_finished: the engine is torn down here.
    connector.shutdown()

    assert connector._has_stored_donor("d1", "ns"), (  # noqa: SLF001
        "the donor's layers are durable and its record was never submitted"
    )


def test_a_save_after_shutdown_is_a_named_decline_not_a_restarted_writer(tmp_path) -> None:
    """Shutdown stops capture. A silent restart leaks a thread per teardown."""
    torch = pytest.importorskip("torch")
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))
    connector.save_kv_layer("layers.0", layer, object())
    connector.shutdown()

    connector.bind_connector_metadata(
        SemBlendConnectorMetadata(stores=[_store(request_id="d2", token_ids=range(8))])
    )
    connector.save_kv_layer("layers.0", layer, object())
    connector.wait_for_save()

    assert not connector._has_stored_donor("d2", "ns")  # noqa: SLF001
    assert connector.stats_snapshot["capture_write_errors_total"] >= 1
    assert [row["request_id"] for row in _events(tmp_path, "donor_capture_write_failed")] == ["d2"]
    assert [
        thread for thread in threading.enumerate() if thread.name.startswith("semblend-capture")
    ] == [], "shutdown was followed by a silently restarted writer thread"


# ---------------------------------------------------------------------------
# What the write costs after it moves
# ---------------------------------------------------------------------------


def test_the_join_that_finishes_the_write_is_measured(tmp_path, monkeypatch) -> None:
    """``write_ms`` alone understates exactly the cost this change moves.

    vLLM calls ``wait_for_save`` inside the same ``execute_model`` as the save
    hooks, so the part of the write the forward pass did not overlap away is
    still on the critical path. An operator re-running phase-0 would otherwise
    see ``write_ms`` collapse to ~0 and read a partly-faster step as a
    fully-eliminated cost.
    """
    torch = pytest.importorskip("torch")

    def _save(tensors, filename):
        time.sleep(0.05)
        with open(filename, "wb") as f:
            f.write(b"kv")

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))

    connector.save_kv_layer("layers.0", layer, object())
    connector.save_kv_layer("layers.1", layer, object())
    connector.wait_for_save()
    connector.get_finished({"d1"})

    (cost,) = _events(tmp_path, "donor_capture_cost")
    assert cost["flush_ms"] >= 40, (
        "the unoverlapped remainder of the write is on the forward thread and "
        f"nothing measured it: {cost}"
    )
    assert cost["write_ms"] < cost["flush_ms"], "write_ms is not the whole cost any more"
    assert connector.stats_snapshot["capture_flush_ms_total"] >= 40
    connector.shutdown()


# ---------------------------------------------------------------------------
# The gate has to read the namespace the capture was actually written under
# ---------------------------------------------------------------------------


def test_a_salted_request_registers_against_its_capture_namespace(tmp_path) -> None:
    """The two hooks are handed different request types, and they differ.

    vLLM's ``NewRequestData`` -- what the capture hook sees -- has no
    ``cache_salt`` field, while the ``Request`` the finish hook sees does, and
    the namespace digest folds the salt in. A gate that recomputes the
    namespace at finish stats a directory the worker will never write to, so
    on any tenant-salted deployment every donor is declined forever while its
    tensors and its record sit on disk under the other digest.
    """
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.SCHEDULER)

    scheduled = _scheduled("d1")
    metadata = _step(connector, new_reqs=[scheduled])
    capture_namespace = _capture_namespace(connector, scheduled)
    assert metadata.stores[0].namespace == capture_namespace
    # The worker lands the capture where the capture hook addressed it.
    _publish_record(connector, "d1", capture_namespace, metadata.stores[0].token_count)

    finished = FakeRequest("d1", list(range(64)), cache_salt="tenant-acme")
    assert namespace_for_request(connector._config, connector._vllm_config, finished) != (  # noqa: SLF001
        capture_namespace
    ), "this test proves nothing unless the two namespaces actually differ"
    connector.request_finished(finished, [0])

    assert "d1" in connector._provider._donors, (  # noqa: SLF001
        "a salted request's donor was declined although its capture is on disk"
    )
    assert _events(tmp_path, "donor_registration_skipped") == []


# ---------------------------------------------------------------------------
# A capture still open at finish closes one step later, and vLLM says so
# ---------------------------------------------------------------------------


def test_a_donor_that_finishes_mid_capture_registers_when_its_record_lands(tmp_path) -> None:
    """The finalize handoff is structurally one step behind the gate.

    vLLM's ``Scheduler._free_request`` calls ``request_finished`` and only then
    adds the id to ``finished_req_ids``, which ships in the next
    ``SchedulerOutput`` -- so the worker publishes the record after the gate
    has already run. Declining there loses a donor whose bytes do land, on a
    tier that never evicts them.
    """
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.SCHEDULER)

    # Eight blocks for a 64-token prompt: the capture stops at 32 and stays open.
    scheduled = _scheduled("d1", blocks=8)
    metadata = _step(connector, new_reqs=[scheduled])
    assert [store.final for store in metadata.stores] == [False]
    capture_namespace = _capture_namespace(connector, scheduled)

    connector.request_finished(FakeRequest("d1", list(range(64))), [0])
    assert "d1" not in connector._provider._donors  # noqa: SLF001

    # The step that carries the finish to the worker. Its own finish hook
    # publishes the shorter donor it holds, in the other process.
    _step(connector, finished=["d1"])
    _publish_record(connector, "d1", capture_namespace, 32)

    _step(connector)

    assert "d1" in connector._provider._donors, (  # noqa: SLF001
        "the donor's record landed and it was never registered"
    )
    (registered,) = _events(tmp_path, "donor_registered")
    assert registered["deferred_steps"] >= 1
    assert _events(tmp_path, "donor_registration_skipped") == []


def test_a_batch_of_finishes_does_not_spend_a_held_donors_budget(tmp_path) -> None:
    """The budget is scheduler steps, and a batch of finishes is not steps.

    vLLM frees every request that finished in a step one after another, each
    call landing in ``request_finished``, and the worker gets no step to
    publish in until the next one. Aging the held donors once per call spends
    a donor's whole budget inside the step it finished in, so a batch larger
    than the budget declines donors whose captures are complete and readable
    -- the default capture policy quietly stops being today's selection at
    batch sizes an engine reaches routinely.
    """
    admitted = [f"d{index}" for index in range(12)]
    assert len(admitted) > 8, "the default budget is 8; a smaller batch proves nothing"
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.SCHEDULER)

    # One step admits them all, with their captures left open.
    scheduled = [_scheduled(request_id, blocks=8) for request_id in admitted]
    _step(connector, new_reqs=scheduled)
    namespaces = {request.req_id: _capture_namespace(connector, request) for request in scheduled}

    # One step's worth of finishes: vLLM frees them all before the next step.
    for request_id in admitted:
        connector.request_finished(FakeRequest(request_id, list(range(64))), [0])
    assert sorted(connector._deferred_registrations) == sorted(admitted)  # noqa: SLF001
    assert _events(tmp_path, "donor_registration_skipped") == [], (
        "a donor was declined inside the step it finished in, before the worker "
        "had a step to publish its record in"
    )

    # The step that carries those ids to the worker, which then publishes.
    _step(connector, finished=admitted)
    for request_id in admitted:
        _publish_record(connector, request_id, namespaces[request_id], 32)

    _step(connector)

    assert sorted(connector._provider._donors) == sorted(admitted)  # noqa: SLF001
    assert _events(tmp_path, "donor_registration_skipped") == []
    assert {row["deferred_steps"] for row in _events(tmp_path, "donor_registered")} == {2}


def test_a_capture_that_never_lands_is_declined_without_touching_its_files(tmp_path) -> None:
    """The scheduler role declines. It does not delete.

    Deleting here was a corruption bug, not a cleanup: the files belong to the
    worker, which is another process with a write queue this role cannot see,
    so "no record" on this side means "no record that this role can see yet"
    and never "no record is coming". The worker deletes what it never
    published, once its own writer has drained.
    """
    connector = SemBlendVllmConnector(
        _config(tmp_path, donor_registration_retry_steps=2), KVConnectorRole.SCHEDULER
    )

    scheduled = _scheduled("d1", blocks=8)
    _step(connector, new_reqs=[scheduled])
    capture_namespace = _capture_namespace(connector, scheduled)
    # Layers on disk, record not written yet: a capture in flight looks
    # exactly like one that failed, from here.
    layer = _layer_file(connector, "d1", capture_namespace)

    connector.request_finished(FakeRequest("d1", list(range(64))), [0])
    for _ in range(3):
        _step(connector)

    (skipped,) = _events(tmp_path, "donor_registration_skipped")
    assert skipped["reason"] == "capture_not_durable"
    assert skipped["steps_waited"] == 2
    assert "d1" not in connector._provider._donors  # noqa: SLF001
    assert os.path.exists(layer), (
        "the scheduler role deleted a capture whose writer may still hold jobs for it"
    )
    assert connector.stats_snapshot.get("capture_orphans_discarded_total", 0) == 0


def test_the_worker_deletes_the_capture_it_never_published(tmp_path, monkeypatch) -> None:
    """A capture with no record must not also be a permanent disk leak.

    The disk tier has no eviction at all, so layer files with no record are
    unreferenced, unmatchable and there for the life of the volume. The worker
    is the role that can say so: its finish hook joins the writer first, so
    past that join "no record" is final.
    """
    torch = pytest.importorskip("torch")

    def _save(tensors, filename):
        if filename.endswith("layers.1.safetensors"):
            raise OSError("no space left on device")
        with open(filename, "wb") as f:
            f.write(b"kv")

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))

    connector.save_kv_layer("layers.0", layer, object())
    connector.save_kv_layer("layers.1", layer, object())
    connector.wait_for_save()
    # One layer landed and the other did not, so the donor is never published
    # and the layer that did land is unreachable.
    orphan = os.path.join(connector._donor_dir("d1", "ns"), "layers.0.safetensors")  # noqa: SLF001
    assert os.path.exists(orphan)

    connector.get_finished({"d1"})

    assert not connector._has_stored_donor("d1", "ns")  # noqa: SLF001
    assert not os.path.exists(orphan), "the unpublished capture's bytes were left on the volume"
    assert connector.stats_snapshot["capture_orphans_discarded_total"] == 1
    (discarded,) = _events(tmp_path, "capture_orphan_discarded")
    assert discarded["files"] == 1
    connector.shutdown()


def test_a_declined_donors_files_survive_a_writer_that_still_holds_them(
    tmp_path, monkeypatch
) -> None:
    """The two roles, in the order vLLM runs them, with the writer backed up.

    The scheduler's budget can run out while the worker's writer is still
    landing that donor's layers. Deleting the layers there and letting the
    writer publish the record afterwards leaves a donor whose record promises
    tensors that are gone -- matched by a later request, and a loud raise
    inside its ``execute_model``. Whatever the scheduler decides, a published
    record and the layers it names go together.
    """
    torch = pytest.importorskip("torch")
    wedged = threading.Event()
    release = threading.Event()

    def _save(tensors, filename):
        # The layer lands, and the writer then stays inside this call: on
        # disk, owned by a job that has not finished.
        with open(filename, "wb") as f:
            f.write(b"kv")
        if not wedged.is_set():
            wedged.set()
            assert release.wait(30), "the writer was never released"

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    scheduler = SemBlendVllmConnector(
        _config(tmp_path, donor_registration_retry_steps=1), KVConnectorRole.SCHEDULER
    )
    scheduled = _scheduled("d1", blocks=8)
    _step(scheduler, new_reqs=[scheduled])
    capture_namespace = _capture_namespace(scheduler, scheduled)

    worker = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    layer = torch.randn(2, 6, 4, 2, 16)
    worker.register_kv_caches({LAYER_NAME: layer})
    worker.bind_connector_metadata(
        SemBlendConnectorMetadata(
            stores=[
                PendingStore(
                    request_id="d1",
                    token_ids=list(range(8)),
                    token_count=8,
                    namespace=capture_namespace,
                    block_ids=([0, 1],),
                    final=True,
                )
            ]
        )
    )
    try:
        worker.save_kv_layer("layers.0", layer, object())
        assert wedged.wait(10), "the writer never picked the first layer up"
        # Queued behind the layer the writer is holding.
        worker.save_kv_layer("layers.1", layer, object())

        # The scheduler gives up on this donor while all of that is in flight.
        scheduler.request_finished(FakeRequest("d1", list(range(64))), [0])
        for _ in range(2):
            _step(scheduler)
        (skipped,) = _events(tmp_path, "donor_registration_skipped")
        assert skipped["reason"] == "capture_not_durable"
    finally:
        release.set()
    worker.wait_for_save()

    donor_dir = worker._donor_dir("d1", capture_namespace)  # noqa: SLF001
    names = set(os.listdir(donor_dir))
    if "metadata.json" in names:
        assert names == {"metadata.json", "layers.0.safetensors", "layers.1.safetensors"}, (
            "a donor record was published over layers that are no longer on the store"
        )
    else:
        assert not worker._has_stored_donor("d1", capture_namespace)  # noqa: SLF001
    worker.shutdown()


def test_a_record_is_not_published_over_a_layer_that_is_gone(tmp_path) -> None:
    """The publish rechecks the layers it is about to vouch for.

    Nothing inside this process should be able to take them -- a capture
    belongs to the writer until it has drained -- so this is the backstop for
    everything that is not this process. A record over a hole is worse than no
    record: it is matchable, and the load fails inside a forward pass.
    """
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    donor_dir = connector._donor_dir("d1", "ns")  # noqa: SLF001
    os.makedirs(donor_dir, exist_ok=True)
    with open(os.path.join(donor_dir, "layers.0.safetensors"), "wb") as f:
        f.write(b"kv")

    with pytest.raises(FileNotFoundError):
        connector._write_donor_metadata(  # noqa: SLF001
            WriteJob(
                request_id="d1",
                namespace="ns",
                kind=METADATA,
                token_count=8,
                layer_names=("layers.0", "layers.1"),
            )
        )

    assert not connector._has_stored_donor("d1", "ns")  # noqa: SLF001


def test_shutdown_returns_when_a_record_is_pending_and_the_queue_is_full(
    tmp_path, monkeypatch
) -> None:
    """The record's own submission is inside teardown's deadline too.

    Teardown queues the records of donors whose layers all landed, and a
    wedged store leaves the queue full, so an unbounded wait for room there
    hangs the process exactly as an unbounded join would. The record is a
    failed write instead, which the donor's decline already has a name for.
    """
    torch = pytest.importorskip("torch")
    wedged = threading.Event()
    release = threading.Event()

    def _save(tensors, filename):
        wedged.set()
        assert release.wait(30), "the writer was never released"

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(
        _config(tmp_path, capture_write_queue_depth=1, capture_write_close_timeout_s=0.5),
        KVConnectorRole.WORKER,
    )
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    # final: this store completes the donor, so a metadata record is pending.
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))

    connector.save_kv_layer("layers.0", layer, object())
    assert wedged.wait(10), "the writer never picked the first layer up"
    connector.save_kv_layer("layers.1", layer, object())
    assert connector._finalize_pending == {"d1"}  # noqa: SLF001

    returned = threading.Event()
    done = threading.Thread(target=lambda: (connector.shutdown(), returned.set()), daemon=True)
    done.start()
    try:
        assert returned.wait(10), "shutdown() hung queueing a record onto a full queue"
    finally:
        release.set()
        done.join(10)


def test_a_donor_still_held_at_teardown_gets_one_last_chance(tmp_path) -> None:
    """Teardown is the last hook, so it is the last retry.

    A donor deferred on the final step of a drain would otherwise sit in the
    held set until the process ends: its record landed, nothing looked again.
    """
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.SCHEDULER)

    scheduled = _scheduled("d1", blocks=8)
    _step(connector, new_reqs=[scheduled])
    connector.request_finished(FakeRequest("d1", list(range(64))), [0])
    assert "d1" not in connector._provider._donors  # noqa: SLF001

    # The worker publishes after the engine's last scheduler step.
    _publish_record(connector, "d1", _capture_namespace(connector, scheduled), 32)
    connector.shutdown()

    assert "d1" in connector._provider._donors  # noqa: SLF001


def test_teardown_declines_a_donor_whose_record_never_came(tmp_path) -> None:
    """The last pass is terminal: nothing runs after it to try again."""
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.SCHEDULER)

    scheduled = _scheduled("d1", blocks=8)
    _step(connector, new_reqs=[scheduled])
    orphan = _layer_file(connector, "d1", _capture_namespace(connector, scheduled))
    connector.request_finished(FakeRequest("d1", list(range(64))), [0])

    connector.shutdown()

    (skipped,) = _events(tmp_path, "donor_registration_skipped")
    assert skipped["reason"] == "capture_not_durable"
    assert connector._deferred_registrations == {}  # noqa: SLF001
    # Still the worker's to delete: this role never owned those bytes.
    assert os.path.exists(orphan)

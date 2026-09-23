"""An in-flight host copy is waited on by whoever reads it, never by the forward pass.

Phase-0 b5 (A10G, 1,250 requests): every unserved request paid ~26 ms of
blocking device-to-host copy per capture, plus a host-device sync per layer
from building the slot index out of pageable memory. The copy now runs on a
side stream into pinned memory and completes on an event; these tests pin the
readers of that memory to the event, which is the ordering that makes the
deferral safe. The CUDA path itself is exercised on the GPU campaign.
"""

from __future__ import annotations

import sys
import types

import torch
from test_deferred_capture_write import LAYER_NAME, _config, _store

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import SemBlendConnectorMetadata


class _Event:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.done = False

    def synchronize(self) -> None:
        self.log.append("synchronize")
        self.done = True


def _patched_connector(tmp_path, monkeypatch, log: list[str], events: list[_Event]):
    def _save(tensors, filename):
        log.append("write")
        assert events and all(e.done for e in events), "wrote bytes still in flight"
        with open(filename, "wb") as f:
            f.write(b"kv")

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = _save
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)

    def _copy(kv_cache):
        event = _Event(log)
        events.append(event)
        log.append("copy_started")
        return kv_cache.detach().contiguous().cpu(), event

    monkeypatch.setattr(connector, "_copy_to_host", _copy)
    return connector


def test_the_writer_waits_for_the_copy_and_the_forward_pass_does_not(tmp_path, monkeypatch):
    log: list[str] = []
    events: list[_Event] = []
    connector = _patched_connector(tmp_path, monkeypatch, log, events)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[_store()]))

    connector.save_kv_layer(LAYER_NAME, layer, object())
    connector.wait_for_save()

    assert log.index("synchronize") < log.index("write")
    assert connector._has_stored_donor("d1", "ns")  # noqa: SLF001
    connector.shutdown()


def test_a_continuation_chunk_waits_before_appending(tmp_path, monkeypatch):
    log: list[str] = []
    events: list[_Event] = []
    connector = _patched_connector(tmp_path, monkeypatch, log, events)
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({LAYER_NAME: layer})
    # Keep the first chunk staged: the writer is not allowed to run yet.
    writer = connector._ensure_writer()  # noqa: SLF001
    gate = __import__("threading").Event()
    original = writer._write  # noqa: SLF001

    def _held(job):
        gate.wait(10)
        original(job)

    writer._write = _held  # noqa: SLF001
    first = _store(token_count=4, final=False)
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[first]))
    connector.save_kv_layer(LAYER_NAME, layer, object())
    assert events[0].done is False

    second = _store(token_count=8, final=True)
    connector.bind_connector_metadata(SemBlendConnectorMetadata(stores=[second]))
    connector.save_kv_layer(LAYER_NAME, layer, object())
    # The first chunk's copy was read back to append to: it had to land first,
    # and the second chunk's own copy was waited on before the concat.
    assert events[0].done and events[1].done
    gate.set()
    connector.wait_for_save()
    connector.shutdown()


def test_cpu_tensors_and_the_off_switch_copy_synchronously(tmp_path):
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.WORKER)
    host, ready = connector._copy_to_host(torch.ones(2, 3))  # noqa: SLF001
    assert ready is None and host.device.type == "cpu"
    off = SemBlendVllmConnector(_config(tmp_path, capture_async_copy=False), KVConnectorRole.WORKER)
    assert off._config.capture_async_copy is False  # noqa: SLF001
    connector.shutdown()
    off.shutdown()


def test_index_tensor_matches_the_old_construction_on_cpu():
    values = [3, 1, 4, 1, 5]
    built = SemBlendVllmConnector._index_tensor(values, torch.device("cpu"))
    assert built.dtype == torch.int64
    assert built.tolist() == values

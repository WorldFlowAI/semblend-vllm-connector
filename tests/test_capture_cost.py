"""What donor capture costs the request that pays for it, on the record.

Measured in phase-0 (2026-09-14, stock vLLM 0.29): with capture on for every
request, median TTFT on the connector arm was 14.80 s against 6.20 s on stock
vLLM with prefix caching -- +8.6 s on every request, served or not -- while the
same configuration at 3.7K-token prompts cost +0.17 s. The copy scales with the
prompt because it is the prompt's whole KV: ~56 KiB per token for
Qwen2.5-7B-Instruct at fp16, so ~1.2 GB at the run's median 21.3K tokens.

The copy and the write both happen inside ``save_kv_layer``, which vLLM calls
from the forward pass, so the cost is not inferable from anything else in the
audit. These tests pin the instrumentation that makes it readable.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from test_connector_discovery import (
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeSchedulerOutput,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import PendingStore, SemBlendConnectorMetadata


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
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row["event"] == name]


def test_capture_cost_is_reported_when_the_worker_sees_the_request_finish(tmp_path, monkeypatch):
    """One row per captured donor, carrying the bytes and the wall time.

    The event is the worker's, because the copy is: the scheduler-role
    connector that writes ``donor_registered`` is a different instance in a
    different process and has never seen a tensor.
    """
    torch = pytest.importorskip("torch")
    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(
        _config(tmp_path, kv_storage_backend="memory", kv_memory_max_donors=4),
        KVConnectorRole.WORKER,
    )
    layer_name = "model.layers.0.self_attn.attn"
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({layer_name: layer})
    store = PendingStore(
        request_id="d1",
        token_ids=list(range(8)),
        token_count=8,
        namespace="ns",
        block_ids=([0, 1],),
    )
    connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[], stores=[store]))
    connector.save_kv_layer(layer_name, layer, object())

    # Nothing is written until the request finishes: a model has one of these
    # per layer per prefill chunk.
    assert _events(tmp_path, "donor_capture_cost") == []

    connector.get_finished({"d1"})

    (cost,) = _events(tmp_path, "donor_capture_cost")
    # 8 tokens x 2 (K,V) x 2 heads x 16 dims x 4 bytes.
    assert cost["copy_bytes"] == 2 * 8 * 2 * 16 * 4
    assert cost["layers"] == 1
    assert cost["store_tier"] == "memory"
    assert cost["capture_ms"] >= cost["copy_ms"]
    assert cost["request_id"] == "d1"
    stats = connector.stats_snapshot
    assert stats["capture_bytes_total"] == cost["copy_bytes"]
    assert stats["capture_layers_total"] == 1
    # Reported once; a second finish for the same id writes nothing.
    connector.get_finished({"d1"})
    assert len(_events(tmp_path, "donor_capture_cost")) == 1


def test_a_skipped_capture_names_the_tier_it_did_not_write_to(tmp_path):
    """The cost a skip avoided is read off the same field as a cost paid."""
    connector = SemBlendVllmConnector(
        _config(tmp_path, kv_storage_backend="memory"), KVConnectorRole.SCHEDULER
    )
    short = types.SimpleNamespace(req_id="s1", prompt_token_ids=[1, 2], block_ids=([0],))

    connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[short]))

    (skipped,) = _events(tmp_path, "capture_skipped")
    assert skipped["reason"] == "short_prompt"
    assert skipped["store_tier"] == "memory"


def test_a_registered_donor_names_the_tier_its_kv_went_to(tmp_path):
    """A donor's registration says where the KV behind it was written."""
    connector = SemBlendVllmConnector(
        _config(tmp_path, mode="discovery_only", kv_storage_backend="disk"),
        KVConnectorRole.SCHEDULER,
    )

    connector.request_finished(FakeRequest("d1", [1, 2, 3, 4]), [0])

    (registered,) = _events(tmp_path, "donor_registered")
    assert registered["store_tier"] == "disk"

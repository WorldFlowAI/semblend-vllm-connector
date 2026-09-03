"""Hit-path cost controls: served requests are not re-captured, and the
in-memory donor store serves loads without touching disk."""

from __future__ import annotations

import json
import sys
import types

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
from semblend_vllm_connector.types import (
    MaterializationKind,
    PendingLoad,
    SemanticLookupResult,
    SemanticSegment,
    SemBlendConnectorMetadata,
)


def _config(tmp_path, **extra):
    cfg = {
        "mode": "semantic_span_experimental",
        "provider": "local",
        "min_prompt_tokens": 4,
        "min_similarity": 0.3,
        "min_semantic_span": 8,
        "kv_storage_path": str(tmp_path),
        "audit_path": str(tmp_path / "audit.jsonl"),
    }
    cfg.update(extra)
    return FakeVllmConfig(FakeKvTransferConfig(cfg), cache_config=FakeCacheConfig(block_size=4))


class _ScriptedProvider:
    def __init__(self, result):
        self.result = result

    def lookup(self, req):
        return self.result

    def register(self, reg):
        return True


def _whole_span_result(n=80):
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.95,
        reusable_token_count=n,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[SemanticSegment(donor_id="d1", donor_start=0, target_start=0, token_count=n)],
    )


def _write_capture(connector, request, donor_id, token_count):
    import os

    from semblend_vllm_connector.namespace import namespace_for_request

    ns = namespace_for_request(connector._config, connector._vllm_config, request)  # noqa: SLF001
    os.makedirs(connector._donor_dir(donor_id, ns), exist_ok=True)  # noqa: SLF001
    with open(connector._donor_metadata_path(donor_id, ns), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": token_count}, f)


def test_served_request_is_not_recaptured_by_default(tmp_path):
    """A recipient that just received donor KV is not itself captured as a
    donor: the capture copy was ~230 ms of the hit path at 3.5K tokens, and
    the donor it was served from already covers the same content."""
    connector = SemBlendVllmConnector(_config(tmp_path), KVConnectorRole.SCHEDULER)
    connector._provider = _ScriptedProvider(_whole_span_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))
    _write_capture(connector, recipient, "d1", 4096)

    matched, _ = connector.get_num_new_matched_tokens(recipient, 0)
    assert matched > 0
    connector.update_state_after_alloc(recipient, FakeBlocks(([3, 4, 5],)), matched)

    new_req = types.SimpleNamespace(req_id="r1", prompt_token_ids=list(range(100)), block_ids=([1, 2, 3, 4, 5],))
    meta = connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[new_req]))
    assert [s.request_id for s in meta.stores] == []


def test_served_request_captured_when_policy_says_so(tmp_path):
    connector = SemBlendVllmConnector(
        _config(tmp_path, capture_served_requests=True), KVConnectorRole.SCHEDULER
    )
    connector._provider = _ScriptedProvider(_whole_span_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))
    _write_capture(connector, recipient, "d1", 4096)
    matched, _ = connector.get_num_new_matched_tokens(recipient, 0)
    connector.update_state_after_alloc(recipient, FakeBlocks(([3, 4, 5],)), matched)
    new_req = types.SimpleNamespace(req_id="r1", prompt_token_ids=list(range(100)), block_ids=([1, 2, 3, 4, 5],))
    meta = connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[new_req]))
    assert [s.request_id for s in meta.stores] == ["r1"]


def test_memory_backend_round_trips_without_files(tmp_path, monkeypatch):
    """With kv_storage_backend=memory the worker keeps donor layers in host
    RAM: save_kv_layer writes no safetensors file, start_load_kv reads the
    tensor back from memory. The tiny metadata file still lands on disk
    because the scheduler-role connector (another process) reads the
    captured token count from it."""
    import torch

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    calls = {"save": 0, "load": 0}

    def _save(*a, **k):
        calls["save"] += 1

    def _load(*a, **k):
        calls["load"] += 1
        raise AssertionError("disk load must not happen with the memory backend")

    fake_st_torch.save_file = _save
    fake_st_torch.load_file = _load
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    connector = SemBlendVllmConnector(
        _config(tmp_path, kv_storage_backend="memory", kv_memory_max_donors=4), KVConnectorRole.WORKER
    )

    class _HF:
        rope_theta = 10000.0
        head_dim = 16

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001
    layer_name = "model.layers.0.self_attn.attn"
    src_layer = torch.randn(2, 6, 4, 2, 16)  # kv_first paged [2, pages, bs, H, D]
    connector.register_kv_caches({layer_name: src_layer})

    # Capture donor "d1": slots for blocks 0,1 (8 tokens).
    from semblend_vllm_connector.types import PendingStore

    store = PendingStore(request_id="d1", token_ids=list(range(8)), token_count=8, namespace="ns", block_ids=([0, 1],))
    connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[], stores=[store]))
    connector.save_kv_layer(layer_name, src_layer, object())
    assert calls["save"] == 0
    assert connector._stored_donor_token_count("d1", "ns") == 8  # noqa: SLF001

    # Load into blocks 3,4 for a recipient at delta 0.
    dst_layer = torch.zeros_like(src_layer)
    connector.register_kv_caches({layer_name: dst_layer})
    load = PendingLoad(
        request_id="r1", donor_id="d1", token_count=8, materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns", block_ids=([3, 4],), donor_start=0, target_start=0,
    )
    connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[load], stores=[]))
    connector.start_load_kv(FakeForwardContext(attn_metadata=object()))
    assert calls["load"] == 0

    src_slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    dst_slots = torch.tensor([12, 13, 14, 15, 16, 17, 18, 19])
    got = connector._extract_kv_from_layer(dst_layer, dst_slots, object())  # noqa: SLF001
    want = connector._extract_kv_from_layer(src_layer, src_slots, object())  # noqa: SLF001
    torch.testing.assert_close(got, want)

    events = [json.loads(line)["event"] for line in open(tmp_path / "audit.jsonl") if line.strip()]
    assert "runtime_materialized" in events


def test_memory_backend_evicts_oldest(tmp_path, monkeypatch):
    import torch

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)
    connector = SemBlendVllmConnector(
        _config(tmp_path, kv_storage_backend="memory", kv_memory_max_donors=2), KVConnectorRole.WORKER
    )
    layer_name = "l0"
    layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({layer_name: layer})
    from semblend_vllm_connector.types import PendingStore

    for i, did in enumerate(("a", "b", "c")):
        store = PendingStore(request_id=did, token_ids=list(range(4)), token_count=4, namespace="ns", block_ids=([i],))
        connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[], stores=[store]))
        connector.save_kv_layer(layer_name, layer, object())
    keys = list(connector._memory_store.keys())  # noqa: SLF001
    assert len(keys) == 2
    assert connector._storage_key("a", "ns") not in keys  # noqa: SLF001


def test_load_materializes_on_a_no_forward_step(tmp_path, monkeypatch):
    """vLLM issues a no-forward step (forward_context.attn_metadata is None)
    when a step carries loads but no tokens to compute, which happens under
    concurrent long prefills. The scheduler already counts the span as
    computed, so refusing to materialize there killed the engine
    ("SemBlend materialization requires forward_context.attn_metadata")."""
    import torch

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.save_file = lambda *a, **k: None
    fake_st_torch.load_file = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no disk"))
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)
    connector = SemBlendVllmConnector(
        _config(tmp_path, kv_storage_backend="memory", kv_memory_max_donors=4), KVConnectorRole.WORKER
    )

    class _HF:
        rope_theta = 10000.0
        head_dim = 16

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001
    layer_name = "model.layers.0.self_attn.attn"
    src_layer = torch.randn(2, 6, 4, 2, 16)
    connector.register_kv_caches({layer_name: src_layer})
    from semblend_vllm_connector.types import PendingStore

    store = PendingStore(request_id="d1", token_ids=list(range(8)), token_count=8, namespace="ns", block_ids=([0, 1],))
    connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[], stores=[store]))
    connector.save_kv_layer(layer_name, src_layer, object())

    dst_layer = torch.zeros_like(src_layer)
    connector.register_kv_caches({layer_name: dst_layer})
    load = PendingLoad(
        request_id="r1", donor_id="d1", token_count=8, materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns", block_ids=([3, 4],), donor_start=0, target_start=0,
    )
    connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[load], stores=[]))
    connector.start_load_kv(FakeForwardContext(attn_metadata=None))

    got = connector._extract_kv_from_layer(dst_layer, torch.tensor(list(range(12, 20))), object())  # noqa: SLF001
    want = connector._extract_kv_from_layer(src_layer, torch.tensor(list(range(8))), object())  # noqa: SLF001
    torch.testing.assert_close(got, want)
    assert connector.stats_snapshot.get("materialized_without_forward") == 1

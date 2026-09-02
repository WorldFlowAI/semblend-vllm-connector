"""SEMANTIC_SPAN mode: boundary-anchored block-aligned span advertisement.

The connector consumes token-verified alignment segments from the provider,
snaps them to block edges, and answers get_num_new_matched_tokens with the
contiguous run servable from the scheduler's computed boundary. Mid-span
boundaries advance the donor offset; boundaries in novel regions advertise
zero. The pending load carries the donor start for the re-rotation loader.
"""

from __future__ import annotations

import json
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
from semblend_vllm_connector.types import (
    MaterializationKind,
    PendingLoad,
    SemanticLookupResult,
    SemanticSegment,
    SemBlendConnectorMetadata,
)


def _connector(tmp_path, block_size=4, min_span=8):
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "semantic_span_experimental",
                "provider": "local",
                "min_prompt_tokens": 4,
                "min_similarity": 0.3,
                "kv_storage_path": str(tmp_path),
                "max_materialized_tokens": 4096,
                "min_semantic_span": min_span,
            }
        ),
        cache_config=FakeCacheConfig(block_size=block_size),
    )
    return SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)


def _spanned_result():
    # Donor span: target [10..90) from donor position 210 (delta 200).
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.99,
        reusable_token_count=80,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[
            SemanticSegment(
                donor_id="d1", donor_start=210, target_start=10, token_count=80
            )
        ],
    )


class _ScriptedProvider:
    def __init__(self, result):
        self.result = result
        self.lookups = []

    def lookup(self, req):
        self.lookups.append(req)
        return self.result

    def register(self, reg):
        return True


def _write_donor_capture(connector, request, donor_id, token_count) -> None:
    """Record a donor KV capture of ``token_count`` tokens on disk, the
    prerequisite the load path reads back per layer."""
    import json
    import os

    from semblend_vllm_connector.namespace import namespace_for_request

    namespace = namespace_for_request(
        connector._config, connector._vllm_config, request  # noqa: SLF001
    )
    os.makedirs(connector._donor_dir(donor_id, namespace), exist_ok=True)  # noqa: SLF001
    path = connector._donor_metadata_path(donor_id, namespace)  # noqa: SLF001
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"token_count": token_count}, f)


def test_first_schedule_advertises_zero_when_span_is_interior(tmp_path) -> None:
    connector = _connector(tmp_path)
    connector._provider = _ScriptedProvider(_spanned_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))

    matched, async_load = connector.get_num_new_matched_tokens(recipient, 0)

    # Boundary 0 sits in the novel head [0..10): nothing servable yet.
    assert (matched, async_load) == (0, False)


def test_boundary_inside_span_advertises_block_aligned_tail(tmp_path) -> None:
    connector = _connector(tmp_path)
    connector._provider = _ScriptedProvider(_spanned_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    # Simulate a continuation boundary at 12 (block-aligned, inside span
    # [12..88) after snapping 10->12, 90->88).
    matched, async_load = connector.get_num_new_matched_tokens(recipient, 12)

    assert (matched, async_load) == (76, False)

    connector.update_state_after_alloc(recipient, FakeBlocks(([3, 4, 5],)), matched)
    metadata = connector.build_connector_meta(FakeSchedulerOutput())
    assert len(metadata.loads) == 1
    load = metadata.loads[0]
    assert load.materialization_kind == MaterializationKind.SEMANTIC_SPAN
    assert load.token_count == 76
    # Donor start advanced by the snap (10->12): 210 + 2.
    assert load.donor_start == 212


def test_mid_span_boundary_advances_donor_offset(tmp_path) -> None:
    connector = _connector(tmp_path)
    connector._provider = _ScriptedProvider(_spanned_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    matched, _ = connector.get_num_new_matched_tokens(recipient, 40)

    assert matched == 48  # [40..88)
    load = connector._pending_loads["r1"]  # noqa: SLF001
    assert load.donor_start == 240  # 210 + (40 - 10)


def test_span_advertisement_capped_by_stored_donor_kv(tmp_path) -> None:
    """(take-15 pre-flight) Donor KV capture is a block-aligned prefix of
    the FIRST scheduled chunk, so a span reaching past the captured length
    would load a short tensor and kill the engine at inject. The advertise
    path must trim every span to the stored capture."""
    connector = _connector(tmp_path)
    result = SemanticLookupResult(
        donor_id="d1",
        similarity=0.99,
        reusable_token_count=80,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[
            SemanticSegment(
                donor_id="d1", donor_start=10, target_start=10, token_count=80
            )
        ],
    )
    connector._provider = _ScriptedProvider(result)  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=40)

    matched, _ = connector.get_num_new_matched_tokens(recipient, 12)

    # Trimmed to donor window [10..40): 30 tokens, snapped 10->12 -> 28.
    assert matched == 28
    load = connector._pending_loads["r1"]  # noqa: SLF001
    assert load.donor_start == 12
    assert load.token_count == 28


def test_span_advertisement_zero_without_stored_donor_kv(tmp_path) -> None:
    """(take-15 pre-flight) A donor with no KV capture on disk has nothing
    to load; advertising anyway dies at start_load_kv (take-8 class)."""
    connector = _connector(tmp_path)
    connector._provider = _ScriptedProvider(_spanned_result())  # noqa: SLF001
    recipient = FakeRequest("r1", list(range(100)))

    matched, async_load = connector.get_num_new_matched_tokens(recipient, 12)

    assert (matched, async_load) == (0, False)


def test_semantic_span_slice_rotates_k_and_preserves_v(tmp_path) -> None:
    import torch

    from semblend_vllm_connector.semantic_span import rerotate_k
    from semblend_vllm_connector.types import PendingLoad

    connector = _connector(tmp_path)

    class _HF:
        rope_theta = 10000.0
        head_dim = 16

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001

    # Stored donor KV uses the flattened extract contract [2, n, H*D].
    donor_kv = torch.randn(2, 100, 32)  # heads=2, head_dim=16 flattened
    load = PendingLoad(
        request_id="r1",
        donor_id="d1",
        token_count=8,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns",
        donor_start=40,
        target_start=12,
    )

    out = connector._semantic_span_slice(donor_kv, load, attn_metadata=object())  # noqa: SLF001

    assert out.shape == (2, 8, 32)
    expected_k = rerotate_k(
        donor_kv[0, 40:48].reshape(8, 2, 16),
        donor_start=40,
        target_start=12,
        head_dim=16,
        rope_theta=10000.0,
    ).reshape(8, 32)
    torch.testing.assert_close(out[0], expected_k)
    torch.testing.assert_close(out[1], donor_kv[1, 40:48])  # V untouched


def test_rope_params_from_unified_rope_parameters(tmp_path) -> None:
    """(take-15 regression) transformers 5.x moved rope_theta into the
    hf config's rope_parameters dict and dropped the flat attribute;
    reading only the legacy attribute declined every layer of every
    semantic-span load (materialized 0 layers, loud engine death)."""
    connector = _connector(tmp_path)

    class _HF:
        rope_parameters = {"rope_theta": 1000000.0, "rope_type": "default"}
        hidden_size = 3584
        num_attention_heads = 28

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001
    assert connector._rope_params() == (1000000.0, 128)  # noqa: SLF001


def test_rope_params_decline_non_default_rope_type(tmp_path) -> None:
    """Scaled rope variants (yarn, linear, dynamic) rotate K differently
    from the plain re-rotation the realizer applies; serving them would
    place mis-rotated keys. Decline instead."""
    connector = _connector(tmp_path)

    class _HF:
        rope_parameters = {"rope_theta": 1000000.0, "rope_type": "yarn"}
        hidden_size = 3584
        num_attention_heads = 28

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001
    assert connector._rope_params() is None  # noqa: SLF001


def test_semantic_span_slice_declines_without_rope_params(tmp_path) -> None:
    import torch

    from semblend_vllm_connector.types import PendingLoad

    connector = _connector(tmp_path)
    load = PendingLoad(
        request_id="r1",
        donor_id="d1",
        token_count=8,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns",
        donor_start=40,
        target_start=12,
    )
    out = connector._semantic_span_slice(torch.randn(2, 100, 32), load, object())  # noqa: SLF001
    assert out is None


def test_semantic_span_mode_enables_donor_stores(tmp_path) -> None:
    """(take-8 regression) SEMANTIC_SPAN mode must persist donor KV: the
    load path reads per-layer safetensors, so a mode missing from the
    materialization set advertises loads whose files were never written
    (FileNotFoundError at start_load_kv, engine death)."""
    connector = _connector(tmp_path)
    assert connector._materialization_enabled()  # noqa: SLF001


def test_extract_kv_gathers_without_full_layer_copy(tmp_path) -> None:
    """(take-9 regression) Extracting donor KV must gather selected slots
    via page/offset indexing; reshaping the whole paged layer copies it
    when strides are not flat-contiguous (multi-GiB OOM on capture)."""
    import torch

    connector = _connector(tmp_path)
    # Non-MLA paged layer [2, pages, page_size, heads, dim]; the contract
    # is equality with the flat-reshape reference gather.
    layer = torch.randn(2, 6, 4, 2, 8)
    slot_mapping = torch.tensor([5, 6, 13, 21])  # page 1/1/3/5 offsets 1/2/1/1
    out = connector._extract_kv_from_layer(layer, slot_mapping, object())  # noqa: SLF001
    ref = layer.reshape(2, 24, -1)[:, slot_mapping, ...]
    torch.testing.assert_close(out, ref)


def test_extract_and_inject_handle_blocks_first_layout(tmp_path) -> None:
    """(take-10 regression) vLLM 0.26 stores non-MLA layers as
    [blocks, 2, block_size, H, D]; assuming K/V-first broadcast a
    pages-sized dimension (multi-GiB OOM). Round-trip must hold in BOTH
    layouts."""
    import torch

    connector = _connector(tmp_path)
    slot_mapping = torch.tensor([5, 6, 13, 21])

    for layout in ("kv_first", "blocks_first"):
        if layout == "kv_first":
            layer = torch.randn(2, 6, 4, 2, 8)
            ref = layer.reshape(2, 24, -1)[:, slot_mapping, ...]
        else:
            layer = torch.randn(6, 2, 4, 2, 8)
            ref = layer.permute(1, 0, 2, 3, 4).reshape(2, 24, -1)[:, slot_mapping, ...]
        out = connector._extract_kv_from_layer(layer, slot_mapping, object())  # noqa: SLF001
        torch.testing.assert_close(out, ref, msg=layout)

        dst = torch.zeros_like(layer)
        connector._inject_kv_into_layer(dst, out, slot_mapping, object())  # noqa: SLF001
        back = connector._extract_kv_from_layer(dst, slot_mapping, object())  # noqa: SLF001
        torch.testing.assert_close(back, ref, msg=f"round-trip {layout}")


def _worker_connector(tmp_path, audit_path):
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "semantic_span_experimental",
                "provider": "local",
                "min_prompt_tokens": 4,
                "kv_storage_path": str(tmp_path),
                "audit_path": str(audit_path),
            }
        ),
        cache_config=FakeCacheConfig(block_size=4),
    )
    return SemBlendVllmConnector(vllm_config, KVConnectorRole.WORKER)


def _semantic_span_load(**overrides):
    fields = dict(
        request_id="r1",
        donor_id="d1",
        token_count=8,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns",
        block_ids=([3, 4],),
        donor_start=40,
        target_start=12,
    )
    fields.update(overrides)
    return PendingLoad(**fields)


def _audit_events(audit_path):
    return [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_zero_layer_semantic_span_load_fails_loud(tmp_path, monkeypatch) -> None:
    """(take-14 regression) A semantic-span load that yields no KV layers
    must raise, never decline silently: the scheduler already skipped
    compute for the advertised tokens, so a silent skip decodes garbage
    over uninitialized blocks (the take-14 'supplied-but-not-loaded')."""
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda filename: {}  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    audit_path = tmp_path / "audit.jsonl"
    connector = _worker_connector(tmp_path, audit_path)
    connector.bind_connector_metadata(
        SemBlendConnectorMetadata(loads=[_semantic_span_load()])
    )

    # vLLM 0.26 shape: no register_kv_caches call yet, and the context walk
    # finds layers without a kv_cache attribute.
    context = FakeForwardContext(
        no_compile_layers={"layer0": types.SimpleNamespace(kv_cache=None)}
    )
    with pytest.raises(RuntimeError, match="materialized 0 layers"):
        connector.start_load_kv(context)

    events = {event["event"] for event in _audit_events(audit_path)}
    assert "runtime_materialization_failed_loud" in events
    assert "runtime_materialized" not in events


def test_registered_kv_caches_feed_semantic_span_load(tmp_path, monkeypatch) -> None:
    """(take-14 regression) vLLM 0.26 workers hand KV tensors to the
    connector via register_kv_caches; layer.kv_cache is gone, so the
    forward-context walk finds nothing. The registered dict must feed the
    load path end to end: slice, re-rotate, inject, audit materialized."""
    import torch

    audit_path = tmp_path / "audit.jsonl"
    connector = _worker_connector(tmp_path, audit_path)

    class _HF:
        rope_theta = 10000.0
        head_dim = 16

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001

    donor_kv = torch.randn(2, 100, 32)  # [2, tokens, H*D], extract contract
    load = _semantic_span_load()
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda filename: {"kv_cache": donor_kv}  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    dst_layer = torch.zeros(2, 6, 4, 2, 16)  # kv_first paged [2, pages, bs, H, D]
    connector.register_kv_caches({"model.layers.0.self_attn.attn": dst_layer})
    connector.bind_connector_metadata(SemBlendConnectorMetadata(loads=[load]))

    attn_metadata = object()
    connector.start_load_kv(
        FakeForwardContext(
            attn_metadata=attn_metadata,
            no_compile_layers={"layer0": types.SimpleNamespace(kv_cache=None)},
        )
    )

    slot_mapping = torch.tensor([12, 13, 14, 15, 16, 17, 18, 19])  # blocks 3,4
    injected = connector._extract_kv_from_layer(  # noqa: SLF001
        dst_layer, slot_mapping, attn_metadata
    )
    expected = connector._semantic_span_slice(donor_kv, load, attn_metadata)  # noqa: SLF001
    torch.testing.assert_close(injected, expected)

    by_event = {event["event"]: event for event in _audit_events(audit_path)}
    assert by_event["kv_caches_registered"]["layer_count"] == 1
    assert by_event["runtime_materialized"]["layers_materialized"] == 1


def test_extract_and_inject_round_trip_packed_content_layout(tmp_path) -> None:
    """(take-11 regression) vLLM 0.26 flash_attn packs K/V into the content
    dim: (blocks, kv_heads, block_size, 2*head_size), K first half. The
    4-dim layout must gather, split, and round-trip exactly."""
    import torch

    connector = _connector(tmp_path)
    layer = torch.randn(6, 2, 4, 16)  # blocks=6, H=2, bs=4, 2D=16
    slot_mapping = torch.tensor([5, 6, 13, 21])
    out = connector._extract_kv_from_layer(layer, slot_mapping, object())  # noqa: SLF001
    pages, offs = slot_mapping // 4, slot_mapping % 4
    ref_g = layer[pages, :, offs, :]
    ref_k, ref_v = ref_g.split(8, dim=-1)
    ref = torch.stack((ref_k, ref_v)).reshape(2, 4, -1)
    torch.testing.assert_close(out, ref)

    dst = torch.zeros_like(layer)
    connector._inject_kv_into_layer(dst, out, slot_mapping, object())  # noqa: SLF001
    back = connector._extract_kv_from_layer(dst, slot_mapping, object())  # noqa: SLF001
    torch.testing.assert_close(back, ref)

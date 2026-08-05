"""SEMANTIC_SPAN mode: boundary-anchored block-aligned span advertisement.

The connector consumes token-verified alignment segments from the provider,
snaps them to block edges, and answers get_num_new_matched_tokens with the
contiguous run servable from the scheduler's computed boundary. Mid-span
boundaries advance the donor offset; boundaries in novel regions advertise
zero. The pending load carries the donor start for the re-rotation loader.
"""

from __future__ import annotations

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import (
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
)

from test_connector_discovery import (
    FakeBlocks,
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeSchedulerOutput,
    FakeVllmConfig,
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

    matched, _ = connector.get_num_new_matched_tokens(recipient, 40)

    assert matched == 48  # [40..88)
    load = connector._pending_loads["r1"]  # noqa: SLF001
    assert load.donor_start == 240  # 210 + (40 - 10)


def test_semantic_span_slice_rotates_k_and_preserves_v(tmp_path) -> None:
    import torch

    from semblend_vllm_connector.semantic_span import rerotate_k
    from semblend_vllm_connector.types import PendingLoad

    connector = _connector(tmp_path)

    class _HF:
        rope_theta = 10000.0
        head_dim = 16

    connector._vllm_config.model_config.hf_config = _HF()  # noqa: SLF001

    donor_kv = torch.randn(2, 100, 2, 16)  # [K/V, tokens, heads, head_dim]
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

    assert out.shape == (2, 8, 2, 16)
    expected_k = rerotate_k(
        donor_kv[0, 40:48], donor_start=40, target_start=12, head_dim=16, rope_theta=10000.0
    )
    torch.testing.assert_close(out[0], expected_k)
    torch.testing.assert_close(out[1], donor_kv[1, 40:48])  # V untouched


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
    out = connector._semantic_span_slice(torch.randn(2, 100, 2, 16), load, object())  # noqa: SLF001
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

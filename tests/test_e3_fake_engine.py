"""Fake-engine integration harness: the stock vLLM 0.29 connector contract.

Every other test in this repo pokes one connector method. This one drives the
four calls stock vLLM makes, in the order it makes them, across the role split
the engine imposes (scheduler process advertises and allocates, worker process
loads):

    get_num_new_matched_tokens(request, B)   -> scheduler
    update_state_after_alloc(request, blocks, N)
    build_connector_meta(scheduler_output)
    start_load_kv(forward_context)           -> worker

The destination is real CPU torch tensors shaped like a vLLM 0.29 per-layer KV
view: 4-D ``[num_blocks, num_heads, block_size, content]``, which the engine
standardizes on for every backend (``create_kv_cache_views`` in
``vllm/v1/kv_cache_interface.py``; the H of that page comes from
``AttentionSpec.num_heads``, i.e. ``num_head_slots`` when a backend publishes
one and ``num_kv_heads`` otherwise).

The contract this pins:

* a load advertised at boundary ``B`` for ``N`` tokens writes exactly the
  target-frame slots ``[B, B + N)`` -- and the target frame is the request's
  own block list, so slot ``t`` lives at block ``group[t // block_size]``;
* every block before ``B`` is byte-identical afterwards. Those are the blocks
  vLLM's exact prefix cache handed the request; they are shared by reference
  count with every other request holding that prefix, so a write into them
  corrupts unrelated traffic and leaves ``[B, B + N)`` -- which the scheduler
  has already marked computed -- uninitialized.

To extend: build a layer with ``make_kv_layer``, hand ``FakeEngine`` the
alignment segment the provider would return, and call ``run_load``. The
returned ``LoadOutcome`` carries pre/post snapshots of every layer, so new
cases assert on ``changed_slots`` and ``gather_target_frame`` rather than on
connector internals.
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass
from typing import Any

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
from semblend_vllm_connector.semantic_span import rerotate_k
from semblend_vllm_connector.types import (
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
    SemBlendConnectorMetadata,
)

torch = pytest.importorskip("torch")

BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 8
NUM_BLOCKS = 12
ROPE_THETA = 10000.0
LAYER_NAMES = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")

PROMPT_TOKENS = 160
DONOR_TOKENS = 256
# Physically scattered, deliberately non-monotonic: a destination derived from
# block index 0 (or from an assumed-contiguous run) lands somewhere visibly
# wrong instead of somewhere plausible.
BLOCK_IDS = (7, 2, 9, 4, 11, 6, 1, 8, 3, 10)
BOUNDARY = 32  # block-aligned local exact-prefix hit: blocks 7, 2
SPAN_TOKENS = 64  # target [32, 96) -> blocks 9, 4, 11, 6
DONOR_START = 96  # RoPE delta of -64 between donor and target frames


class _HfConfig:
    rope_theta = ROPE_THETA
    head_dim = HEAD_SIZE


class _SpanProvider:
    """Returns one token-verified alignment segment, or a miss when None."""

    def __init__(self, segment: SemanticSegment | None) -> None:
        self._segment = segment

    def lookup(self, request: Any) -> SemanticLookupResult | None:
        if self._segment is None:
            return None
        return SemanticLookupResult(
            donor_id=self._segment.donor_id,
            similarity=0.99,
            reusable_token_count=self._segment.token_count,
            materialization_kind=MaterializationKind.SEMANTIC_SPAN,
            segments=[self._segment],
        )

    def register_donor(self, donor: Any) -> None:
        return None

    def clear_donors(self) -> None:
        return None


def make_kv_layer(
    *,
    num_blocks: int = NUM_BLOCKS,
    heads: int = NUM_KV_HEADS,
    block_size: int = BLOCK_SIZE,
    content: int = 2 * HEAD_SIZE,
    seed: int = 0,
) -> Any:
    """One per-layer KV view in vLLM 0.29's 4-D [B, H, N, C] page layout."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_blocks, heads, block_size, content, generator=generator)


def make_donor_kv(num_tokens: int = DONOR_TOKENS, *, seed: int = 0) -> Any:
    """Stored donor KV in the capture contract: [2, tokens, heads * head_size]."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, num_tokens, NUM_KV_HEADS * HEAD_SIZE, generator=generator)


def _patch_donor_reads(monkeypatch: Any, donor_kv: dict[str, Any]) -> None:
    """Serve stored donor layers by layer name, as safetensors would."""

    def load_file(filename: str) -> dict[str, Any]:
        layer_name = os.path.basename(filename).removesuffix(".safetensors")
        return {"kv_cache": donor_kv[layer_name]}

    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = load_file
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)


@dataclass
class LoadOutcome:
    advertised: int
    async_load: bool
    metadata: SemBlendConnectorMetadata
    before: dict[str, Any]
    after: dict[str, Any]

    @property
    def load(self) -> Any:
        assert len(self.metadata.loads) == 1
        return self.metadata.loads[0]


class FakeEngine:
    """The slice of stock vLLM the connector actually sees.

    Both connector roles are constructed from one config, as the engine does
    in two processes, and the worker holds the KV tensors a load writes into.
    """

    def __init__(
        self,
        tmp_path: Any,
        monkeypatch: Any,
        *,
        segment: SemanticSegment | None,
        layers: dict[str, Any] | None = None,
        donor_tokens: int = DONOR_TOKENS,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        config = {
            "mode": "semantic_span_experimental",
            "provider": "local",
            "min_prompt_tokens": BLOCK_SIZE,
            "min_semantic_span": BLOCK_SIZE,
            "max_materialized_tokens": 4096,
            "kv_storage_path": str(tmp_path / "kv"),
            "audit_path": str(tmp_path / "audit.jsonl"),
        }
        config.update(extra_config or {})
        vllm_config = FakeVllmConfig(
            FakeKvTransferConfig(config),
            cache_config=FakeCacheConfig(block_size=BLOCK_SIZE),
        )
        vllm_config.model_config.hf_config = _HfConfig()

        self.scheduler = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)
        self.worker = SemBlendVllmConnector(vllm_config, KVConnectorRole.WORKER)
        self.scheduler._provider = _SpanProvider(segment)  # noqa: SLF001

        if layers is None:
            layers = {name: make_kv_layer(seed=index) for index, name in enumerate(LAYER_NAMES)}
        self.layers = dict(layers)
        self.donor_kv = {
            name: make_donor_kv(donor_tokens, seed=100 + index)
            for index, name in enumerate(self.layers)
        }
        self.worker.register_kv_caches(self.layers)
        _patch_donor_reads(monkeypatch, self.donor_kv)

    def record_donor_capture(self, request: Any, donor_id: str, token_count: int) -> None:
        """Publish the donor's captured length, as save_kv_layer does."""
        connector = self.scheduler
        namespace = namespace_for_request(
            connector._config,
            connector._vllm_config,
            request,  # noqa: SLF001
        )
        os.makedirs(connector._donor_dir(donor_id, namespace), exist_ok=True)  # noqa: SLF001
        path = connector._donor_metadata_path(donor_id, namespace)  # noqa: SLF001
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token_count": token_count}, f)


def run_load(
    engine: FakeEngine,
    request: Any,
    *,
    boundary: int,
    block_ids: tuple[int, ...],
) -> LoadOutcome:
    """Drive one scheduling step end to end, in the engine's own call order."""
    advertised, async_load = engine.scheduler.get_num_new_matched_tokens(request, boundary)
    engine.scheduler.update_state_after_alloc(request, FakeBlocks((list(block_ids),)), advertised)
    metadata = engine.scheduler.build_connector_meta(FakeSchedulerOutput())

    before = {name: layer.clone() for name, layer in engine.layers.items()}
    engine.worker.bind_connector_metadata(metadata)
    engine.worker.start_load_kv(FakeForwardContext())
    after = {name: layer.clone() for name, layer in engine.layers.items()}

    return LoadOutcome(advertised, async_load, metadata, before, after)


def target_frame_slots(
    block_ids: tuple[int, ...],
    start: int,
    end: int,
    block_size: int = BLOCK_SIZE,
) -> set[int]:
    """Physical KV slots backing target-frame token positions [start, end)."""
    return {
        block_ids[position // block_size] * block_size + position % block_size
        for position in range(start, end)
    }


def changed_slots(
    before: dict[str, Any],
    after: dict[str, Any],
    block_size: int = BLOCK_SIZE,
) -> set[int]:
    """Physical slots whose KV differs in any layer."""
    slots: set[int] = set()
    for name, original in before.items():
        touched = (after[name] != original).any(dim=1).any(dim=-1)
        for page, offset in touched.nonzero().tolist():
            slots.add(page * block_size + offset)
    return slots


def gather_target_frame(
    layer: Any,
    block_ids: tuple[int, ...],
    start: int,
    end: int,
    block_size: int = BLOCK_SIZE,
) -> Any:
    """The layer's cells for target-frame positions [start, end): [n, H, C]."""
    positions = range(start, end)
    pages = torch.tensor([block_ids[p // block_size] for p in positions])
    offsets = torch.tensor([p % block_size for p in positions])
    return layer[pages, :, offsets, :]


def expected_kv(donor_layer: Any, *, donor_start: int, target_start: int, token_count: int) -> Any:
    """K re-rotated donor -> target, V verbatim: [n, H, head_size] each."""
    window = donor_layer[:, donor_start : donor_start + token_count, :]
    k = rerotate_k(
        window[0].reshape(token_count, NUM_KV_HEADS, HEAD_SIZE),
        donor_start=donor_start,
        target_start=target_start,
        head_dim=HEAD_SIZE,
        rope_theta=ROPE_THETA,
    )
    return k, window[1].reshape(token_count, NUM_KV_HEADS, HEAD_SIZE)


def _byte_view(tensor: Any) -> Any:
    return tensor.contiguous().view(torch.uint8)


def _span_engine(tmp_path, monkeypatch, *, target_start=BOUNDARY, **kwargs) -> FakeEngine:
    return FakeEngine(
        tmp_path,
        monkeypatch,
        segment=SemanticSegment(
            donor_id="d1",
            donor_start=DONOR_START,
            target_start=target_start,
            token_count=SPAN_TOKENS,
        ),
        **kwargs,
    )


def _recipient() -> FakeRequest:
    return FakeRequest("r1", list(range(PROMPT_TOKENS)))


def test_load_at_a_nonzero_boundary_writes_exactly_the_advertised_window(
    tmp_path, monkeypatch
) -> None:
    """The engine skips prefill for [B, B+N) in the target frame, so that is
    precisely the set of slots the worker owes it."""
    engine = _span_engine(tmp_path, monkeypatch)
    recipient = _recipient()
    engine.record_donor_capture(recipient, "d1", DONOR_TOKENS)

    outcome = run_load(engine, recipient, boundary=BOUNDARY, block_ids=BLOCK_IDS)

    assert (outcome.advertised, outcome.async_load) == (SPAN_TOKENS, False)
    assert outcome.load.target_start == BOUNDARY
    assert outcome.load.donor_start == DONOR_START
    assert changed_slots(outcome.before, outcome.after) == target_frame_slots(
        BLOCK_IDS, BOUNDARY, BOUNDARY + SPAN_TOKENS
    )


def test_blocks_before_the_boundary_are_byte_identical_after_a_load(tmp_path, monkeypatch) -> None:
    """The blocks under [0, B) came from vLLM's exact prefix cache and are
    shared by reference count with every other request holding that prefix.
    A load must not touch one byte of them."""
    engine = _span_engine(tmp_path, monkeypatch)
    recipient = _recipient()
    engine.record_donor_capture(recipient, "d1", DONOR_TOKENS)

    outcome = run_load(engine, recipient, boundary=BOUNDARY, block_ids=BLOCK_IDS)

    prefix_pages = list(BLOCK_IDS[: BOUNDARY // BLOCK_SIZE])
    for name, original in outcome.before.items():
        assert torch.equal(
            _byte_view(outcome.after[name][prefix_pages]),
            _byte_view(original[prefix_pages]),
        ), f"{name}: shared prefix blocks {prefix_pages} were overwritten"


def test_loaded_window_carries_the_rerotated_donor_kv(tmp_path, monkeypatch) -> None:
    """Right slots, right contents: K rotated by (target - donor), V verbatim,
    and each layer served from its own stored donor tensor."""
    engine = _span_engine(tmp_path, monkeypatch)
    recipient = _recipient()
    engine.record_donor_capture(recipient, "d1", DONOR_TOKENS)

    outcome = run_load(engine, recipient, boundary=BOUNDARY, block_ids=BLOCK_IDS)

    for name, layer in outcome.after.items():
        k, v = expected_kv(
            engine.donor_kv[name],
            donor_start=DONOR_START,
            target_start=BOUNDARY,
            token_count=SPAN_TOKENS,
        )
        written = gather_target_frame(layer, BLOCK_IDS, BOUNDARY, BOUNDARY + SPAN_TOKENS)
        torch.testing.assert_close(written, torch.cat((k, v), dim=-1), msg=name)


def test_boundary_zero_load_writes_the_head_of_the_target_frame(tmp_path, monkeypatch) -> None:
    """The degenerate case where target frame and block list coincide. It has
    always worked; it is here so a destination fix cannot regress it."""
    engine = _span_engine(tmp_path, monkeypatch, target_start=0)
    recipient = _recipient()
    engine.record_donor_capture(recipient, "d1", DONOR_TOKENS)

    outcome = run_load(engine, recipient, boundary=0, block_ids=BLOCK_IDS)

    assert outcome.advertised == SPAN_TOKENS
    assert changed_slots(outcome.before, outcome.after) == target_frame_slots(
        BLOCK_IDS, 0, SPAN_TOKENS
    )


def test_declined_lookup_leaves_every_kv_byte_untouched(tmp_path, monkeypatch) -> None:
    """A miss must be inert: no advertisement, no metadata, no write."""
    engine = FakeEngine(tmp_path, monkeypatch, segment=None)
    recipient = _recipient()

    outcome = run_load(engine, recipient, boundary=BOUNDARY, block_ids=BLOCK_IDS)

    assert (outcome.advertised, outcome.async_load) == (0, False)
    assert outcome.metadata.loads == []
    assert changed_slots(outcome.before, outcome.after) == set()


def test_stock_four_dim_view_packs_k_then_v_in_the_content_dim(tmp_path, monkeypatch) -> None:
    """Pin the two shape facts the write path infers from, so a layout change
    fails here rather than silently in a serving run.

    The 4-D branch reads ``heads = shape[1]`` and ``head_size = shape[3] // 2``
    -- i.e. it assumes one head slot per KV head and a content dim packing K
    then V for that head. That holds for the dense CUDA backends and is what
    the extract side splits on.
    """
    engine = _span_engine(tmp_path, monkeypatch)
    recipient = _recipient()
    engine.record_donor_capture(recipient, "d1", DONOR_TOKENS)

    outcome = run_load(engine, recipient, boundary=BOUNDARY, block_ids=BLOCK_IDS)

    for name, layer in outcome.after.items():
        assert layer.shape == (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE)
        k, v = expected_kv(
            engine.donor_kv[name],
            donor_start=DONOR_START,
            target_start=BOUNDARY,
            token_count=SPAN_TOKENS,
        )
        written = gather_target_frame(layer, BLOCK_IDS, BOUNDARY, BOUNDARY + SPAN_TOKENS)
        torch.testing.assert_close(written[..., :HEAD_SIZE], k, msg=f"{name} K half")
        torch.testing.assert_close(written[..., HEAD_SIZE:], v, msg=f"{name} V half")


def test_packed_head_slot_view_is_refused_rather_than_silently_written(
    tmp_path, monkeypatch
) -> None:
    """``head_size = shape[3] // 2`` is wrong wherever a backend publishes
    ``num_head_slots``: FlashInfer with nvfp4 uses ``2 * num_kv_heads`` head
    slots and the ROCm attention backends use 2, and in both the content dim
    holds a single head_size rather than a packed K|V pair.

    Nothing downstream catches it -- the element counts still multiply out, so
    the reshape and the indexed assignment both succeed and K and V land in
    the wrong cells of a live page. The connector must decline a page whose
    content dim is not twice the model's head size.
    """
    packed = {
        name: make_kv_layer(heads=2 * NUM_KV_HEADS, content=HEAD_SIZE, seed=index)
        for index, name in enumerate(LAYER_NAMES)
    }
    engine = _span_engine(tmp_path, monkeypatch, layers=packed)
    recipient = _recipient()
    engine.record_donor_capture(recipient, "d1", DONOR_TOKENS)

    with pytest.raises(RuntimeError):
        run_load(engine, recipient, boundary=BOUNDARY, block_ids=BLOCK_IDS)

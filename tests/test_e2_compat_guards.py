"""E2 — B7 compatibility guards: rope shapes and engine configs we decline.

B7(a) — the semantic-span realizer re-rotates cached K by a constant
position delta. That identity only holds for plain, full-width RoPE: every
scaled variant (llama3, yarn, linear, dynamic, longrope) bends the frequency
curve and every partial variant rotates a prefix of the head dim, so keys
re-rotated as if they were plain land on the wrong angle. Those configs must
decline rather than serve mis-rotated keys — and, just as importantly, a
plain config must NOT decline, or the mode silently does nothing.

B7(c) — context parallelism moves the scheduler's alignment unit from
``block_size`` to ``block_size * cp``, so every destination this connector
computes would address the wrong slots. The decline is taken once at startup
and has to suppress both halves of the connector: serving loads and
registering donors (a donor captured under an unserveable layout is a
landmine for every later recipient).

The last section runs the same startup gate against vLLM's own KV-cache
specs rather than stand-ins, which is the only way these guards can catch the
drift they exist for. It needs the real types; ``tests/conftest.py`` loads
them when it can and the pytest header says whether it did, so a run can
never claim vLLM compatibility while testing a shim.
"""

from __future__ import annotations

import dataclasses
import json
import types

import pytest
from test_connector_discovery import (
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import MaterializationKind, PendingLoad


def _connector(tmp_path, *, parallel_config=None, audit_path=None, kv_cache_config=None):
    settings = {
        "mode": "semantic_span_experimental",
        "provider": "local",
        "min_prompt_tokens": 4,
        "min_similarity": 0.3,
        "min_semantic_span": 8,
        "max_materialized_tokens": 4096,
        "kv_storage_path": str(tmp_path),
        "log_decisions": False,
    }
    if audit_path is not None:
        settings["audit_path"] = str(audit_path)
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(settings),
        cache_config=FakeCacheConfig(block_size=4),
    )
    if parallel_config is not None:
        # FakeVllmConfig mirrors only the fields the connector reads; the
        # parallel config is attached here because these guards are the only
        # callers that consult it.
        vllm_config.parallel_config = parallel_config
    return SemBlendVllmConnector(
        vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config=kv_cache_config
    )


def _with_hf_config(connector, hf_config):
    connector._vllm_config.model_config.hf_config = hf_config  # noqa: SLF001
    return connector


def _audit_events(audit_path):
    if not audit_path.exists():
        return []
    return [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# B7(a) — rope shapes
# --------------------------------------------------------------------------


def test_llama3_scaling_beside_flat_theta_declines(tmp_path) -> None:
    """Llama-3.1/3.3 carry a flat ``rope_theta`` *and* a ``rope_scaling``
    dict. Reading the flat attribute first and taking its presence as proof
    of plain RoPE skips the bail-out on exactly the models that need it: the
    llama3 curve rescales low frequencies, so a constant-delta re-rotation
    places keys at an angle the model never assigned to that position.
    """
    connector = _connector(tmp_path)

    class _HF:
        rope_theta = 500000.0
        head_dim = 128
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }

    _with_hf_config(connector, _HF())

    assert connector._rope_params() is None  # noqa: SLF001


def test_nested_rope_parameters_decline_when_a_layer_type_is_scaled(tmp_path) -> None:
    """transformers 5.x may key ``rope_parameters`` by layer type rather
    than carrying one flat dict (vLLM's ``is_rope_parameters_nested``: every
    key drawn from ``ALLOWED_LAYER_TYPES``). Reading ``rope_type`` off the
    outer dict then finds nothing, reads it as "default", and a flat
    ``rope_theta`` satisfies the rest of the lookup — so a yarn-scaled layer
    type is served as if it were plain RoPE.
    """
    connector = _connector(tmp_path)

    class _HF:
        rope_theta = 1000000.0
        head_dim = 128
        rope_parameters = {
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        }

    _with_hf_config(connector, _HF())

    assert connector._rope_params() is None  # noqa: SLF001


def test_partial_rotary_factor_declines(tmp_path) -> None:
    """Partial rotary embeddings rotate only the first
    ``partial_rotary_factor * head_dim`` channels and leave the tail as
    plain content. The realizer rotates the full width, which corrupts that
    tail. vLLM reads the factor from the flat attribute or from the rope
    dict, so both shapes have to decline.
    """
    connector = _connector(tmp_path)

    class _FlatAttribute:
        rope_theta = 10000.0
        head_dim = 128
        partial_rotary_factor = 0.4

    class _InRopeDict:
        rope_theta = 10000.0
        head_dim = 128
        rope_parameters = {
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.4,
        }

    _with_hf_config(connector, _FlatAttribute())
    assert connector._rope_params() is None  # noqa: SLF001

    _with_hf_config(connector, _InRopeDict())
    assert connector._rope_params() is None  # noqa: SLF001


def test_plain_qwen_config_is_served(tmp_path) -> None:
    """The guard against over-declining. A stock Qwen2.5 config carries a
    flat ``rope_theta``, an explicit null ``rope_scaling`` and no
    ``head_dim`` (it is ``hidden_size // num_attention_heads``). Declining
    it leaves the mode advertising spans that every layer then skips, which
    shows up only as a stat — so assert the slice actually materializes.
    """
    torch = pytest.importorskip("torch")

    connector = _connector(tmp_path)

    class _HF:
        rope_theta = 1000000.0
        rope_scaling = None
        hidden_size = 3584
        num_attention_heads = 28

    _with_hf_config(connector, _HF())

    assert connector._rope_params() == (1000000.0, 128)  # noqa: SLF001

    load = PendingLoad(
        request_id="r1",
        donor_id="d1",
        token_count=8,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns",
        block_ids=([0, 1, 2, 3, 4],),
        donor_start=16,
        target_start=12,
    )
    donor_kv = torch.randn(2, 32, 256)  # [2, tokens, H*D], head_dim 128
    sliced = connector._semantic_span_slice(donor_kv, load, attn_metadata=object())  # noqa: SLF001

    assert sliced is not None
    assert sliced.shape == (2, 8, 256)
    assert connector.stats_snapshot.get("semantic_span_declined_no_rope_params", 0) == 0


# --------------------------------------------------------------------------
# B7(c) — context parallelism
# --------------------------------------------------------------------------


def test_context_parallel_declines_and_skips_donor_registration(tmp_path) -> None:
    """Context parallelism is refused at startup, and the refusal has to
    reach both hooks: the match hook advertises nothing, and no donor is
    registered. Registering anyway publishes a donor whose captured KV was
    never addressable with this connector's block arithmetic, and every
    later recipient plans against it.
    """
    audit_path = tmp_path / "audit.jsonl"
    connector = _connector(
        tmp_path,
        parallel_config=types.SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
        ),
        audit_path=audit_path,
    )

    assert connector._compat_decline is not None  # noqa: SLF001
    assert connector._materialization_enabled() is False  # noqa: SLF001

    recipient = FakeRequest("r1", list(range(100)))
    assert connector.get_num_new_matched_tokens(recipient, 0) == (0, False)
    assert connector.stats_snapshot["skipped_incompatible_engine_config"] == 1

    connector.request_finished(FakeRequest("d1", list(range(100))), [0, 1, 2])

    stats_after = connector.stats_snapshot
    assert stats_after.get("donors_registered_total", 0) == 0
    assert not connector._provider._donors  # noqa: SLF001
    # The skip is a decline, not a no-op: it has to land on its own key so the
    # phase-0 join can tell "no donors because the engine is unserveable" from
    # "no donors because nothing finished".
    assert stats_after["donor_registration_skipped_incompatible"] == 1


def test_donors_register_when_the_engine_config_is_serveable(tmp_path) -> None:
    """Control for the guard above: without context parallelism the same
    call registers, so the skip is attributable to the compatibility
    decline and not to a blanket gate on donor capture.
    """
    connector = _connector(tmp_path)

    connector.request_finished(FakeRequest("d1", list(range(100))), [0, 1, 2])

    assert connector.stats_snapshot["donors_registered_total"] == 1
    assert "d1" in connector._provider._donors  # noqa: SLF001


# --------------------------------------------------------------------------
# The startup gate against vLLM's own KV-cache specs
# --------------------------------------------------------------------------


def _spec(kv_cache_interface, **overrides):
    """A plain full-attention spec, built from vLLM's own dataclass."""
    import torch

    fields = {
        "block_size": 4,
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": torch.float16,
        **overrides,
    }
    return kv_cache_interface.FullAttentionSpec(**fields)


def _kv_cache_config(kv_cache_interface, *specs):
    """A real KVCacheConfig carrying one group per spec."""
    return kv_cache_interface.KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            kv_cache_interface.KVCacheGroupSpec(
                layer_names=[f"model.layers.{index}.self_attn.attn"],
                kv_cache_spec=spec,
            )
            for index, spec in enumerate(specs)
        ],
    )


def test_real_full_attention_spec_is_accepted(tmp_path, kv_cache_interface) -> None:
    """The over-declining guard, against the real type: a stock full-attention
    engine must pass the startup gate. If vLLM renames or re-shapes
    ``FullAttentionSpec`` the gate would decline every engine, and the mode
    would go quiet in production while every shim-based test still passed.
    """
    connector = _connector(
        tmp_path, kv_cache_config=_kv_cache_config(kv_cache_interface, _spec(kv_cache_interface))
    )

    assert connector._compat_decline is None  # noqa: SLF001
    assert connector._materialization_enabled() is True  # noqa: SLF001


def test_real_packed_head_slot_spec_declines(tmp_path, kv_cache_interface) -> None:
    """``num_head_slots`` is applied with ``dataclasses.replace`` on the same
    class, so the spec stays literally a ``FullAttentionSpec`` and the class
    check cannot see it. vLLM's ``num_heads`` property is what reports the
    packing; this asserts the gate reads that property off the real spec.
    """
    packed = dataclasses.replace(_spec(kv_cache_interface), num_head_slots=4)
    assert type(packed) is kv_cache_interface.FullAttentionSpec
    assert packed.num_heads != packed.num_kv_heads

    connector = _connector(tmp_path, kv_cache_config=_kv_cache_config(kv_cache_interface, packed))

    assert connector._compat_decline is not None  # noqa: SLF001
    assert "head slots" in connector._compat_decline  # noqa: SLF001


def test_real_sliding_window_spec_declines(tmp_path, kv_cache_interface) -> None:
    """A sliding-window page holds a moving window, not a prefix, so block
    ``p // block_size`` stops holding token ``p``. vLLM's own predicate
    reports it as not-full-attention, and the gate has to act on that.
    """
    torch = pytest.importorskip("torch")
    sliding = kv_cache_interface.SlidingWindowSpec(
        block_size=4, num_kv_heads=8, head_size=128, dtype=torch.float16, sliding_window=512
    )

    connector = _connector(tmp_path, kv_cache_config=_kv_cache_config(kv_cache_interface, sliding))

    assert connector._compat_decline is not None  # noqa: SLF001
    assert "SlidingWindowSpec" in connector._compat_decline  # noqa: SLF001


def test_real_mla_spec_declines_although_vllm_calls_it_full_attention(
    tmp_path, kv_cache_interface
) -> None:
    """The case that only a real type can pose. ``MLAAttentionSpec`` subclasses
    ``FullAttentionSpec`` and vLLM's own ``is_full_attention_spec`` admits it
    on purpose (its docstring names DeepSeek's MLA layers), so a gate that
    stopped at that predicate would accept an engine whose page holds one
    latent and no V at all. The spec says so itself with ``head_size_v = 0``,
    and the gate has to read it.
    """
    torch = pytest.importorskip("torch")
    mla = kv_cache_interface.MLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=576, dtype=torch.float16
    )
    assert kv_cache_interface.is_full_attention_spec(mla) is True
    assert mla.head_size_v == 0

    connector = _connector(tmp_path, kv_cache_config=_kv_cache_config(kv_cache_interface, mla))

    assert connector._compat_decline is not None  # noqa: SLF001
    assert "head_size_v" in connector._compat_decline  # noqa: SLF001


def test_real_hybrid_two_group_config_declines(tmp_path, kv_cache_interface) -> None:
    """A hybrid model publishes one group per attention type. The connector
    indexes ``block_ids[0]``, so a second group makes every destination it
    computes ambiguous; the decline is taken before any spec is inspected and
    is recorded in the audit trail with its reason.
    """
    torch = pytest.importorskip("torch")
    audit_path = tmp_path / "audit.jsonl"
    sliding = kv_cache_interface.SlidingWindowSpec(
        block_size=4, num_kv_heads=8, head_size=128, dtype=torch.float16, sliding_window=512
    )
    connector = _connector(
        tmp_path,
        audit_path=audit_path,
        kv_cache_config=_kv_cache_config(kv_cache_interface, _spec(kv_cache_interface), sliding),
    )

    assert connector._compat_decline is not None  # noqa: SLF001
    assert "2 KV-cache groups" in connector._compat_decline  # noqa: SLF001
    initialized = [e for e in _audit_events(audit_path) if e["event"] == "connector_initialized"]
    assert initialized and initialized[0]["compat_declined_reason"] == connector._compat_decline  # noqa: SLF001


def test_resolve_kv_cache_block_sizes_keeps_the_signature_the_gate_calls(
    vllm_module_source,
) -> None:
    """``_check_hash_block_size`` calls this helper positionally and unpacks two
    ints from it. The module cannot be imported without the full engine runtime
    (see README-compat.md), so the drift check reads vLLM's source: a rename or
    a re-ordered signature turns the gate into a permanent "check incomplete",
    which is a silent loss of the block-size leg rather than a failure.
    """
    import ast

    source = vllm_module_source("vllm.v1.core.kv_cache_utils").read_text(encoding="utf-8")
    functions = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_kv_cache_block_sizes"
    ]

    assert len(functions) == 1, "vLLM no longer defines resolve_kv_cache_block_sizes"
    assert [arg.arg for arg in functions[0].args.args] == ["kv_cache_config", "vllm_config"]
    assert ast.unparse(functions[0].returns) == "tuple[int, int]"

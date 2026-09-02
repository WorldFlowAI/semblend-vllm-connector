"""Composed worker-path oracle check (take-15 arm A geometry).

Simulates the full chain with REAL numbers: donor K rotated at donor
positions by an INDEPENDENT oracle implementing vLLM's neox convention
(transcribed from vllm/model_executor/layers/rotary_embedding native
forward, not from our helpers), written through the connector's extract
contract, then slice -> rerotate -> inject at mid-request target slots,
and finally compared against the oracle's K at target positions.
"""

import torch

from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import MaterializationKind, PendingLoad

HEADS, HEAD_DIM, THETA = 4, 128, 1_000_000.0
BLOCK = 16


def oracle_rotate(k_raw, positions):
    """vLLM RotaryEmbedding.forward_native, neox style, transcribed."""
    inv_freq = 1.0 / (
        THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = freqs.cos(), freqs.sin()
    x1 = k_raw[..., : HEAD_DIM // 2]
    x2 = k_raw[..., HEAD_DIM // 2 :]
    c = cos[:, None, :]
    s = sin[:, None, :]
    # neox: out = x*cos + rotate_neox(x)*sin, rotate_neox = cat(-x2, x1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def test_composed_worker_path_matches_vllm_convention_oracle():
    torch.manual_seed(7)
    donor_tokens = 2048
    donor_start, target_start, span = 779, 2048, 1264

    connector = SemBlendVllmConnector.__new__(SemBlendVllmConnector)
    connector._stats = {
        "semantic_span_declined_mla": 0,
        "semantic_span_declined_no_rope_params": 0,
        "semantic_span_declined_short_donor": 0,
    }

    class _HF:
        rope_parameters = {"rope_theta": THETA, "rope_type": "default"}
        hidden_size = HEADS * HEAD_DIM * 7  # num_attention_heads*head_dim
        num_attention_heads = HEADS * 7  # GQA: 28 q heads, 4 kv heads

    class _MC:
        hf_config = _HF()

    class _VC:
        model_config = _MC()

    connector._vllm_config = _VC()
    connector._block_size = BLOCK

    # Donor K raw, rotated at donor absolute positions [0..2048).
    k_raw = torch.randn(donor_tokens, HEADS, HEAD_DIM, dtype=torch.float32)
    v_ref = torch.randn(donor_tokens, HEADS, HEAD_DIM, dtype=torch.float32)
    k_donor = oracle_rotate(k_raw, torch.arange(0, donor_tokens))

    # Extract-contract file layout: [2, n, H*D] flattened.
    src_kv = torch.stack(
        (k_donor.reshape(donor_tokens, -1), v_ref.reshape(donor_tokens, -1))
    )

    load = PendingLoad(
        request_id="r",
        donor_id="d",
        token_count=span,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns",
        block_ids=([list(range(128, 128 + (span + BLOCK - 1) // BLOCK))],)[0],
        donor_start=donor_start,
        target_start=target_start,
    )

    out = connector._semantic_span_slice(src_kv, load, attn_metadata=object())
    assert out is not None, (
        f"slice declined: {connector._stats}"
    )

    # Inject into a paged kv_first layer and read back at the load slots.
    n_blocks = (span + BLOCK - 1) // BLOCK + 130
    dst = torch.zeros(2, n_blocks, BLOCK, HEADS, HEAD_DIM)
    slot_mapping = connector._slot_mapping(load.block_ids, span, dst.device)
    connector._inject_kv_into_layer(dst, out, slot_mapping, object())
    back = connector._extract_kv_from_layer(dst, slot_mapping, object())

    k_mat = back[0].reshape(span, HEADS, HEAD_DIM)
    v_mat = back[1].reshape(span, HEADS, HEAD_DIM)

    # Oracle: same raw K, rotated at TARGET absolute positions.
    expected_k = oracle_rotate(
        k_raw[donor_start : donor_start + span],
        torch.arange(target_start, target_start + span),
    )
    expected_v = v_ref[donor_start : donor_start + span]

    k_err = (k_mat - expected_k).abs().max().item()
    v_err = (v_mat - expected_v).abs().max().item()
    print(f"K max-abs error vs vLLM-convention oracle: {k_err:.3e}")
    print(f"V max-abs error: {v_err:.3e}")
    assert k_err < 1e-3, "K diverges from the vLLM-convention oracle"
    assert v_err == 0.0, "V must move unchanged"

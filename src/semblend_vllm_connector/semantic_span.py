"""Block-aligned semantic-span planning (SEMANTIC_SPAN reuse mode).

Token-level alignment produces donor spans at arbitrary offsets; vLLM
allocates and loads external KV at block granularity. Spans are snapped
inward to block edges in the target frame with the donor start advanced
identically (token identity preserved), and the supply decision is
boundary-anchored per the get_num_new_matched_tokens contract: report the
block-aligned contiguous run servable from the scheduler's computed-token
boundary, or zero when the boundary sits in a novel region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class BlockSpan:
    """A donor span snapped to block edges in the target frame."""

    target_start: int  # inclusive, block-aligned
    target_end: int  # exclusive, block-aligned
    donor_start: int  # donor-side position of target_start's token


def block_align_spans(
    spans: Sequence[dict],
    block_size: int,
    min_span: int,
) -> list[BlockSpan]:
    """Snap raw spans inward to block edges; fold short ones; drop overlaps.

    Each raw span is a mapping with target_start, length, donor_start
    (token-identity-verified alignment output). Later spans overlapping an
    earlier snapped span are dropped, mirroring the plan builder's safe
    resolution.
    """
    out: list[BlockSpan] = []
    cursor = 0
    for raw in sorted(spans, key=lambda s: s["target_start"]):
        t0 = int(raw["target_start"])
        t1 = t0 + int(raw["length"])
        snapped_start = ((t0 + block_size - 1) // block_size) * block_size
        snapped_end = (t1 // block_size) * block_size
        if snapped_start < cursor:
            continue
        if snapped_end - snapped_start < max(min_span, 1):
            continue
        out.append(
            BlockSpan(
                target_start=snapped_start,
                target_end=snapped_end,
                donor_start=int(raw["donor_start"]) + (snapped_start - t0),
            )
        )
        cursor = snapped_end
    return out


def supply_at_boundary(
    spans: Sequence[BlockSpan],
    boundary: int,
    block_size: int,
) -> tuple[int, Optional[int]]:
    """Boundary-anchored supply decision.

    Returns (num_tokens, donor_start) for the contiguous block-aligned run
    servable from ``boundary``, or (0, None) when the boundary is not
    inside any donor span. An unaligned boundary snaps the usable start up
    to the next block edge; the donor start advances by the tokens skipped.
    """
    usable_from = ((boundary + block_size - 1) // block_size) * block_size
    for span in spans:
        if span.target_start <= usable_from < span.target_end:
            offset = usable_from - span.target_start
            return span.target_end - usable_from, span.donor_start + offset
    return 0, None


def rope_cos_sin(positions, head_dim: int, rope_theta: float):
    """Cos/sin tables for the given absolute positions (neox half-split)."""
    import torch

    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0, head_dim, 2, dtype=torch.float32, device=positions.device
            )
            / head_dim
        )
    )
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


def rerotate_k(
    k,
    donor_start: int,
    target_start: int,
    head_dim: int,
    rope_theta: float,
):
    """Re-rotate cached K from donor positions to target positions.

    RoPE rotations compose, so rotating by (target - donor) maps a key
    cached at donor position p + delta exactly onto the key for target
    position p. V carries no positional encoding and is never touched.
    K is [tokens, heads, head_dim], neox half-split, full rotary width.
    """
    import torch

    if donor_start == target_start:
        return k
    n = k.shape[0]
    delta = target_start - donor_start
    # Rotation by a constant delta: the angle depends on the frequency
    # only, applied uniformly across tokens.
    dpos = torch.full((n,), float(delta), device=k.device)
    cos, sin = rope_cos_sin(dpos, head_dim, rope_theta)
    orig_dtype = k.dtype
    kf = k.to(torch.float32)
    c = cos[:, None, :]
    s = sin[:, None, :]
    x1, x2 = kf[..., : head_dim // 2], kf[..., head_dim // 2 :]
    out = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    return out.to(orig_dtype)

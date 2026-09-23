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

    The returned count is measured from ``usable_from``, NOT from
    ``boundary``. A caller that adds it to ``boundary`` (as vLLM's scheduler
    does with what the match hook returns) is only correct when the two
    coincide, i.e. when ``boundary`` is already block-aligned; off an aligned
    boundary the tokens in between would be counted as computed and never
    written.
    """
    usable_from = ((boundary + block_size - 1) // block_size) * block_size
    for span in spans:
        if span.target_start <= usable_from < span.target_end:
            offset = usable_from - span.target_start
            return span.target_end - usable_from, span.donor_start + offset
    return 0, None


def chain_at_boundary(
    runs: Sequence[dict],
    boundary: int,
    block_size: int,
) -> list[dict]:
    """The longest contiguous span from the boundary, assembled from any donors.

    ``runs`` are token-identity-verified (donor_id, donor_start, target_start,
    length) runs, possibly from several donors. From the block-aligned
    boundary the chain repeatedly takes the run covering the current position
    that reaches furthest -- the greedy interval cover, which maximises the
    contiguous reach -- so pieces from different donors meet at arbitrary
    tokens; only the chain's two outer ends are block-aligned, because only
    those are what vLLM allocates and counts. Returns the pieces in target
    order, or an empty list when no run covers the boundary.
    """
    position = ((boundary + block_size - 1) // block_size) * block_size
    pieces: list[dict] = []
    while True:
        best = None
        for run in runs:
            start = int(run["target_start"])
            end = start + int(run["length"])
            if start <= position < end and (best is None or end > best[1]):
                best = (run, end)
        if best is None:
            break
        run, end = best
        offset = position - int(run["target_start"])
        pieces.append(
            {
                "donor_id": run["donor_id"],
                "donor_start": int(run["donor_start"]) + offset,
                "target_start": position,
                "token_count": end - position,
            }
        )
        position = end
    return trim_pieces(pieces, (position // block_size) * block_size)


def trim_pieces(pieces: Sequence[dict], end: int) -> list[dict]:
    """``pieces`` cut off at target position ``end``; empty pieces dropped."""
    out = []
    for piece in pieces:
        count = min(int(piece["token_count"]), end - int(piece["target_start"]))
        if count > 0:
            out.append({**piece, "token_count": count})
    return out


def rope_cos_sin(positions, head_dim: int, rope_theta: float):
    """Cos/sin tables for the given absolute positions (neox half-split)."""
    import torch

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)
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

"""No lookup when the tail left to serve is shorter than the span floor.

Measured on stock vLLM: verbatim re-issues hit the engine's own exact prefix
cache for 3408 of 3411 tokens. The connector was configured not to stand down
on the exact-prefix ratio, so it ran the whole lookup -- an embedding plus a
~26 ms provider round trip -- for the three tokens that were left, on every
repeat, and every one of them ended in ``semantic_span_boundary_missed``.

Three tokens can never be served: ``block_align_spans`` drops anything below
``min_semantic_span``, and so does the floor re-applied after the clamps. The
decision is therefore already made before the lookup runs, and this pins the
connector making it there.
"""

from __future__ import annotations

import json

from test_connector_discovery import (
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import (
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
)

BLOCK_SIZE = 4
MIN_SPAN = 8
PROMPT_TOKENS = 100


class _CountingProvider:
    """Records every lookup that reaches it; always answers with a span."""

    def __init__(self) -> None:
        self.lookups: list = []

    def lookup(self, request):
        self.lookups.append(request)
        return SemanticLookupResult(
            donor_id="d1",
            similarity=0.99,
            reusable_token_count=80,
            materialization_kind=MaterializationKind.SEMANTIC_SPAN,
            segments=[
                SemanticSegment(donor_id="d1", donor_start=210, target_start=10, token_count=80)
            ],
        )

    def register_donor(self, donor) -> None:
        return None

    def clear_donors(self) -> None:
        return None


def _connector(
    tmp_path,
    *,
    mode: str = "semantic_span_experimental",
    min_span: int = MIN_SPAN,
    block_size: int = BLOCK_SIZE,
) -> SemBlendVllmConnector:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": mode,
                "provider": "local",
                "min_prompt_tokens": min_span,
                "min_similarity": 0.3,
                "min_semantic_span": min_span,
                # The measured arm's setting: stand down only on a whole-prompt
                # exact prefix, so the ratio gate cannot be what skips here.
                "skip_when_exact_prefix_ratio_at_least": 1.0,
                "kv_storage_path": str(tmp_path / "kv"),
                "audit_path": str(tmp_path / "audit.jsonl"),
            }
        ),
        cache_config=FakeCacheConfig(block_size=block_size),
    )
    return SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)


def _audit(tmp_path, event: str) -> list[dict]:
    with open(tmp_path / "audit.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row for row in rows if row["event"] == event]


def test_remaining_equal_to_the_floor_still_runs_the_lookup(tmp_path) -> None:
    """The boundary case is inclusive: exactly one floor's worth is servable."""
    connector = _connector(tmp_path)
    provider = _CountingProvider()
    connector._provider = provider  # noqa: SLF001

    connector.get_num_new_matched_tokens(
        FakeRequest("r1", list(range(PROMPT_TOKENS))), PROMPT_TOKENS - MIN_SPAN
    )

    assert len(provider.lookups) == 1
    assert _audit(tmp_path, "lookup_skipped_remaining_below_min_span") == []


def test_one_token_below_the_floor_skips_the_lookup(tmp_path) -> None:
    """One token less and the lookup cannot end in a load, so it is not run."""
    connector = _connector(tmp_path)
    provider = _CountingProvider()
    connector._provider = provider  # noqa: SLF001
    boundary = PROMPT_TOKENS - MIN_SPAN + 1

    matched, async_load = connector.get_num_new_matched_tokens(
        FakeRequest("r1", list(range(PROMPT_TOKENS))), boundary
    )

    assert (matched, async_load) == (0, False)
    assert provider.lookups == []

    (event,) = _audit(tmp_path, "lookup_skipped_remaining_below_min_span")
    assert event["remaining"] == MIN_SPAN - 1
    assert event["min_semantic_span"] == MIN_SPAN
    assert event["boundary"] == boundary
    assert event["prompt_tokens"] == PROMPT_TOKENS
    assert event["block_size"] == BLOCK_SIZE
    assert event["attempt"] == 0
    assert connector.stats_snapshot["skipped_remaining_below_min_span_per_attempt"] == 1


def test_the_measured_repeat_skips_instead_of_embedding(tmp_path) -> None:
    """The run this gate comes from: 3408 of 3411 tokens already computed."""
    connector = _connector(tmp_path, min_span=512, block_size=16)
    provider = _CountingProvider()
    connector._provider = provider  # noqa: SLF001

    connector.get_num_new_matched_tokens(FakeRequest("r1", list(range(3411))), 3408)

    assert provider.lookups == []
    (event,) = _audit(tmp_path, "lookup_skipped_remaining_below_min_span")
    assert (event["remaining"], event["min_semantic_span"]) == (3, 512)


def test_the_gate_is_scoped_to_the_mode_that_has_the_floor(tmp_path) -> None:
    """``min_semantic_span`` gates span planning and nothing else.

    In the other modes there is no floor for a short tail to fall under, so a
    gate that read it would decline work those modes can still serve.
    """
    connector = _connector(tmp_path, mode="request_only_experimental")
    provider = _CountingProvider()
    connector._provider = provider  # noqa: SLF001

    connector.get_num_new_matched_tokens(
        FakeRequest("r1", list(range(PROMPT_TOKENS))), PROMPT_TOKENS - 1
    )

    assert len(provider.lookups) == 1
    assert _audit(tmp_path, "lookup_skipped_remaining_below_min_span") == []


def test_the_skip_is_counted_on_every_attempt(tmp_path) -> None:
    """The boundary moves between attempts, so the skip is not memoized.

    Gating it on the first attempt would hide every later decline of a request
    that was admitted to the lookup at an earlier boundary.
    """
    connector = _connector(tmp_path)
    connector._provider = _CountingProvider()  # noqa: SLF001
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))

    connector.get_num_new_matched_tokens(request, PROMPT_TOKENS - 1)
    connector.get_num_new_matched_tokens(request, PROMPT_TOKENS - 2)

    events = _audit(tmp_path, "lookup_skipped_remaining_below_min_span")
    assert [event["attempt"] for event in events] == [0, 1]
    assert connector.stats_snapshot["skipped_remaining_below_min_span_per_attempt"] == 2

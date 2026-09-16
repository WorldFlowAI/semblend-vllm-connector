"""The text a lookup embeds starts at the exact-prefix boundary, not at token 0.

Measured in phase-0 (2026-09-14, stock vLLM 0.29, 32K-token prompts): 228 of
231 lookups missed although every recipient carried the same document as the
donor it should have matched. The recipients were that document behind a
512-token operator preamble, and the provider embeds the *head* of the prompt
text -- a MiniLM sentence encoder truncates to roughly its first 256 tokens,
and SemBlend caps the text by characters on top of that. The whole embedded
window was therefore preamble: every recipient embedded alike, and none of them
embedded like its donor.

These tests model that with an embedder that is only its truncation: a provider
whose "embedding" is the first ``_EMBED_WINDOW_CHARS`` characters of whatever
text it is handed. Nothing else about the run has to be reproduced for the miss
to appear, which is the point.
"""

from __future__ import annotations

import json
import os

from test_connector_discovery import (
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeSchedulerOutput,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.namespace import namespace_for_request

# One token decodes to one fixed-width word, so a prefix of N tokens is exactly
# N * _WORD_CHARS characters of prompt text and every offset in these tests is
# checkable by hand.
_WORD_CHARS = 4
_EMBED_WINDOW_CHARS = 64

_PREAMBLE_TOKENS = list(range(1, 129))
_DOCUMENT_TOKENS = list(range(200, 500))


def _word(token_id: int) -> str:
    return f"w{token_id:03d}"


def _text(token_ids: list[int]) -> str:
    return "".join(_word(t) for t in token_ids)


class _WordTokenizer:
    """Decodes each token id to its own fixed-width word."""

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return _text(list(token_ids))


class _WindowEmbedProvider:
    """A provider whose whole embedding is the head of the text it is given.

    This is the failure mode under test with everything else removed: a donor
    and a recipient match when, and only when, the first
    ``_EMBED_WINDOW_CHARS`` characters of their texts agree.
    """

    def __init__(self) -> None:
        self.donors: dict[str, str] = {}
        self.queries: list[str | None] = []

    @staticmethod
    def _window(text: str | None) -> str:
        return (text or "")[:_EMBED_WINDOW_CHARS]

    def register_donor(self, donor) -> None:
        self.donors[donor.donor_id] = self._window(donor.prompt_text)

    def lookup(self, request):
        self.queries.append(request.prompt_text)
        window = self._window(request.prompt_text)
        for donor_id, donor_window in self.donors.items():
            if donor_window == window:
                from semblend_vllm_connector.types import (
                    MaterializationKind,
                    SemanticLookupResult,
                )

                return SemanticLookupResult(
                    donor_id=donor_id,
                    similarity=1.0,
                    reusable_token_count=0,
                    materialization_kind=MaterializationKind.DISCOVERY_ONLY,
                    reason="window_match",
                )
        return None

    def clear_donors(self) -> None:
        self.donors.clear()


def _connector(tmp_path, **extra):
    cfg = {
        "mode": "discovery_only",
        "provider": "local",
        "min_prompt_tokens": 64,
        "enable_prompt_text": True,
        # Off, so the boundary these tests set is never taken as "the engine
        # has already computed enough" before the lookup runs.
        "skip_when_exact_prefix_ratio_at_least": 1.0,
        "audit_path": str(tmp_path / "audit.jsonl"),
        "kv_storage_path": str(tmp_path / "kv"),
    }
    cfg.update(extra)
    connector = SemBlendVllmConnector(
        FakeVllmConfig(FakeKvTransferConfig(cfg), cache_config=FakeCacheConfig(block_size=16)),
        KVConnectorRole.SCHEDULER,
    )
    provider = _WindowEmbedProvider()
    connector._provider = provider  # noqa: SLF001
    connector._prompt_tokenizer = _WordTokenizer()  # noqa: SLF001
    return connector, provider


def _events(tmp_path, name: str) -> list[dict]:
    path = tmp_path / "audit.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row["event"] == name]


def _write_donor_record(connector, request, store) -> None:
    """The record the worker publishes once a donor's layers are durable.

    A donor is registered only after this file exists, so a scheduler-side
    test that opens a capture has to stand up the worker's half of it.
    """
    namespace = namespace_for_request(connector._config, connector._vllm_config, request)  # noqa: SLF001
    os.makedirs(connector._donor_dir(store.request_id, namespace), exist_ok=True)  # noqa: SLF001
    path = connector._donor_metadata_path(store.request_id, namespace)  # noqa: SLF001
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"token_count": int(store.token_count)}, f)


def _seed_and_recipient(connector):
    """Register the document as a donor, then look up the wrapped copy of it."""
    seed = FakeRequest("seed", list(_DOCUMENT_TOKENS))
    connector.request_finished(seed, [0])
    recipient = FakeRequest("r1", [*_PREAMBLE_TOKENS, *_DOCUMENT_TOKENS])
    return connector.get_num_new_matched_tokens(recipient, len(_PREAMBLE_TOKENS))


def test_head_embedding_misses_the_donor_it_duplicates(tmp_path) -> None:
    """0.2.4's behaviour, kept reachable by the knob: the wrapper is embedded.

    The recipient holds the donor's document verbatim, and the lookup still
    misses, because the only part of the prompt that reaches the embedder is
    the preamble in front of it.
    """
    connector, provider = _connector(tmp_path, boundary_sliced_query_text=False)

    _seed_and_recipient(connector)

    assert connector.stats_snapshot["semantic_misses_total"] == 1
    assert connector.stats_snapshot.get("semantic_hits_total", 0) == 0
    # What was embedded: the preamble, in full, from token 0.
    assert provider.queries == [_text([*_PREAMBLE_TOKENS, *_DOCUMENT_TOKENS])]
    miss = _events(tmp_path, "semantic_lookup_miss")[0]
    assert miss["query_text_offset_tokens"] == 0
    assert miss["query_text_fallback"] == "disabled_by_config"


def test_boundary_embedding_hits_the_donor_it_duplicates(tmp_path) -> None:
    """The same run with the query text sliced at the boundary hits."""
    connector, provider = _connector(tmp_path)

    _seed_and_recipient(connector)

    assert connector.stats_snapshot["semantic_hits_total"] == 1
    assert connector.stats_snapshot["query_text_sliced_total"] == 1
    # What was embedded: the document, starting exactly where the preamble ends.
    assert provider.queries == [_text(_DOCUMENT_TOKENS)]
    hit = _events(tmp_path, "semantic_lookup_hit")[0]
    assert hit["query_text_offset_tokens"] == len(_PREAMBLE_TOKENS)
    assert hit["query_text_chars"] == len(_DOCUMENT_TOKENS) * _WORD_CHARS
    assert hit["query_text_fallback"] is None


def test_a_slice_below_the_floor_falls_back_to_the_whole_prompt(tmp_path) -> None:
    """A boundary near the prompt end leaves too little text to identify anything.

    The fallback is to the full prompt text -- which is what 0.2.4 always sent
    -- and it is named in the audit, so a run is never left claiming it
    embedded a slice it did not embed.
    """
    connector, provider = _connector(tmp_path, min_query_text_chars=10_000)

    _seed_and_recipient(connector)

    assert connector.stats_snapshot["query_text_fallback_below_min_chars"] == 1
    assert connector.stats_snapshot.get("query_text_sliced_total", 0) == 0
    assert provider.queries == [_text([*_PREAMBLE_TOKENS, *_DOCUMENT_TOKENS])]
    miss = _events(tmp_path, "semantic_lookup_miss")[0]
    assert miss["query_text_fallback"] == "below_min_chars"
    assert miss["query_text_offset_tokens"] == 0


def test_without_a_tokenizer_the_slice_falls_back_and_says_so(tmp_path) -> None:
    """No decode, no character offset; the audit carries the reason.

    The prompt text itself can still arrive (vLLM carries the string on the
    request), so this path is reachable without the lookup losing its text
    entirely -- which is why it is a fallback and not a skip.
    """
    connector, provider = _connector(tmp_path)
    connector._prompt_tokenizer = None  # noqa: SLF001
    connector._prompt_tokenizer_failed = True  # noqa: SLF001

    seed = FakeRequest("seed", list(_DOCUMENT_TOKENS))
    seed.prompt = _text(_DOCUMENT_TOKENS)
    connector.request_finished(seed, [0])
    recipient = FakeRequest("r1", [*_PREAMBLE_TOKENS, *_DOCUMENT_TOKENS])
    recipient.prompt = _text([*_PREAMBLE_TOKENS, *_DOCUMENT_TOKENS])
    connector.get_num_new_matched_tokens(recipient, len(_PREAMBLE_TOKENS))

    assert connector.stats_snapshot["query_text_fallback_offset_unavailable"] == 1
    assert provider.queries == [recipient.prompt]
    miss = _events(tmp_path, "semantic_lookup_miss")[0]
    assert miss["query_text_fallback"] == "offset_unavailable"


def test_a_seed_is_registered_with_its_whole_text(tmp_path) -> None:
    """No wrapper, no boundary: a donor admitted at 0 keeps its full text."""
    connector, provider = _connector(tmp_path)

    connector.request_finished(FakeRequest("seed", list(_DOCUMENT_TOKENS)), [0])

    assert provider.donors["seed"] == _text(_DOCUMENT_TOKENS)[:_EMBED_WINDOW_CHARS]
    registered = _events(tmp_path, "donor_registered")[0]
    assert registered["donor_text_offset_tokens"] == 0
    assert registered["donor_text_fallback"] is None


def test_a_wrapped_donor_is_registered_from_its_own_boundary(tmp_path) -> None:
    """A donor behind a wrapper registers the text after it, not the wrapper.

    Otherwise the fix is one-sided: a recipient would embed its document while
    every donor that arrived wrapped still embedded its wrapper, and the two
    sides would keep missing each other for the original reason.
    """
    connector, provider = _connector(
        tmp_path,
        mode="semantic_span_experimental",
        min_semantic_span=64,
    )
    donor_tokens = [*_PREAMBLE_TOKENS, *_DOCUMENT_TOKENS]

    # The one hook that hands the connector a request's own exact-prefix hit.
    new_req = FakeRequest("d1", list(donor_tokens))
    new_req.req_id = "d1"
    new_req.block_ids = ([0, 1, 2, 3, 4, 5, 6, 7],)
    new_req.num_computed_tokens = len(_PREAMBLE_TOKENS)
    metadata = connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[new_req]))
    assert connector._capture_boundaries["d1"] == len(_PREAMBLE_TOKENS)  # noqa: SLF001
    # The worker's half: a donor is registered only once its capture is
    # readable, so the record the worker writes has to exist by now.
    _write_donor_record(connector, new_req, metadata.stores[0])

    connector.request_finished(FakeRequest("d1", list(donor_tokens)), [0])

    assert provider.donors["d1"] == _text(_DOCUMENT_TOKENS)[:_EMBED_WINDOW_CHARS]
    registered = _events(tmp_path, "donor_registered")[0]
    assert registered["donor_text_offset_tokens"] == len(_PREAMBLE_TOKENS)
    assert registered["donor_text_chars"] == len(_DOCUMENT_TOKENS) * _WORD_CHARS
    # And the boundary does not outlive the request that was admitted at it.
    assert "d1" not in connector._capture_boundaries  # noqa: SLF001

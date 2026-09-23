"""The lookup is skipped only when the provider rules out every servable span."""

from __future__ import annotations

import json

from test_connector_discovery import FakeRequest
from test_lookup_remaining_span_gate import BLOCK_SIZE, MIN_SPAN, _connector, _CountingProvider

PROMPT_TOKENS = 100


class _PrecheckingProvider(_CountingProvider):
    def __init__(self, answer) -> None:
        super().__init__()
        self.answer = answer
        self.prechecks: list[tuple] = []

    def span_run_possible(self, token_ids, namespace, start, length):
        self.prechecks.append((len(token_ids), namespace, start, length))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _audit(tmp_path, event: str) -> list[dict]:
    with open(tmp_path / "audit.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row for row in rows if row["event"] == event]


def _run(tmp_path, provider, boundary=10, **kwargs):
    connector = _connector(tmp_path, **kwargs)
    connector._provider = provider  # noqa: SLF001
    out = connector.get_num_new_matched_tokens(
        FakeRequest("r1", list(range(PROMPT_TOKENS))), boundary
    )
    return connector, out


def test_ruled_out_skips_the_lookup_and_says_why(tmp_path) -> None:
    provider = _PrecheckingProvider(False)
    connector, out = _run(tmp_path, provider, boundary=10)

    assert out == (0, False)
    assert provider.lookups == []
    # Asked from the block-aligned boundary, for one floor's worth.
    assert provider.prechecks[0][2:] == (12, MIN_SPAN)
    (event,) = _audit(tmp_path, "lookup_skipped_no_shared_run")
    assert (event["boundary"], event["usable_from"], event["min_semantic_span"]) == (
        10,
        12,
        MIN_SPAN,
    )
    assert connector.stats_snapshot["skipped_no_shared_run_per_attempt"] == 1
    assert BLOCK_SIZE == 4


def test_possible_runs_the_lookup(tmp_path) -> None:
    provider = _PrecheckingProvider(True)
    _run(tmp_path, provider)
    assert len(provider.lookups) == 1
    assert _audit(tmp_path, "lookup_skipped_no_shared_run") == []


def test_cannot_say_runs_the_lookup(tmp_path) -> None:
    provider = _PrecheckingProvider(None)
    _run(tmp_path, provider)
    assert len(provider.lookups) == 1


def test_a_failing_precheck_is_a_maybe(tmp_path) -> None:
    provider = _PrecheckingProvider(RuntimeError("boom"))
    connector, _ = _run(tmp_path, provider)
    assert len(provider.lookups) == 1
    assert connector.stats_snapshot["lookup_precheck_errors_total"] == 1


def test_a_provider_without_the_method_is_unchanged(tmp_path) -> None:
    provider = _CountingProvider()
    _run(tmp_path, provider)
    assert len(provider.lookups) == 1


def test_only_semantic_span_mode_prechecks(tmp_path) -> None:
    provider = _PrecheckingProvider(False)
    _run(tmp_path, provider, mode="request_only_experimental")
    assert provider.prechecks == []

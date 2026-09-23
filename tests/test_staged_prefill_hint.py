"""A stage of a staged prefill keeps its filled blocks for the stage after it.

Staged prefill runs a prompt with an edit in it as a sequence of prefixes: the
connector serves the shared text up to the edit, the engine computes the edit,
and the next stage -- the same prompt, longer -- finds all of that in vLLM's
prefix cache and has its boundary past the edit, where the connector can serve
the shared text that follows. That only works if the stage's filled blocks are
still in the cache, so a stage is exempt from the B6 eviction. Nothing else is.
"""

from __future__ import annotations

from test_b6_contamination import (
    PROMPT_TOKENS,
    _admission_step,
    _audit_events,
    _bound_connector,
    _serve,
    _write_donor_capture,
)
from test_connector_discovery import FakeRequest


def _run(tmp_path, *, staged: bool, **overrides):
    connector, pool = _bound_connector(tmp_path, **overrides)
    request = FakeRequest("r1", list(range(PROMPT_TOKENS)))
    if staged:
        request.semblend_stage = "1"
    _write_donor_capture(connector, request)
    table = (list(range(100, 200)),)
    pool.cache_blocks(table[0])
    _serve(connector, request, 12, table)
    connector.build_connector_meta(_admission_step("r1", request.all_token_ids, table, 12))
    return connector, pool


def test_a_stage_keeps_its_filled_blocks(tmp_path) -> None:
    connector, pool = _run(tmp_path, staged=True)
    assert pool.evicted_since() == set()
    assert connector.stats_snapshot.get("prefix_cache_blocks_evicted", 0) == 0
    (event,) = [
        e for e in _audit_events(tmp_path) if e["event"] == "prefix_cache_blocks_left_cached"
    ]
    assert event["staged_prefill"] is True


def test_an_unhinted_request_is_still_evicted(tmp_path) -> None:
    connector, pool = _run(tmp_path, staged=False)
    assert pool.evicted_since() == set(range(103, 200))


def test_the_hint_key_is_configurable(tmp_path) -> None:
    _, pool = _run(tmp_path, staged=True, stage_hint_key="other_key")
    assert pool.evicted_since() == set(range(103, 200))


def test_the_exemption_ends_with_the_request(tmp_path) -> None:
    connector, _ = _run(tmp_path, staged=True)
    connector.request_finished(FakeRequest("r1", list(range(PROMPT_TOKENS))), [])
    assert "r1" not in connector._stage_requests  # noqa: SLF001


def test_a_stage_is_never_captured_under_any_policy(tmp_path) -> None:
    """A stage is a prefix of the request after it: as a donor it would outrank
    the real one for that request and supply nothing past its own end."""
    import types

    from test_deferred_capture_write import _captured_ids, _config, _events, _scheduled

    from semblend_vllm_connector._vllm_compat import KVConnectorRole
    from semblend_vllm_connector.connector import SemBlendVllmConnector

    stage = _scheduled(
        "stage", sampling_params=types.SimpleNamespace(extra_args={"semblend_stage": "1"})
    )
    for policy in ("all", "hinted"):
        root = tmp_path / policy
        connector = SemBlendVllmConnector(
            _config(root, capture_policy=policy), KVConnectorRole.SCHEDULER
        )
        hinted_too = _scheduled(
            "both",
            sampling_params=types.SimpleNamespace(
                extra_args={"semblend_stage": "1", "semblend_capture": "1"}
            ),
        )
        assert _captured_ids(connector, [stage, hinted_too]) == []
        reasons = {row["reason"] for row in _events(root, "capture_skipped")}
        assert reasons == {"staged_prefill_stage"}


def test_the_memory_tier_keeps_pageable_copies() -> None:
    import torch

    from semblend_vllm_connector.connector import _pageable

    plain = torch.ones(3)
    assert _pageable(plain) is plain

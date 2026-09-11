"""Every decline path lands on a named counter, exactly as often as it happens.

Phase-0 reads these counters as per-request joins: a decline that is silent
disappears from the denominator, and one that fires per scheduling attempt
inflates it. So each test here pins two things a rename or a refactor could
break independently — the counter's exact key, and how many times one occurrence
of the event increments it.

Counters whose key ends in ``_per_attempt`` are deliberately attempt-scoped:
they are decided from ``num_computed_tokens``, which moves between scheduling
attempts, so the same request can legitimately decline more than once and the
suffix warns the metrics side not to join them per request. Those tests call
the hook twice and assert two.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest
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
from semblend_vllm_connector.types import (
    MaterializationKind,
    PendingLoad,
    SemanticBlockRef,
    SemanticLookupResult,
    SemanticSegment,
)

BLOCK_SIZE = 4

# The two vLLM modules the startup gate imports. Tests that exercise an
# incomplete check force their absence rather than depending on whether this
# machine happens to have vLLM installed, so they assert the same counter in
# both environments (see tests/README-compat.md).
GATE_IMPORTS = ("vllm.v1.kv_cache_interface", "vllm.v1.core.kv_cache_utils")


def _connector(tmp_path, *, kv_cache_config=None, parallel_config=None, **overrides):
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
    settings.update(overrides)
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(settings),
        cache_config=FakeCacheConfig(block_size=BLOCK_SIZE),
    )
    if parallel_config is not None:
        # FakeVllmConfig mirrors only the fields the connector reads; the
        # startup gate is the only caller that consults this one.
        vllm_config.parallel_config = parallel_config
    return SemBlendVllmConnector(
        vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config=kv_cache_config
    )


def _assert_counted(stats, key, expected) -> None:
    """Assert one counter's exact value, printing the whole snapshot on failure.

    The likely failure here is a renamed key, and the snapshot names the
    replacement instead of leaving a reader with a bare missing-key error.
    """
    assert stats.get(key, 0) == expected, (key, expected, stats)


class _ScriptedProvider:
    """Provider stub: one fixed alignment result per lookup."""

    def __init__(self, result):
        self.result = result

    def lookup(self, request):
        return self.result

    def register_donor(self, registration):
        return True


def _span_result(*, donor_start=210, target_start=10, token_count=80):
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.99,
        reusable_token_count=token_count,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        segments=[
            SemanticSegment(
                donor_id="d1",
                donor_start=donor_start,
                target_start=target_start,
                token_count=token_count,
            )
        ],
    )


def _exact_prefix_result(reusable_tokens):
    return SemanticLookupResult(
        donor_id="d1",
        similarity=0.99,
        reusable_token_count=reusable_tokens,
        materialization_kind=MaterializationKind.EXACT_PREFIX,
        block_refs=[SemanticBlockRef(block_id="b0", start_token=0, token_count=reusable_tokens)],
    )


def _write_donor_capture(connector, request, donor_id, token_count) -> None:
    """Record the donor's captured length where the advertise path reads it."""
    namespace = namespace_for_request(
        connector._config, connector._vllm_config, request  # noqa: SLF001
    )
    os.makedirs(connector._donor_dir(donor_id, namespace), exist_ok=True)  # noqa: SLF001
    with open(connector._donor_metadata_path(donor_id, namespace), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": token_count}, f)


def _serving_connector(tmp_path, result, *, request, **overrides):
    connector = _connector(tmp_path, **overrides)
    connector._provider = _ScriptedProvider(result)  # noqa: SLF001
    _write_donor_capture(connector, request, "d1", token_count=4096)
    return connector


def _group_stub(**spec_fields):
    """A KV-cache config shaped like the engine's, with a stand-in spec.

    The spec's identity only matters to the startup gate, which these tests do
    not assert on; what matters is that the group list is non-empty so the gate
    runs past the unresolved-config branch.
    """
    return types.SimpleNamespace(
        kv_cache_groups=[
            types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(**spec_fields))
        ]
    )


class _BlocklessAllocation:
    """What the scheduler hands back when it allocated nothing for the load."""

    def get_block_ids(self, allow_none: bool = False):
        return None if allow_none else ()


# --------------------------------------------------------------------------
# Startup gate: checks that declined, and checks that could not run
# --------------------------------------------------------------------------


def test_unresolved_kv_cache_config_counts_once_per_connector(tmp_path) -> None:
    """The gate cannot inspect groups or specs without a KV-cache config, and a
    skipped check must not read like a passed one. It is answered once at
    construction, so the counter must not grow with traffic.
    """
    connector = _connector(tmp_path, kv_cache_config=None)

    _assert_counted(
        connector.stats_snapshot, "compat_check_incomplete_kv_cache_config_unresolved", 1
    )

    connector.get_num_new_matched_tokens(FakeRequest("r1", list(range(100))), 0)
    connector.get_num_new_matched_tokens(FakeRequest("r2", list(range(100))), 0)

    _assert_counted(
        connector.stats_snapshot, "compat_check_incomplete_kv_cache_config_unresolved", 1
    )


def test_missing_block_size_helper_counts_the_check_it_could_not_run(
    tmp_path, monkeypatch
) -> None:
    """Without ``resolve_kv_cache_block_sizes`` the connector cannot confirm
    that vLLM hashes prefixes at the block size it indexes with. That leg is
    then unverified, not verified.
    """
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", None)

    connector = _connector(tmp_path, kv_cache_config=_group_stub(head_size=64, head_size_v=64))

    _assert_counted(
        connector.stats_snapshot,
        "compat_check_incomplete_block_size_resolution_unavailable",
        1,
    )


def test_failing_block_size_helper_counts_a_distinct_incomplete_check(
    tmp_path, monkeypatch
) -> None:
    """A helper that raises is a different diagnosis from one that is absent:
    the engine shape confused vLLM's own resolver. Both leave the check
    unverified, and each gets its own key so the audit can tell them apart.
    """
    module = types.ModuleType("vllm.v1.core.kv_cache_utils")

    def _resolve(kv_cache_config, vllm_config):
        raise RuntimeError("engine shape not resolvable")

    module.resolve_kv_cache_block_sizes = _resolve
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", module)

    connector = _connector(tmp_path, kv_cache_config=_group_stub(head_size=64, head_size_v=64))

    stats = connector.stats_snapshot
    _assert_counted(stats, "compat_check_incomplete_block_size_resolution_failed", 1)
    _assert_counted(stats, "compat_check_incomplete_block_size_resolution_unavailable", 0)


def test_missing_spec_types_count_the_check_they_could_not_run(tmp_path, monkeypatch) -> None:
    """Without vLLM's spec types the gate cannot tell full attention from a
    sliding window or a packed-head-slot page, so it records the skip instead
    of admitting the engine quietly.
    """
    for name in GATE_IMPORTS:
        monkeypatch.setitem(sys.modules, name, None)

    connector = _connector(tmp_path, kv_cache_config=_group_stub(head_size=64, head_size_v=64))

    _assert_counted(connector.stats_snapshot, "compat_check_incomplete_spec_import_failed", 1)


def test_donor_registration_skip_counts_once_per_finished_request(tmp_path) -> None:
    """A donor captured under a layout this connector cannot address is a
    landmine for every later recipient, so the skip is a decline with its own
    key — one per finished request, which is the join the metrics use.
    """
    connector = _connector(
        tmp_path,
        parallel_config=types.SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
        ),
    )
    assert connector._compat_decline is not None  # noqa: SLF001

    connector.request_finished(FakeRequest("d1", list(range(100))), [0, 1, 2])

    stats = connector.stats_snapshot
    _assert_counted(stats, "donor_registration_skipped_incompatible", 1)
    _assert_counted(stats, "donors_registered_total", 0)

    connector.request_finished(FakeRequest("d2", list(range(100))), [0, 1, 2])

    _assert_counted(connector.stats_snapshot, "donor_registration_skipped_incompatible", 2)


# --------------------------------------------------------------------------
# Worker: destinations the connector refuses to compute
# --------------------------------------------------------------------------


def test_multi_group_block_ids_are_refused_and_counted(tmp_path) -> None:
    """Two block-id groups make ``block_ids[0]`` an arbitrary choice, so the
    write is refused rather than aimed at one group's slots.
    """
    pytest.importorskip("torch")
    connector = _connector(tmp_path)

    with pytest.raises(RuntimeError, match="single KV-cache group"):
        connector._slot_mapping(([0, 1], [2, 3]), 4, "cpu")  # noqa: SLF001

    _assert_counted(connector.stats_snapshot, "slot_mapping_multi_group_refused", 1)


def test_uncovered_destination_window_is_refused_and_counted(tmp_path) -> None:
    """One block covers four tokens here; an eight-token window would write a
    prefix and leave the rest of a region the scheduler already marked computed
    uninitialized.
    """
    pytest.importorskip("torch")
    connector = _connector(tmp_path)

    with pytest.raises(RuntimeError, match="do not cover the window"):
        connector._slot_mapping(([0],), 8, "cpu")  # noqa: SLF001

    _assert_counted(connector.stats_snapshot, "slot_mapping_window_not_covered", 1)


def test_packed_page_refusal_names_the_spec_as_its_source(tmp_path) -> None:
    """The KV-cache spec publishes the page's content width. A page that
    disagrees packs head slots, so K and V would land in the wrong cells; the
    counter carries the source that answered so a refusal from the fallback is
    never read as one the engine's own spec disagreed with.
    """
    connector = _connector(tmp_path, kv_cache_config=_group_stub(head_size=64, head_size_v=64))

    with pytest.raises(RuntimeError, match="content dim"):
        connector._require_packed_content_dim((2, 8, 16, 64))  # noqa: SLF001

    _assert_counted(
        connector.stats_snapshot, "packed_head_slot_layout_refused_vs_kv_cache_spec", 1
    )


def test_packed_page_refusal_from_the_hf_fallback_is_counted_separately(tmp_path) -> None:
    """With no spec to read, the model's head size stands in. It answers the
    model's question rather than the page's, so its refusals are counted under
    their own key.
    """
    connector = _connector(tmp_path, kv_cache_config=None)
    connector._vllm_config.model_config.hf_config = types.SimpleNamespace(head_dim=64)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="content dim"):
        connector._require_packed_content_dim((2, 8, 16, 64))  # noqa: SLF001

    _assert_counted(
        connector.stats_snapshot, "packed_head_slot_layout_refused_vs_hf_head_size", 1
    )


def test_unknown_page_width_counts_the_skipped_check_once_per_engine(tmp_path) -> None:
    """Neither source could answer, so the page check is a no-op for the life
    of this engine. The expectation is resolved once, so the counter reports
    one unverified engine rather than one per page inspected.
    """
    connector = _connector(tmp_path, kv_cache_config=None)

    connector._require_packed_content_dim((2, 8, 16, 64))  # noqa: SLF001
    connector._require_packed_content_dim((2, 8, 16, 128))  # noqa: SLF001

    _assert_counted(
        connector.stats_snapshot, "packed_content_dim_check_skipped_unknown_head_size", 1
    )


# --------------------------------------------------------------------------
# Scheduler: spans and prefixes declined at the match hook
# --------------------------------------------------------------------------


def test_unaligned_boundary_decline_is_counted_per_attempt(tmp_path) -> None:
    """Attempt-scoped by design: the boundary moves between attempts, so a
    request declined at an unaligned boundary now may be servable at the next
    one. Counting it once per request would hide the later decline.
    """
    recipient = FakeRequest("r1", list(range(100)))
    connector = _serving_connector(tmp_path, _span_result(), request=recipient)

    assert connector.get_num_new_matched_tokens(recipient, 2) == (0, False)
    assert connector.get_num_new_matched_tokens(recipient, 6) == (0, False)

    _assert_counted(
        connector.stats_snapshot,
        "semantic_span_declined_unaligned_boundary_per_attempt",
        2,
    )


def test_span_clamped_below_the_operator_floor_is_declined_and_counted(tmp_path) -> None:
    """``max_materialized_tokens`` cuts the 76-token span to one block, which
    is below the configured floor. Serving it would break the operator's policy
    and fill the boundary-alignment metric with trivial spans.
    """
    recipient = FakeRequest("r1", list(range(100)))
    connector = _serving_connector(
        tmp_path,
        _span_result(),
        request=recipient,
        max_materialized_tokens=BLOCK_SIZE,
        min_semantic_span=8,
    )

    assert connector.get_num_new_matched_tokens(recipient, 12) == (0, False)

    stats = connector.stats_snapshot
    _assert_counted(stats, "semantic_span_declined_below_min_after_clamp_per_attempt", 1)
    # Attribution: the prompt-headroom clamp is not what cut this span.
    _assert_counted(stats, "semantic_span_clamped_to_prompt_headroom_per_attempt", 0)

    assert connector.get_num_new_matched_tokens(recipient, 12) == (0, False)

    _assert_counted(
        connector.stats_snapshot,
        "semantic_span_declined_below_min_after_clamp_per_attempt",
        2,
    )


def test_span_clamped_to_prompt_headroom_is_counted_per_attempt(tmp_path) -> None:
    """Not a decline but a quiet shortening, which is worse to lose: the span
    reaches the prompt end, and the scheduler asserts there is still a token
    left to compute. The load is served at the clamped length, so the counter
    is the only trace that the advertised span was not the matched one.
    """
    recipient = FakeRequest("r1", list(range(100)))
    connector = _serving_connector(
        tmp_path,
        _span_result(target_start=10, token_count=90),
        request=recipient,
    )

    # Span [12, 100) offers 88 tokens from the boundary; the cacheable prompt
    # prefix (96) leaves only 84.
    assert connector.get_num_new_matched_tokens(recipient, 12) == (84, False)

    _assert_counted(
        connector.stats_snapshot, "semantic_span_clamped_to_prompt_headroom_per_attempt", 1
    )


def test_exact_prefix_below_one_block_is_declined_and_counted_per_attempt(tmp_path) -> None:
    """A match shorter than one block cannot be advertised: vLLM credits
    external tokens in whole blocks, so the truncation leaves nothing to serve.
    The window is re-derived from the boundary on every attempt — it shrinks as
    the local prefix cache grows — so the key is attempt-scoped.
    """
    recipient = FakeRequest("r1", list(range(100)))
    connector = _serving_connector(
        tmp_path,
        _exact_prefix_result(reusable_tokens=BLOCK_SIZE - 1),
        request=recipient,
        mode="exact_prefix",
    )

    assert connector.get_num_new_matched_tokens(recipient, 0) == (0, False)

    _assert_counted(
        connector.stats_snapshot,
        "exact_prefix_declined_below_one_block_after_clamp_per_attempt",
        1,
    )

    assert connector.get_num_new_matched_tokens(recipient, 0) == (0, False)

    _assert_counted(
        connector.stats_snapshot,
        "exact_prefix_declined_below_one_block_after_clamp_per_attempt",
        2,
    )


# --------------------------------------------------------------------------
# Scheduler: loads dropped after they were advertised
# --------------------------------------------------------------------------


def test_allocation_without_blocks_declines_the_load_once(tmp_path) -> None:
    """The scheduler admitted the request but allocated no blocks for the
    external tokens. The load is dropped here so the request stays eligible as
    a donor and never joins the reuse metrics as served.
    """
    recipient = FakeRequest("r1", list(range(100)))
    connector = _serving_connector(tmp_path, _span_result(), request=recipient)

    advertised, _ = connector.get_num_new_matched_tokens(recipient, 12)
    assert advertised == 76

    connector.update_state_after_alloc(recipient, _BlocklessAllocation(), advertised)

    stats = connector.stats_snapshot
    _assert_counted(stats, "load_declined_no_blocks", 1)
    assert not connector._pending_loads  # noqa: SLF001


def test_blockless_pending_load_is_dropped_from_metadata_once(tmp_path) -> None:
    """The same decline on the other side of the hand-off: a pending load that
    reaches metadata without blocks is dropped rather than handed to the worker
    to address nothing.
    """
    connector = _connector(tmp_path)
    connector._pending_loads["r1"] = PendingLoad(  # noqa: SLF001
        request_id="r1",
        donor_id="d1",
        token_count=8,
        materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        namespace="ns",
        block_ids=None,
    )

    metadata = connector.build_connector_meta(FakeSchedulerOutput())

    assert metadata.loads == []
    _assert_counted(connector.stats_snapshot, "load_declined_no_blocks", 1)

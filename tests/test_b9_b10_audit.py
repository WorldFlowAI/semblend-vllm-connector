"""B9/B10: every exit is audited once per request, and every event joins.

Phase 0 measures boundary alignment as a rate over requests, so a decision
the connector takes and does not record is a missing denominator, not a
missing log line: a lookup that returns ``0, False`` from an unaudited exit
is indistinguishable in the trail from a request the connector never saw.
B9 gives each of those exits an event; B10 gives every event a key the
result rows can be joined on.

Two properties carry the measurement and are asserted here over and over:

* one event per REQUEST, not per scheduling attempt. vLLM re-queries the
  match hook on every attempt while a request waits for admission, so an
  exit that emits per attempt scales the reported rate with queueing depth
  instead of with traffic. Exits whose decision moves with
  ``num_computed_tokens`` are allowed to repeat (connector.py's per-request
  vs per-attempt convention) but must then carry an ``attempt`` index, or
  the duplicates cannot be collapsed downstream either.
* a join key stamped at first sight. The vLLM request id alone is not one:
  ids are reused, and a reused id silently merges two requests' rows.

Event names and payload keys are matched loosely (the exit's reason has to
appear in the event name or a reason field) so these tests pin the trail's
content rather than one spelling of it.
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field

import pytest
from test_connector_discovery import (
    FakeBlocks,
    FakeCacheConfig,
    FakeForwardContext,
    FakeKvTransferConfig,
    FakeRequest,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.namespace import namespace_for_request
from semblend_vllm_connector.types import (
    MaterializationKind,
    PendingLoad,
    PendingStore,
    SemanticBlockRef,
    SemanticLookupResult,
    SemanticSegment,
    SemBlendConnectorMetadata,
)


@dataclass(frozen=True)
class FakeParallelConfig:
    decode_context_parallel_size: int = 1
    prefill_context_parallel_size: int = 1


@dataclass
class FakeVllmConfigWithParallel(FakeVllmConfig):
    """Discovery's engine config plus the knob the startup gate declines on."""

    parallel_config: FakeParallelConfig = field(default_factory=FakeParallelConfig)


@dataclass(frozen=True)
class FakeBlocksWithoutIds:
    """What the scheduler hands back when the allocation carries no table."""

    def get_block_ids(self, allow_none: bool = False) -> None:
        return None


class _ScriptedProvider:
    """Returns one prepared lookup result; records the calls it saw."""

    def __init__(self, result: SemanticLookupResult | None) -> None:
        self.result = result
        self.lookups: list[object] = []

    def lookup(self, request):
        self.lookups.append(request)
        return self.result

    def register_donor(self, donor) -> None:
        return None

    def clear_donors(self) -> None:
        return None


class _ExplodingProvider:
    def lookup(self, request):
        raise RuntimeError("provider backend is down")

    def register_donor(self, donor) -> None:
        return None

    def clear_donors(self) -> None:
        return None


# ---------------------------------------------------------------- audit trail

# Where an exit's reason may legitimately be written: the event name itself,
# or one of the reason fields the connector already uses.
_REASON_FIELDS = (
    "event",
    "reason",
    "declined_reason",
    "decline_reason",
    "skip_reason",
    "skipped_reason",
    "exit_reason",
    "outcome",
    "cause",
)

# B10 join key. `request_seq` / `event_seq` are the names the connector's
# AuditJoinKey uses; the rest are accepted so a rename does not read as a
# missing key.
_REQUEST_SEQ_FIELDS = (
    "request_seq",
    "request_key",
    "request_sequence",
    "request_index",
    "request_ordinal",
    "join_key",
    "req_seq",
)
_EVENT_SEQ_FIELDS = (
    "event_seq",
    "event_sequence",
    "event_index",
    "event_ordinal",
    "event_no",
    "seq",
)


# The lookup skip and the donor-registration skip carry the same reason
# ("short_prompt") for two different exits, so the lookup one is matched on
# both needles: a test counting one exit must not collect the other.
_SHORT_PROMPT_LOOKUP_SKIP = (("lookup", "short_prompt"), ("lookup", "min_prompt_tokens"))


def _audit_path(tmp_path):
    return tmp_path / "audit.jsonl"


def _read_events(tmp_path) -> list[dict]:
    path = _audit_path(tmp_path)
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _reason_text(event: dict) -> str:
    return " ".join(str(event.get(field_, "")) for field_ in _REASON_FIELDS).lower()


def _matches(event: dict, alternatives: tuple[tuple[str, ...], ...]) -> bool:
    text = _reason_text(event)
    return any(all(needle in text for needle in needles) for needles in alternatives)


def _for_request(events: list[dict], request_id: str) -> list[dict]:
    return [event for event in events if event.get("request_id") == request_id]


def _matching(events, request_id, alternatives) -> list[dict]:
    return [event for event in _for_request(events, request_id) if _matches(event, alternatives)]


def _trail(events: list[dict]) -> str:
    return (
        ", ".join(
            f"{event.get('event')}(request_id={event.get('request_id')!r}, "
            f"reason={event.get('reason') or event.get('declined_reason')!r})"
            for event in events
        )
        or "<no audit events>"
    )


def _pick(event: dict, names: tuple[str, ...]):
    for name in names:
        if event.get(name) is not None:
            return event[name]
    return None


def _request_seq(event: dict):
    return _pick(event, _REQUEST_SEQ_FIELDS)


def _event_seq(event: dict):
    return _pick(event, _EVENT_SEQ_FIELDS)


def assert_audited_once(events, request_id, alternatives, label) -> dict:
    """Exactly one event for this request names the exit."""
    matched = _matching(events, request_id, alternatives)
    assert len(matched) == 1, (
        f"expected exactly one {label} audit event for request {request_id!r}, "
        f"got {len(matched)}; trail: {_trail(events)}"
    )
    return matched[0]


def assert_audited_deduplicably(events, request_id, alternatives, label) -> dict:
    """One event, or repeats that carry distinct attempt indices.

    An exit decided from num_computed_tokens may fire again at a later
    boundary; two records of the SAME attempt cannot be told apart when the
    rate is computed, so repeats have to be attributable to their attempt.
    """
    matched = _matching(events, request_id, alternatives)
    assert matched, f"no {label} audit event for request {request_id!r}; trail: {_trail(events)}"
    if len(matched) == 1:
        return matched[0]
    attempts = [event.get("attempt") for event in matched]
    assert all(attempt is not None for attempt in attempts), (
        f"{len(matched)} {label} audit events for request {request_id!r} and no attempt "
        f"index to dedupe them by; trail: {_trail(events)}"
    )
    assert len(set(attempts)) == len(attempts), (
        f"{label} audit events for request {request_id!r} repeat attempt indices "
        f"{attempts}; trail: {_trail(events)}"
    )
    return matched[0]


def assert_exit_audited(events, request_id, alternatives, label) -> dict:
    """The exit is recorded, and no single event name is written twice.

    Used where more than one event legitimately describes the exit (a clamp
    plus the terminal zero-supply record), so the count is not pinned, only
    the absence of duplicates within one attempt.
    """
    matched = _matching(events, request_id, alternatives)
    assert matched, f"no {label} audit event for request {request_id!r}; trail: {_trail(events)}"
    names = [event.get("event") for event in matched]
    assert len(set(names)) == len(names), (
        f"{label} audit events repeat for request {request_id!r}: {names}; trail: {_trail(events)}"
    )
    return matched[0]


# ------------------------------------------------------------------- fixtures


def _connector(
    tmp_path,
    overrides: dict,
    *,
    role: KVConnectorRole = KVConnectorRole.SCHEDULER,
    block_size: int = 16,
    config_cls=FakeVllmConfig,
    **config_kwargs,
) -> SemBlendVllmConnector:
    extra = {
        "provider": "local",
        "audit_path": str(_audit_path(tmp_path)),
        "kv_storage_path": str(tmp_path / "kv"),
        **overrides,
    }
    vllm_config = config_cls(
        FakeKvTransferConfig(extra),
        cache_config=FakeCacheConfig(block_size=block_size),
        **config_kwargs,
    )
    return SemBlendVllmConnector(vllm_config, role)


def _span_connector(tmp_path, *, max_materialized_tokens: int = 4096):
    connector = _connector(
        tmp_path,
        {
            "mode": "semantic_span_experimental",
            "min_prompt_tokens": 4,
            "min_semantic_span": 8,
            "max_materialized_tokens": max_materialized_tokens,
        },
        block_size=4,
    )
    # Donor span: target [10..90) from donor position 210. Snapped inward to
    # block edges it is [12..88) from donor 212.
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        SemanticLookupResult(
            donor_id="d1",
            similarity=0.99,
            reusable_token_count=80,
            materialization_kind=MaterializationKind.SEMANTIC_SPAN,
            segments=[
                SemanticSegment(donor_id="d1", donor_start=210, target_start=10, token_count=80)
            ],
        )
    )
    return connector


def _write_donor_capture(connector, request, donor_id: str, token_count: int) -> None:
    """The donor length the scheduler role reads back before planning a span."""
    namespace = namespace_for_request(
        connector._config,
        connector._vllm_config,
        request,  # noqa: SLF001
    )
    os.makedirs(connector._donor_dir(donor_id, namespace), exist_ok=True)  # noqa: SLF001
    with open(
        connector._donor_metadata_path(donor_id, namespace),
        "w",
        encoding="utf-8",  # noqa: SLF001
    ) as f:
        json.dump({"token_count": token_count}, f)


def _named(events: list[dict], request_id: str, event_name: str) -> list[dict]:
    """Events for one request written under one exact event name.

    Used where a test is about how many rows one site wrote, which the loose
    reason matcher cannot answer: it collects every event whose text mentions
    the needle, including the neighbouring sites' rows.
    """
    return [event for event in _for_request(events, request_id) if event.get("event") == event_name]


def _exact_prefix_connector(tmp_path, *, reusable_token_count: int = 96):
    """An exact_prefix connector whose hit is servable at several boundaries."""
    connector = _connector(
        tmp_path,
        {"mode": "exact_prefix", "min_prompt_tokens": 4},
        block_size=16,
    )
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        SemanticLookupResult(
            donor_id="d1",
            similarity=0.99,
            reusable_token_count=reusable_token_count,
            materialization_kind=MaterializationKind.EXACT_PREFIX,
            block_refs=[
                SemanticBlockRef(block_id="b0", start_token=0, token_count=reusable_token_count)
            ],
        )
    )
    return connector


def _request_only_connector_with_donor(tmp_path):
    """A request_only connector holding one stored donor, ready to advertise."""
    connector = _connector(
        tmp_path,
        {
            "mode": "request_only_experimental",
            "min_prompt_tokens": 4,
            "min_similarity": 0.3,
            "max_materialized_tokens": 12,
        },
        block_size=4,
    )
    donor = FakeRequest("d1", list(range(20)))
    connector.request_finished(donor, [0, 1, 2, 3, 4])
    _write_donor_capture(connector, donor, "d1", token_count=12)
    return connector


# ------------------------------------------------- B9: the silent exits fire


def test_short_prompt_exit_is_audited_once_per_request(tmp_path) -> None:
    """A prompt below min_prompt_tokens never reaches lookup; say so once.

    Nothing about this decision changes between scheduling attempts, so the
    three attempts below are one request in every rate computed from the
    trail.
    """
    connector = _connector(tmp_path, {"min_prompt_tokens": 8})
    request = FakeRequest("r1", [1, 2, 3])

    for _ in range(3):
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

    events = _read_events(tmp_path)
    event = assert_audited_once(events, "r1", _SHORT_PROMPT_LOOKUP_SKIP, "short-prompt skip")
    assert event["request_id"] == "r1"


def test_incompatible_engine_exit_is_audited_once_per_request(tmp_path) -> None:
    """An engine the startup gate declined still answers the match hook.

    Without an event per request the trail shows a run of requests that were
    never looked up and no reason why: the reuse rate reads as a property of
    the traffic instead of a declined engine configuration.
    """
    connector = _connector(
        tmp_path,
        {"min_prompt_tokens": 4},
        config_cls=FakeVllmConfigWithParallel,
        parallel_config=FakeParallelConfig(decode_context_parallel_size=2),
    )
    assert connector._compat_decline is not None  # noqa: SLF001
    request = FakeRequest("r1", list(range(64)))

    for _ in range(3):
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

    events = _read_events(tmp_path)
    assert_audited_once(events, "r1", (("incompat",),), "incompatible-engine skip")


def test_exact_prefix_skip_gate_is_audited(tmp_path) -> None:
    """The skip that suppresses precisely the lane being measured.

    Both attempts sit at the same boundary, so they are the same decision:
    the trail must not carry two records of it that cannot be told apart.
    """
    connector = _connector(
        tmp_path,
        {"min_prompt_tokens": 4, "skip_when_exact_prefix_ratio_at_least": 0.5},
    )
    request = FakeRequest("r1", list(range(16)))

    for _ in range(2):
        assert connector.get_num_new_matched_tokens(request, 8) == (0, False)

    events = _read_events(tmp_path)
    event = assert_audited_deduplicably(
        events,
        "r1",
        (("exact_ratio",), ("exact_prefix_sufficient",), ("exact_prefix_skip",)),
        "exact-prefix skip",
    )
    assert event["request_id"] == "r1"


def test_absent_provider_exit_is_audited_once_per_request(tmp_path) -> None:
    """No provider is a configuration fault, and it is request-stable."""
    connector = _connector(tmp_path, {"min_prompt_tokens": 4})
    connector._provider = None  # noqa: SLF001
    request = FakeRequest("r1", list(range(16)))

    for _ in range(2):
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

    events = _read_events(tmp_path)
    assert_audited_once(
        events,
        "r1",
        (("no_provider",), ("provider_missing",), ("lookup_error",), ("provider_error",)),
        "absent-provider skip",
    )


def test_provider_lookup_error_is_audited(tmp_path) -> None:
    """A raising provider falls back to normal prefill; the fallback is data.

    Counted only in stats today, so a run whose provider was down is
    indistinguishable in the trail from a run with no semantic supply.
    """
    connector = _connector(tmp_path, {"min_prompt_tokens": 4})
    connector._provider = _ExplodingProvider()  # noqa: SLF001

    assert connector.get_num_new_matched_tokens(FakeRequest("r1", list(range(16))), 0) == (
        0,
        False,
    )

    events = _read_events(tmp_path)
    assert_audited_once(
        events,
        "r1",
        (("lookup_error",), ("provider_error",), ("provider_failed",), ("lookup_failed",)),
        "provider-error",
    )


def test_below_min_boundary_exit_is_audited_deduplicably(tmp_path) -> None:
    """Two attempts at the same boundary are one decline, not two.

    This gate is what keeps boundary-0 serves out of the alignment metric,
    so its own count is part of that metric's denominator.
    """
    connector = _connector(tmp_path, {"min_prompt_tokens": 4, "min_boundary_tokens": 16})
    request = FakeRequest("r1", list(range(64)))

    for _ in range(2):
        assert connector.get_num_new_matched_tokens(request, 4) == (0, False)

    events = _read_events(tmp_path)
    event = assert_audited_deduplicably(
        events, "r1", (("min_boundary",),), "below-min-boundary skip"
    )
    assert event["boundary"] == 4
    assert event["min_boundary_tokens"] == 16


def test_discovery_only_fallthrough_is_audited(tmp_path) -> None:
    """A hit that carries no servable supply is not the same as a miss.

    The lookup hit is already audited; without this event the trail shows a
    hit and then nothing, which reads as an advertised load that vanished.
    """
    connector = _connector(
        tmp_path,
        {"mode": "request_only_experimental", "min_prompt_tokens": 4, "min_similarity": 0.3},
        block_size=4,
    )
    connector.request_finished(FakeRequest("d1", list(range(20))), [0, 1, 2, 3, 4])
    recipient = FakeRequest("r1", list(range(10)) + [100, 101, 102, 103])

    assert connector.get_num_new_matched_tokens(recipient, 0) == (0, False)

    events = _read_events(tmp_path)
    assert_audited_once(events, "r1", (("discovery_only",),), "discovery-only fall-through")


def test_boundary_outside_every_span_is_audited_once(tmp_path) -> None:
    """The exit the boundary-alignment metric exists to count.

    A boundary in a novel region returns zero from the same statement as a
    donor that was never captured; both are silent on HEAD, so the metric
    has no failure diagnosis at all.
    """
    connector = _span_connector(tmp_path)
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    assert connector.get_num_new_matched_tokens(recipient, 0) == (0, False)

    events = _read_events(tmp_path)
    event = assert_audited_once(
        events,
        "r1",
        (("boundary_missed",), ("no_span",), ("span_missed",), ("no_supply",)),
        "boundary-outside-every-span",
    )
    assert event["request_id"] == "r1"
    assert event["boundary"] == 0


def test_boundary_missed_event_separates_the_four_causes(tmp_path) -> None:
    """Real misalignment, no capture, short capture and below-min all land
    on the same return, so the event has to carry what tells them apart."""
    connector = _span_connector(tmp_path)
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    assert connector.get_num_new_matched_tokens(recipient, 8) == (0, False)

    events = _read_events(tmp_path)
    event = assert_audited_once(
        events,
        "r1",
        (("boundary_missed",), ("no_span",), ("span_missed",), ("no_supply",)),
        "boundary-outside-every-span",
    )
    assert event["donor_id"] == "d1"
    assert event["boundary"] == 8
    assert event["block_size"] == 4
    assert event["min_semantic_span"] == 8
    assert event["max_materialized_tokens"] == 4096
    assert _pick(event, ("stored_donor_tokens", "stored_donor_token_count")) == 4096
    assert _pick(event, ("n_raw_segments", "num_raw_segments", "raw_segment_count")) == 1

    raw_spans = event["raw_spans"]
    assert len(raw_spans) == 1
    assert raw_spans[0]["target_start"] == 10
    assert raw_spans[0]["length"] == 80
    assert raw_spans[0]["donor_start"] == 210

    snapped = event["snapped_spans"]
    assert len(snapped) == 1
    assert snapped[0]["target_start"] == 12
    assert snapped[0]["target_end"] == 88
    assert snapped[0]["donor_start"] == 212


def test_span_clamped_to_zero_supply_is_audited(tmp_path) -> None:
    """An operator budget below one block silently erases a servable span.

    supply_at_boundary found 76 tokens here; the max_materialized_tokens
    clamp floors that to zero and the request leaves through the same
    unaudited return as a genuine miss.
    """
    connector = _span_connector(tmp_path, max_materialized_tokens=2)
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    assert connector.get_num_new_matched_tokens(recipient, 12) == (0, False)

    events = _read_events(tmp_path)
    event = assert_exit_audited(
        events,
        "r1",
        (("clamp",), ("boundary_missed",), ("no_span",), ("no_supply",)),
        "clamped-to-zero supply",
    )
    assert event["boundary"] == 12


def test_alloc_without_pending_load_is_audited(tmp_path) -> None:
    """The scheduler credited external tokens this connector never planned.

    Counted only in stats today; it is the one event that says an advertise
    and an allocation disagreed, which is a correctness signal, not noise.
    """
    connector = _connector(tmp_path, {"mode": "request_only_experimental", "min_prompt_tokens": 4})

    connector.update_state_after_alloc(FakeRequest("r1", list(range(16))), FakeBlocks(([1, 2],)), 8)

    events = _read_events(tmp_path)
    assert_audited_once(
        events,
        "r1",
        (("pending_load",), ("no_plan",), ("plan_missing",), ("without_plan",), ("unplanned",)),
        "allocation without a pending load",
    )


def test_alloc_without_block_ids_is_audited_once(tmp_path) -> None:
    """An allocation with no block table declines the load, once."""
    connector = _request_only_connector_with_donor(tmp_path)
    recipient = FakeRequest("r1", list(range(10)) + [100, 101, 102, 103])
    matched, _ = connector.get_num_new_matched_tokens(recipient, 0)
    assert matched == 8

    connector.update_state_after_alloc(recipient, FakeBlocksWithoutIds(), matched)

    events = _read_events(tmp_path)
    assert_audited_once(events, "r1", (("no_blocks",),), "load declined for missing blocks")


def test_donor_gone_load_is_audited_once(tmp_path, monkeypatch) -> None:
    """A donor evicted between advertise and load, on the worker role.

    The worker sees the request for the first time here, so this is also
    where its join key has to be stamped.
    """
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")

    def _missing_donor_file(filename):
        raise OSError(f"donor file is gone: {filename}")

    fake_safetensors_torch.load_file = _missing_donor_file
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    connector = _connector(
        tmp_path,
        {"mode": "request_only_experimental"},
        role=KVConnectorRole.WORKER,
    )
    connector.register_kv_caches({"layers.0": object()})
    connector.bind_connector_metadata(
        SemBlendConnectorMetadata(
            loads=[
                PendingLoad(
                    request_id="r1",
                    donor_id="d1",
                    token_count=8,
                    materialization_kind=MaterializationKind.REQUEST_ONLY,
                    namespace="test",
                    block_ids=([10, 11],),
                )
            ],
        )
    )

    connector.start_load_kv(FakeForwardContext())

    events = _read_events(tmp_path)
    event = assert_audited_once(events, "r1", (("donor_gone",),), "donor-gone decline")
    assert event["donor_id"] == "d1"


# ------------------------------------------- B9: one advertise per request


def test_requeued_request_advertises_once_not_per_attempt(tmp_path) -> None:
    """Three scheduling attempts, one advertised load.

    The per-request reuse metrics join on the advertise event, so a request
    re-queued under memory pressure must not inflate their numerator.
    """
    connector = _span_connector(tmp_path)
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    for _ in range(3):
        assert connector.get_num_new_matched_tokens(recipient, 12) == (76, False)

    events = _read_events(tmp_path)
    assert_audited_once(events, "r1", (("advertise",),), "span load advertised")


def test_advertised_span_event_carries_snapped_spans(tmp_path) -> None:
    """The advertise event is the join row for a served request, so the plan
    it names has to be readable without re-deriving it from the connector."""
    connector = _span_connector(tmp_path)
    recipient = FakeRequest("r1", list(range(100)))
    _write_donor_capture(connector, recipient, "d1", token_count=4096)

    assert connector.get_num_new_matched_tokens(recipient, 12) == (76, False)

    events = _read_events(tmp_path)
    event = assert_audited_once(events, "r1", (("advertise",),), "span load advertised")
    snapped = event["snapped_spans"]
    assert len(snapped) == 1
    assert snapped[0]["target_start"] == 12
    assert snapped[0]["target_end"] == 88
    assert snapped[0]["donor_start"] == 212


# ----------------------------------------------------- B10: the join key


def test_every_event_for_one_request_shares_a_key_and_increasing_event_seq(tmp_path) -> None:
    """Advertise, allocate and finish have to join on one key per request.

    Equality on the vLLM request id is what the result rows join by, so
    every event naming a request carries the request's own sequence too,
    and the events within that request are ordered by a sequence of their
    own -- without it two events written in the same second cannot be put
    in order at all.
    """
    connector = _request_only_connector_with_donor(tmp_path)
    recipient = FakeRequest("r1", list(range(10)) + [100, 101, 102, 103])
    matched, _ = connector.get_num_new_matched_tokens(recipient, 0)
    connector.update_state_after_alloc(recipient, FakeBlocks(([10, 11],)), matched)
    connector.request_finished(recipient, [10, 11])

    events = _read_events(tmp_path)
    request_events = [event for event in events if event.get("request_id") is not None]
    assert request_events, f"no request-scoped audit events; trail: {_trail(events)}"
    for event in request_events:
        assert _request_seq(event) is not None, (
            f"audit event {event.get('event')!r} has no request key to join on; "
            f"fields: {sorted(event)}"
        )
        assert _event_seq(event) is not None, (
            f"audit event {event.get('event')!r} has no event sequence; fields: {sorted(event)}"
        )

    recipient_events = _for_request(events, "r1")
    assert len(recipient_events) >= 3, _trail(events)
    keys = {_request_seq(event) for event in recipient_events}
    assert len(keys) == 1, f"request r1 carries {len(keys)} different keys: {keys}"

    sequences = [_event_seq(event) for event in recipient_events]
    assert sequences == sorted(set(sequences)), (
        f"event sequence for request r1 is not strictly increasing: {sequences}"
    )

    donor_keys = {_request_seq(event) for event in _for_request(events, "d1")}
    assert donor_keys, f"donor d1 has no audited events; trail: {_trail(events)}"
    assert donor_keys.isdisjoint(keys), (
        f"donor d1 and recipient r1 share a request key: {donor_keys & keys}"
    )


def test_reused_request_id_gets_a_distinct_request_key(tmp_path) -> None:
    """vLLM request ids are reused; the join key must not be.

    Two requests that happen to carry the same id are two rows in the
    result set, and joining them into one silently averages a cold request
    with a warm one.
    """
    connector = _connector(tmp_path, {"min_prompt_tokens": 8})

    first = FakeRequest("r1", [1, 2, 3])
    assert connector.get_num_new_matched_tokens(first, 0) == (0, False)
    connector.request_finished(first, [])

    second = FakeRequest("r1", [4, 5, 6])
    assert connector.get_num_new_matched_tokens(second, 0) == (0, False)

    events = _read_events(tmp_path)
    skips = _matching(events, "r1", _SHORT_PROMPT_LOOKUP_SKIP)
    assert len(skips) == 2, f"expected one skip per request; trail: {_trail(events)}"
    assert all(event["request_id"] == "r1" for event in skips)

    first_key, second_key = (_request_seq(event) for event in skips)
    assert first_key is not None and second_key is not None, (
        f"short-prompt events carry no request key; fields: {sorted(skips[0])}"
    )
    assert first_key != second_key, (
        f"both uses of request id 'r1' were stamped with key {first_key!r}"
    )


# ------------------------------------ a plan that CHANGES is a new advertise


def test_re_advertise_at_a_new_boundary_is_audited_with_its_boundary(tmp_path) -> None:
    """One advertise row per distinct plan, not one per request.

    Deduping the advertise on "this request has a plan" keeps the row from
    the FIRST boundary and drops the one from the boundary actually served.
    The allocate row that follows carries the second plan's token count, so
    the M1/M2 join then reads a promise of 96 tokens against a load of 80 --
    a discrepancy invented entirely by the dedupe.
    """
    connector = _exact_prefix_connector(tmp_path)
    recipient = FakeRequest("r1", list(range(128)))

    # Two attempts at boundary 0 plan the same load: the second is silent.
    assert connector.get_num_new_matched_tokens(recipient, 0) == (96, False)
    assert connector.get_num_new_matched_tokens(recipient, 0) == (96, False)
    # Re-queried after vLLM's own prefix cache computed a block: a new plan.
    assert connector.get_num_new_matched_tokens(recipient, 16) == (80, False)

    events = _read_events(tmp_path)
    advertises = _named(events, "r1", "exact_prefix_load_advertised")
    assert len(advertises) == 2, (
        f"expected one advertise per distinct plan, got {len(advertises)}; trail: {_trail(events)}"
    )
    assert [event["boundary"] for event in advertises] == [0, 16]
    assert [event["tokens"] for event in advertises] == [96, 80]
    attempts = [event["attempt"] for event in advertises]
    assert len(set(attempts)) == len(attempts), (
        f"two advertise rows for request 'r1' share an attempt index: {attempts}"
    )


def test_identical_re_record_does_not_advertise_again(tmp_path) -> None:
    """A request re-queued at the SAME boundary plans the same load.

    The guard the test above relaxes still has to hold: three attempts that
    decide the same thing are one promise, or the numerator scales with
    queueing depth.
    """
    connector = _exact_prefix_connector(tmp_path)
    recipient = FakeRequest("r1", list(range(128)))

    for _ in range(3):
        assert connector.get_num_new_matched_tokens(recipient, 16) == (80, False)

    events = _read_events(tmp_path)
    assert len(_named(events, "r1", "exact_prefix_load_advertised")) == 1, _trail(events)
    assert connector.stats_snapshot["exact_prefix_loads_advertised_total"] == 1


# ------------------------------- a later fall-through is its own exit, too


def test_fall_through_on_a_later_attempt_is_audited_with_its_own_reason(tmp_path) -> None:
    """The same request falls through twice for two different reasons.

    Attempt 0 is turned away inside the request_only branch (no shared
    prefix) and says so. Attempt 1 never enters that branch at all, because
    the branch is boundary-0 only -- a different exit, with a different fix.
    Deduping the fall-through on the first lookup drops the second row
    entirely, so the trail ends on a reason that stopped applying.
    """
    connector = _connector(
        tmp_path,
        {
            "mode": "request_only_experimental",
            "min_prompt_tokens": 4,
            "max_materialized_tokens": 12,
        },
        block_size=4,
    )
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        SemanticLookupResult(
            donor_id="d1",
            similarity=0.99,
            reusable_token_count=12,
            materialization_kind=MaterializationKind.REQUEST_ONLY,
            # Shares no prefix with the recipient below.
            donor_token_ids=list(range(900, 912)),
        )
    )
    recipient = FakeRequest("r1", list(range(40)))
    _write_donor_capture(connector, recipient, "d1", token_count=12)

    assert connector.get_num_new_matched_tokens(recipient, 0) == (0, False)
    assert connector.get_num_new_matched_tokens(recipient, 16) == (0, False)

    events = _read_events(tmp_path)
    # Attempt 0's exit is the specific one, and it does not also fall through.
    assert len(_named(events, "r1", "request_only_load_rejected")) == 1, _trail(events)
    fall_throughs = _named(events, "r1", "materialization_rejected_no_safe_plan")
    assert len(fall_throughs) == 1, (
        f"the boundary-16 fall-through left {len(fall_throughs)} rows; trail: {_trail(events)}"
    )
    assert fall_throughs[0]["reason"] == "boundary_not_zero"
    assert fall_throughs[0]["boundary"] == 16
    assert fall_throughs[0]["attempt"] == 1


def test_fall_through_reason_names_the_missing_precondition(tmp_path) -> None:
    """`mode_kind_mismatch` is the wrong answer when the mode matches.

    This hit is exact_prefix in exact_prefix mode; what is missing is the
    block refs. Reporting the mode/kind pair sends the reader to look at a
    configuration that is in fact correct.
    """
    connector = _exact_prefix_connector(tmp_path)
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        SemanticLookupResult(
            donor_id="d1",
            similarity=0.99,
            reusable_token_count=96,
            materialization_kind=MaterializationKind.EXACT_PREFIX,
            block_refs=None,
        )
    )

    assert connector.get_num_new_matched_tokens(FakeRequest("r1", list(range(128))), 0) == (
        0,
        False,
    )

    events = _read_events(tmp_path)
    fall_throughs = _named(events, "r1", "materialization_rejected_no_safe_plan")
    assert len(fall_throughs) == 1, _trail(events)
    assert fall_throughs[0]["reason"] == "no_block_refs"


def test_repeated_fall_through_for_one_reason_is_still_one_row(tmp_path) -> None:
    """Dedupe on the reason, not away from it.

    Three attempts at the same unservable boundary are one exit; only a
    reason that was not already recorded earns a second row.
    """
    connector = _connector(
        tmp_path,
        {"mode": "exact_prefix", "min_prompt_tokens": 4},
        block_size=16,
    )
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        SemanticLookupResult(
            donor_id="d1",
            similarity=0.99,
            reusable_token_count=96,
            materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        )
    )
    recipient = FakeRequest("r1", list(range(128)))

    for boundary in (0, 16, 32):
        assert connector.get_num_new_matched_tokens(recipient, boundary) == (0, False)

    events = _read_events(tmp_path)
    fall_throughs = _named(events, "r1", "materialization_rejected_no_safe_plan")
    assert len(fall_throughs) == 1, _trail(events)
    assert fall_throughs[0]["reason"] == "mode_kind_mismatch"
    assert connector.stats_snapshot["materialization_rejected_no_safe_plan"] == 1


def test_mode_suppressed_materialization_carries_its_attempt(tmp_path) -> None:
    """Every per-request row needs the attempt it was decided on.

    Without it a duplicate written by a future change cannot be collapsed
    downstream, which is the property the whole trail is deduped on.
    """
    connector = _connector(
        tmp_path,
        {"mode": "discovery_only", "min_prompt_tokens": 4},
        block_size=4,
    )
    connector._provider = _ScriptedProvider(  # noqa: SLF001
        SemanticLookupResult(
            donor_id="d1",
            similarity=0.99,
            reusable_token_count=8,
            materialization_kind=MaterializationKind.SEMANTIC_SPAN,
        )
    )
    request = FakeRequest("r1", list(range(40)))

    for _ in range(2):
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

    events = _read_events(tmp_path)
    event = assert_audited_once(
        events, "r1", (("suppressed_by_mode",),), "mode-suppressed materialization"
    )
    assert event["attempt"] == 0


# ------------------------------------------------- worker-role bookkeeping


def test_worker_retires_join_key_at_get_finished(tmp_path, monkeypatch) -> None:
    """get_finished is the worker's only finish hook.

    request_finished runs on the scheduler role, in another process under a
    real engine, so the worker's per-request maps are never swept: they grow
    for the life of the process, and a reused request id is then stamped with
    the first request's sequence -- silently merging two requests' rows.
    """
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")

    def _missing_donor_file(filename):
        raise OSError(f"donor file is gone: {filename}")

    fake_safetensors_torch.load_file = _missing_donor_file
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    connector = _connector(
        tmp_path,
        {"mode": "request_only_experimental"},
        role=KVConnectorRole.WORKER,
    )
    connector.register_kv_caches({"layers.0": object()})

    def _decline_a_load() -> None:
        connector.bind_connector_metadata(
            SemBlendConnectorMetadata(
                loads=[
                    PendingLoad(
                        request_id="r1",
                        donor_id="d1",
                        token_count=8,
                        materialization_kind=MaterializationKind.REQUEST_ONLY,
                        namespace="test",
                        block_ids=([10, 11],),
                    )
                ],
            )
        )
        connector.start_load_kv(FakeForwardContext())

    _decline_a_load()
    connector._capture_progress["r1"] = {"layers.0": 8}  # noqa: SLF001

    connector.get_finished({"r1"})

    assert "r1" not in connector._request_seq, (  # noqa: SLF001
        "worker kept the request's join sequence after it finished"
    )
    assert "r1" not in connector._request_event_seq, (  # noqa: SLF001
        "worker kept the request's event counter after it finished"
    )
    assert "r1" not in connector._capture_progress  # noqa: SLF001

    # The next use of the id is a different request and must not join to the
    # first one's rows.
    _decline_a_load()

    events = _read_events(tmp_path)
    declines = _matching(events, "r1", (("donor_gone",),))
    assert len(declines) == 2, _trail(events)
    first_key, second_key = (_request_seq(event) for event in declines)
    assert first_key != second_key, (
        f"both uses of request id 'r1' were stamped with key {first_key!r}"
    )


def test_foreign_metadata_on_the_capture_leg_is_named_once(tmp_path) -> None:
    """A MultiConnector misconfiguration drops every queued capture.

    The load leg already names this shape; the capture leg returned on it in
    silence, so a run whose donor pool stayed empty had no row saying why --
    and an empty donor pool reads as "no semantic supply in this traffic".
    """
    connector = _connector(
        tmp_path,
        {"mode": "request_only_experimental"},
        role=KVConnectorRole.WORKER,
    )
    connector.bind_connector_metadata(object())

    connector.save_kv_layer("layers.0", object(), attn_metadata=object())
    connector.save_kv_layer("layers.1", object(), attn_metadata=object())

    events = _read_events(tmp_path)
    named = [event for event in events if event.get("event") == "capture_metadata_unexpected_type"]
    assert len(named) == 1, (
        f"expected one shape warning per connector, got {len(named)}; trail: {_trail(events)}"
    )
    # The counter keeps the true count; the event does not repeat per layer.
    assert connector.stats_snapshot["capture_metadata_unexpected_type_layers"] == 2


def test_empty_store_list_stays_silent(tmp_path) -> None:
    """A step with nothing to capture is normal progress, not an incident."""
    connector = _connector(
        tmp_path,
        {"mode": "request_only_experimental"},
        role=KVConnectorRole.WORKER,
    )
    connector.bind_connector_metadata(SemBlendConnectorMetadata())

    connector.save_kv_layer("layers.0", object(), attn_metadata=object())

    events = _read_events(tmp_path)
    assert [
        event for event in events if event.get("event") == "capture_metadata_unexpected_type"
    ] == []


def test_capture_base_missing_is_audited_once_per_store(tmp_path) -> None:
    """A chunked capture that loses its earlier chunks restarts from zero.

    Nothing in the outcome shows it: the donor is still written and still
    advertises a length. Only the trail can say that this donor re-copied
    the same prefix on every chunk. Every layer of one step loses the same
    base, so the row is written once per store rather than once per layer.
    """
    torch = pytest.importorskip("torch")

    connector = _connector(
        tmp_path,
        {"mode": "request_only_experimental", "kv_storage_backend": "memory"},
        role=KVConnectorRole.WORKER,
        block_size=4,
    )
    kv_layer = torch.zeros(2, 2, 4, 2, 8)

    def _capture(token_count: int) -> None:
        connector.bind_connector_metadata(
            SemBlendConnectorMetadata(
                stores=[
                    PendingStore(
                        request_id="d1",
                        token_ids=list(range(token_count)),
                        token_count=token_count,
                        namespace="ns",
                        block_ids=([0, 1],),
                        final=True,
                    )
                ]
            )
        )
        connector.save_kv_layer("layers.0", kv_layer, attn_metadata=object())
        connector.save_kv_layer("layers.1", kv_layer, attn_metadata=object())
        # The end of the step: the writer is joined and the donor is on the
        # store rather than staged in its queue.
        connector.wait_for_save()

    _capture(4)
    # The donor's stored tensors go between chunks (an eviction, a retracted
    # backend); the next chunk has nothing to append to.
    connector._memory_store.clear()  # noqa: SLF001

    _capture(8)

    events = _read_events(tmp_path)
    rows = _named(events, "d1", "capture_base_missing")
    assert len(rows) == 1, (
        f"expected one base-missing row per store, got {len(rows)}; trail: {_trail(events)}"
    )
    assert rows[0]["discarded_start"] == 4
    assert rows[0]["token_count"] == 8
    assert rows[0]["layer"] == "layers.0"
    # The counter keeps its own unit: both layers lost their base.
    assert connector.stats_snapshot["layer_capture_base_missing"] == 2

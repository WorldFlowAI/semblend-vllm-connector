"""A served request must not join the donor pool it was served from.

Measured on stock vLLM: a prompt served from a donor was re-issued verbatim
five times. Each re-issue was served, each one skipped its own KV capture
(``capture_skipped`` with ``reason=served_request``) -- and each one was then
registered as a donor anyway. Those registrations hold no KV at all, and they
are identical to each other, so by the fifth re-issue they were its five
nearest neighbours: the donor that actually held the KV fell out of the
candidate cut and the reuse stopped, with nothing in the audit trail but a
miss.

Two gates are pinned here. The connector refuses to register a request whose
capture was skipped, and the ranking the connector owns drops donors with no
captured KV before the top-k cut rather than after it.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from test_connector_discovery import (
    FakeBlocks,
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeSchedulerOutput,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.namespace import namespace_for_request
from semblend_vllm_connector.provider import rank_candidates
from semblend_vllm_connector.types import (
    DonorRegistration,
    MaterializationKind,
    SemanticLookupResult,
    SemanticSegment,
)

BLOCK_SIZE = 4
TOP_K = 5
SEED_TOKENS = list(range(40))
# Shares a 36-token prefix with the seed and differs in the tail, the way a
# recipient differs from the donor it matches.
RECIPIENT_TOKENS = [*range(36), 900, 901, 902, 903]
BOUNDARY = 12  # block-aligned; the span runs [0, 36), so [12, 36) is servable
SERVED_TOKENS = 24


class _RankingProvider:
    """A donor index shaped like the production one: score, cut, then verify.

    The cut is what the defect turns on, so it is real here: every registered
    donor is scored against the query, ``rank_candidates`` keeps the best
    ``top_k``, and the verification behind the cut picks from the survivors.
    Donor KV lives in the connector's own store, so a registration alone
    supplies nothing.

    The verification step steps over a donor whose tokens are identical to the
    query -- that is what vLLM's own exact prefix cache is for. It is
    calibrated on the measured run: there the pipeline kept choosing the seed
    over the identical re-issues ranked above it, until enough of them existed
    to fill the cut and the seed was no longer in it.
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k
        self.donors: "OrderedDict[str, DonorRegistration]" = OrderedDict()

    def lookup(self, request: Any) -> SemanticLookupResult | None:
        query = set(request.token_ids)
        scored = []
        for donor in self.donors.values():
            if donor.namespace != request.namespace:
                continue
            tokens = set(donor.token_ids)
            scored.append((len(query & tokens) / len(query | tokens), donor))
        survivors = [
            (similarity, donor)
            for similarity, donor in rank_candidates(scored, top_k=self.top_k)
            if list(donor.token_ids) != list(request.token_ids)
        ]
        if not survivors:
            return None
        similarity, donor = survivors[0]
        shared = 0
        for mine, theirs in zip(donor.token_ids, request.token_ids):
            if mine != theirs:
                break
            shared += 1
        if shared == 0:
            return None
        return SemanticLookupResult(
            donor_id=donor.donor_id,
            similarity=similarity,
            reusable_token_count=shared,
            materialization_kind=MaterializationKind.SEMANTIC_SPAN,
            segments=[
                SemanticSegment(
                    donor_id=donor.donor_id,
                    donor_start=0,
                    target_start=0,
                    token_count=shared,
                )
            ],
        )

    def register_donor(self, donor: DonorRegistration) -> None:
        self.donors[donor.donor_id] = donor

    def clear_donors(self) -> None:
        self.donors.clear()


def _connector(tmp_path) -> SemBlendVllmConnector:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "semantic_span_experimental",
                "provider": "local",
                "min_prompt_tokens": 8,
                "min_similarity": 0.3,
                "min_semantic_span": 8,
                "lookup_top_k": TOP_K,
                "kv_storage_path": str(tmp_path / "kv"),
                "audit_path": str(tmp_path / "audit.jsonl"),
            }
        ),
        cache_config=FakeCacheConfig(block_size=BLOCK_SIZE),
    )
    return SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)


def _write_donor_capture(connector, request, donor_id: str, token_count: int) -> None:
    """The donor-length record the span planner reads before it trims a span."""
    namespace = namespace_for_request(connector._config, connector._vllm_config, request)  # noqa: SLF001
    os.makedirs(connector._donor_dir(donor_id, namespace), exist_ok=True)  # noqa: SLF001
    with open(connector._donor_metadata_path(donor_id, namespace), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": token_count}, f)


def _audit(tmp_path) -> list[dict]:
    with open(tmp_path / "audit.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class _ScheduledNewReq:
    """A request as it reaches the store builder: with its allocated blocks."""

    request_id: str
    all_token_ids: list[int]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int = 0


def _serve_once(connector, request) -> int:
    """One request through the four scheduler calls stock vLLM makes."""
    blocks = (list(range(10)),)
    matched, _ = connector.get_num_new_matched_tokens(request, BOUNDARY)
    connector.update_state_after_alloc(request, FakeBlocks(blocks), matched)
    scheduled = _ScheduledNewReq(request.request_id, list(request.all_token_ids), blocks)
    connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[scheduled]))
    connector.request_finished(request, [0])
    return matched


def test_verbatim_repeats_stay_served_and_never_become_donors(tmp_path) -> None:
    """The seed survives six re-issues of the request it was serving.

    Before the fix the fifth re-issue missed, exactly as the measured run
    did: the recipient and the four re-issues before it were the five nearest
    neighbours of anything identical to them, so ``top_k=5`` had no room left
    for the seed. The donor pool here is the proof -- it holds the seed and
    nothing else, however many times the request comes back.
    """
    connector = _connector(tmp_path)
    provider = _RankingProvider()
    connector._provider = provider  # noqa: SLF001

    seed = FakeRequest("seed", list(SEED_TOKENS))
    connector.request_finished(seed, [0])
    _write_donor_capture(connector, seed, "seed", token_count=len(SEED_TOKENS))
    assert list(provider.donors) == ["seed"]

    served = [
        _serve_once(connector, FakeRequest(f"repeat-{i}", list(RECIPIENT_TOKENS))) for i in range(7)
    ]

    assert served == [SERVED_TOKENS] * 7
    assert list(provider.donors) == ["seed"]

    events = _audit(tmp_path)
    skipped = [e for e in events if e["event"] == "donor_registration_skipped"]
    assert len(skipped) == 7
    assert {e["reason"] for e in skipped} == {"no_captured_kv"}
    assert {e["capture_skip_reason"] for e in skipped} == {"served_request"}
    assert connector.stats_snapshot["donor_registration_skipped_no_captured_kv"] == 7
    # The capture skip itself is unchanged: this is a second gate behind it,
    # not a replacement for it.
    assert connector.stats_snapshot["capture_skipped_served_request"] == 7


def test_capture_served_requests_still_registers_the_recipient(tmp_path) -> None:
    """With capture on, a served request is captured and stays a donor.

    The new gate reads the capture skip, so turning the skip off has to turn
    the gate off with it, or ``capture_served_requests`` would silently lose
    its supply side.
    """
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "semantic_span_experimental",
                "provider": "local",
                "min_prompt_tokens": 8,
                "min_similarity": 0.3,
                "min_semantic_span": 8,
                "capture_served_requests": True,
                "kv_storage_path": str(tmp_path / "kv"),
                "audit_path": str(tmp_path / "audit.jsonl"),
            }
        ),
        cache_config=FakeCacheConfig(block_size=BLOCK_SIZE),
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)
    provider = _RankingProvider()
    connector._provider = provider  # noqa: SLF001

    seed = FakeRequest("seed", list(SEED_TOKENS))
    connector.request_finished(seed, [0])
    _write_donor_capture(connector, seed, "seed", token_count=len(SEED_TOKENS))

    assert _serve_once(connector, FakeRequest("recipient", list(RECIPIENT_TOKENS))) == SERVED_TOKENS

    assert list(provider.donors) == ["seed", "recipient"]
    assert "donor_registration_skipped_no_captured_kv" not in connector.stats_snapshot


def test_a_donor_with_no_captured_kv_is_cut_before_the_top_k(tmp_path) -> None:
    """The ranking gate, on its own: identical no-KV donors cannot fill the cut.

    Five donors identical to the query outrank the one donor that holds KV at
    every similarity there is, so a cut applied first leaves nothing usable
    behind it. The order is what matters, not the filter: both orders drop the
    same donors, only one of them keeps the seed.
    """
    usable = DonorRegistration(
        donor_id="seed",
        token_ids=list(SEED_TOKENS),
        prompt_text=None,
        model_id="m",
        namespace="n",
    )
    scored = [(0.82, usable)]
    scored += [
        (
            1.0,
            DonorRegistration(
                donor_id=f"repeat-{i}",
                token_ids=list(RECIPIENT_TOKENS),
                prompt_text=None,
                model_id="m",
                namespace="n",
                has_captured_kv=False,
            ),
        )
        for i in range(TOP_K)
    ]

    ranked = rank_candidates(scored, top_k=TOP_K)

    assert [donor.donor_id for _, donor in ranked] == ["seed"]
    # Cutting first and filtering after is the defect, stated as an assertion:
    # every survivor of that order is unusable.
    cut_first = sorted(scored, key=lambda item: item[0], reverse=True)[:TOP_K]
    assert all(not donor.has_captured_kv for _, donor in cut_first)


def test_local_provider_skips_donors_without_captured_kv() -> None:
    """The same gate through the provider the connector ships with."""
    from semblend_vllm_connector.provider import LocalSemanticProvider
    from semblend_vllm_connector.types import SemanticLookupRequest

    provider = LocalSemanticProvider(min_similarity=0.3, lookup_top_k=TOP_K)
    provider.register_donor(
        DonorRegistration(
            donor_id="seed",
            token_ids=list(SEED_TOKENS),
            prompt_text=None,
            model_id="m",
            namespace="n",
        )
    )
    for i in range(TOP_K):
        provider.register_donor(
            DonorRegistration(
                donor_id=f"repeat-{i}",
                token_ids=list(RECIPIENT_TOKENS),
                prompt_text=None,
                model_id="m",
                namespace="n",
                has_captured_kv=False,
            )
        )

    result = provider.lookup(
        SemanticLookupRequest(
            request_id="r",
            token_ids=list(RECIPIENT_TOKENS),
            prompt_text=None,
            model_id="m",
            namespace="n",
        )
    )

    assert result is not None
    assert result.donor_id == "seed"

"""The raw cache salt reaches SemBlend, so a donor carries a tenant key.

The connector's per-request namespace is engine-local: it digests the model,
tokenizer, block size, dtype, salt and adapter, so nothing outside this process
can reproduce it. SemBlend publishes a second value beside it — the tenant key
of ``tenant key v1``, derived from the raw ``cache_salt`` alone — and the
connector is the only place that holds that salt.

Contract and shared vector: ``docs/VLLM_CONNECTOR_CONTRACT.md`` ("Tenant key")
and ``tests/tenant_key_v1_vector.json``, whose bytes are identical in every
repository that implements the contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from test_connector_discovery import (
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeVllmConfig,
)

from semblend_vllm_connector import namespace as namespace_module
from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.namespace import cache_salt_for_request, namespace_for_request
from semblend_vllm_connector.providers.semblend import SemBlendPipelineProvider
from semblend_vllm_connector.types import DonorRegistration, SemanticLookupRequest

VECTOR = json.loads((Path(__file__).resolve().parent / "tenant_key_v1_vector.json").read_text())


def tenant_key_for_salt(cache_salt: str | None) -> str:
    """Local restatement of tenant key v1, so the gate is on the contract and
    not on whichever SemBlend release happens to be importable."""
    if not cache_salt:
        return "semblend:tenant:v1:none"
    return "semblend:tenant:v1:" + hashlib.sha256(cache_salt.encode("utf-8")).hexdigest()[:32]


class _Request:
    def __init__(self, cache_salt=None) -> None:
        self.cache_salt = cache_salt


class _RecordingPipeline:
    """Stands in for SemBlendPipeline with the current register_donor shape."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_donor(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str = "",
        extra_key: str | None = None,
        tenant: str | None = None,
        template: str | None = None,
        cache_salt=None,
    ) -> None:
        self.calls.append(
            {
                "request_id": request_id,
                "extra_key": extra_key,
                "cache_salt": cache_salt,
            }
        )


class _OlderPipeline:
    """A SemBlend release from before the tenant key existed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_donor(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str = "",
        extra_key: str | None = None,
    ) -> None:
        self.calls.append({"request_id": request_id, "extra_key": extra_key})


def _provider(pipeline) -> SemBlendPipelineProvider:
    provider = SemBlendPipelineProvider.__new__(SemBlendPipelineProvider)
    provider._config = SemBlendVllmConfig()  # noqa: SLF001
    provider._pipeline = pipeline  # noqa: SLF001
    return provider


def _donor(cache_salt: str | None) -> DonorRegistration:
    return DonorRegistration(
        donor_id="d1",
        token_ids=[1, 2, 3, 4],
        prompt_text="a prompt",
        model_id="m",
        namespace="vllm:9254316461d3e548",
        cache_salt=cache_salt,
    )


# ---------------------------------------------------------------------------
# The vector
# ---------------------------------------------------------------------------


def test_the_shared_vector_is_the_contract_this_connector_feeds() -> None:
    assert VECTOR["contract"] == "tenant key v1"
    assert tenant_key_for_salt(VECTOR["cache_salt"]) == VECTOR["tenant_key"]
    assert tenant_key_for_salt(None) == VECTOR["no_salt_tenant_key"]


# ---------------------------------------------------------------------------
# The salt is read off the request and put on the record
# ---------------------------------------------------------------------------


def test_cache_salt_is_read_from_the_request() -> None:
    assert cache_salt_for_request(_Request(VECTOR["cache_salt"])) == VECTOR["cache_salt"]


def test_a_request_without_a_salt_reads_as_none() -> None:
    assert cache_salt_for_request(_Request()) is None
    assert cache_salt_for_request(_Request("")) is None
    assert cache_salt_for_request(object()) is None


def test_a_request_type_without_the_field_warns_once(caplog) -> None:
    """An unsalted request is an answer; a missing field is a wiring gap.

    Both read as None, so without this the operator of a mismatched vLLM
    version sees every donor published tenant-less and no signal at all.
    """
    namespace_module._warned_missing_cache_salt_field = False  # noqa: SLF001
    try:
        with caplog.at_level(logging.WARNING, logger="semblend_vllm_connector.namespace"):
            assert cache_salt_for_request(object()) is None
            assert cache_salt_for_request(object()) is None
            # A request that HAS the field and no salt is not a wiring gap.
            assert cache_salt_for_request(_Request()) is None
    finally:
        namespace_module._warned_missing_cache_salt_field = False  # noqa: SLF001

    warnings = [r for r in caplog.records if "no cache_salt field" in r.message]
    assert len(warnings) == 1


def test_the_warning_never_carries_a_salt(caplog) -> None:
    namespace_module._warned_missing_cache_salt_field = False  # noqa: SLF001
    try:
        with caplog.at_level(logging.WARNING, logger="semblend_vllm_connector.namespace"):
            cache_salt_for_request(object())
    finally:
        namespace_module._warned_missing_cache_salt_field = False  # noqa: SLF001

    assert VECTOR["cache_salt"] not in caplog.text


def test_the_engine_local_namespace_is_not_the_tenant_key() -> None:
    """The two must never be confused: only one of them a router can derive."""
    namespace = namespace_for_request(SemBlendVllmConfig(), None, _Request(VECTOR["cache_salt"]))
    assert namespace.startswith("vllm:")
    assert namespace != VECTOR["tenant_key"]


def test_the_records_the_provider_receives_carry_the_salt() -> None:
    """The provider reads the salt off the record, never off a vLLM object."""
    assert _donor(VECTOR["cache_salt"]).cache_salt == VECTOR["cache_salt"]
    lookup = SemanticLookupRequest(
        request_id="r1",
        token_ids=[1, 2],
        prompt_text=None,
        model_id="m",
        namespace="vllm:9254316461d3e548",
        cache_salt=VECTOR["cache_salt"],
    )
    assert lookup.cache_salt == VECTOR["cache_salt"]


def test_the_salt_defaults_to_absent_on_both_records() -> None:
    """Every existing construction site keeps working unchanged."""
    donor = DonorRegistration(
        donor_id="d1", token_ids=[1], prompt_text=None, model_id="m", namespace="n"
    )
    lookup = SemanticLookupRequest(
        request_id="r1", token_ids=[1], prompt_text=None, model_id="m", namespace="n"
    )
    assert donor.cache_salt is None
    assert lookup.cache_salt is None


# ---------------------------------------------------------------------------
# The provider forwards it
# ---------------------------------------------------------------------------


def test_register_donor_forwards_the_raw_salt_to_semblend() -> None:
    pipeline = _RecordingPipeline()
    _provider(pipeline).register_donor(_donor(VECTOR["cache_salt"]))

    call = pipeline.calls[0]
    assert call["cache_salt"] == VECTOR["cache_salt"]
    # The engine-local key is still the connector's own namespace string.
    assert call["extra_key"] == "vllm:9254316461d3e548"
    assert tenant_key_for_salt(call["cache_salt"]) == VECTOR["tenant_key"]


def test_register_donor_forwards_an_absent_salt_as_none() -> None:
    pipeline = _RecordingPipeline()
    _provider(pipeline).register_donor(_donor(None))

    assert pipeline.calls[0]["cache_salt"] is None
    assert tenant_key_for_salt(pipeline.calls[0]["cache_salt"]) == VECTOR["no_salt_tenant_key"]


def test_an_older_semblend_without_the_argument_still_registers() -> None:
    """Passing an argument the installed release has no name for would raise,
    and the donor would be lost rather than merely tenant-less."""
    pipeline = _OlderPipeline()
    _provider(pipeline).register_donor(_donor(VECTOR["cache_salt"]))

    assert pipeline.calls == [{"request_id": "d1", "extra_key": "vllm:9254316461d3e548"}]


# ---------------------------------------------------------------------------
# End to end: the salt travels from the vLLM request to the donor record
# ---------------------------------------------------------------------------


class _CapturingProvider:
    def __init__(self) -> None:
        self.donors: list[DonorRegistration] = []
        self.lookups: list[SemanticLookupRequest] = []

    def lookup(self, request: SemanticLookupRequest):
        self.lookups.append(request)
        return None

    def register_donor(self, donor: DonorRegistration) -> None:
        self.donors.append(donor)


def _connector(tmp_path) -> SemBlendVllmConnector:
    settings = {
        "mode": "discovery_only",
        "provider": "local",
        "min_prompt_tokens": 4,
        "min_similarity": 0.3,
        "kv_storage_path": str(tmp_path),
        "log_decisions": False,
    }
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(settings),
        cache_config=FakeCacheConfig(block_size=4),
    )
    return SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)


def test_the_salt_travels_from_the_request_to_the_donor_record(tmp_path) -> None:
    connector = _connector(tmp_path)
    provider = _CapturingProvider()
    connector._provider = provider  # noqa: SLF001
    request = FakeRequest(
        request_id="r1",
        all_token_ids=list(range(32)),
        prompt_token_ids=list(range(32)),
        cache_salt=VECTOR["cache_salt"],
    )

    connector._register_donor_on_finish(request, "r1", [0, 1, 2, 3])  # noqa: SLF001

    assert provider.donors[0].cache_salt == VECTOR["cache_salt"]
    assert tenant_key_for_salt(provider.donors[0].cache_salt) == VECTOR["tenant_key"]
    # The engine-local namespace is still what isolates the donor in-engine.
    assert provider.donors[0].namespace.startswith("vllm:")


def test_a_request_without_a_salt_registers_a_tenant_less_donor(tmp_path) -> None:
    connector = _connector(tmp_path)
    provider = _CapturingProvider()
    connector._provider = provider  # noqa: SLF001
    request = FakeRequest(
        request_id="r1",
        all_token_ids=list(range(32)),
        prompt_token_ids=list(range(32)),
    )

    connector._register_donor_on_finish(request, "r1", [0, 1, 2, 3])  # noqa: SLF001

    assert provider.donors[0].cache_salt is None
    assert tenant_key_for_salt(provider.donors[0].cache_salt) == VECTOR["no_salt_tenant_key"]


def test_the_raw_salt_never_reaches_the_audit_trail(tmp_path) -> None:
    """The salt identifies a tenant; only its digest is safe to emit."""
    connector = _connector(tmp_path)
    provider = _CapturingProvider()
    connector._provider = provider  # noqa: SLF001
    events: list[dict] = []
    connector._audit_event = lambda name, **fields: events.append({"name": name, **fields})  # noqa: SLF001
    request = FakeRequest(
        request_id="r1",
        all_token_ids=list(range(32)),
        prompt_token_ids=list(range(32)),
        cache_salt=VECTOR["cache_salt"],
    )

    connector._register_donor_on_finish(request, "r1", [0, 1, 2, 3])  # noqa: SLF001

    assert events
    assert VECTOR["cache_salt"] not in json.dumps(events)

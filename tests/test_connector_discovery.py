from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector
from semblend_vllm_connector.types import (
    MaterializationKind,
    PendingLoad,
    SemBlendConnectorMetadata,
)


@dataclass
class FakeKvTransferConfig:
    kv_connector_extra_config: dict = field(default_factory=dict)


@dataclass
class FakeCacheConfig:
    block_size: int = 16


@dataclass
class FakeModelConfig:
    model: str = "test-model"
    tokenizer: str = "test-tokenizer"
    dtype: str = "float16"


@dataclass
class FakeVllmConfig:
    kv_transfer_config: FakeKvTransferConfig
    cache_config: FakeCacheConfig = field(default_factory=FakeCacheConfig)
    model_config: FakeModelConfig = field(default_factory=FakeModelConfig)


@dataclass
class FakeRequest:
    request_id: str
    all_token_ids: list[int]
    prompt_token_ids: list[int] | None = None
    cache_salt: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeBlocks:
    block_ids: tuple[list[int], ...]

    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...]:
        return self.block_ids


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list = field(default_factory=list)


@dataclass
class FakeForwardContext:
    attn_metadata: object = field(default_factory=object)
    no_compile_layers: dict = field(default_factory=dict)
    virtual_engine: int = 0


def test_discovery_only_registers_and_discovers_without_reporting_tokens() -> None:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "discovery_only",
                "provider": "local",
                "min_prompt_tokens": 4,
                "min_similarity": 0.3,
            }
        )
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)

    donor = FakeRequest("d1", [1, 2, 3, 4])
    connector.request_finished(donor, [0])

    recipient = FakeRequest("r1", [1, 2, 3, 9])
    matched, async_load = connector.get_num_new_matched_tokens(recipient, 0)

    assert (matched, async_load) == (0, False)
    stats = connector.stats_snapshot
    assert stats["donors_registered_total"] == 1
    assert stats["semantic_hits_total"] == 1
    assert stats["discovery_only_hits_total"] == 1


def test_request_finished_preserves_routing_metadata() -> None:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "discovery_only",
                "provider": "local",
                "min_prompt_tokens": 4,
            }
        )
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)

    donor = FakeRequest(
        "d1",
        [1, 2, 3, 4],
        metadata={
            "x-synapse-tenant": "wf-commercial",
            "x-synapse-template": "wf-rag-v1",
        },
    )
    connector.request_finished(donor, [0])

    registered = connector._provider._donors["d1"]  # noqa: SLF001
    assert registered.metadata["tenant"] == "wf-commercial"
    assert registered.metadata["template"] == "wf-rag-v1"


def test_prompt_tokens_preferred_over_generated_tokens() -> None:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "discovery_only",
                "provider": "local",
                "min_prompt_tokens": 4,
            }
        )
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)

    request = FakeRequest("r1", all_token_ids=[1, 2, 3, 4, 99], prompt_token_ids=[1, 2, 3, 4])

    assert connector._token_ids(request) == [1, 2, 3, 4]  # noqa: SLF001


def test_prompt_text_decodes_tokens_when_prompt_missing() -> None:
    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return "decoded policy text"

    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "discovery_only",
                "provider": "local",
                "enable_prompt_text": True,
            }
        )
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)
    connector._prompt_tokenizer = FakeTokenizer()  # noqa: SLF001

    request = FakeRequest("r1", all_token_ids=[1, 2, 3, 4], prompt_token_ids=[1, 2, 3])

    assert connector._prompt_text(request) == "decoded policy text"  # noqa: SLF001


def test_short_prompt_skips_lookup() -> None:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "discovery_only",
                "provider": "local",
                "min_prompt_tokens": 10,
            }
        )
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)
    matched, async_load = connector.get_num_new_matched_tokens(
        FakeRequest("r1", [1, 2, 3]),
        0,
    )
    assert (matched, async_load) == (0, False)
    assert connector.stats_snapshot["skipped_short_prompt"] == 1


def test_request_only_experimental_advertises_block_aligned_load(tmp_path) -> None:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "request_only_experimental",
                "provider": "local",
                "min_prompt_tokens": 4,
                "min_similarity": 0.3,
                "kv_storage_path": str(tmp_path),
                "max_materialized_tokens": 12,
            }
        ),
        cache_config=FakeCacheConfig(block_size=4),
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)

    donor = FakeRequest("d1", list(range(20)))
    connector.request_finished(donor, [0, 1, 2, 3, 4])
    namespace = connector._provider._donors["d1"].namespace  # noqa: SLF001
    os.makedirs(connector._donor_dir("d1", namespace))  # noqa: SLF001
    with open(connector._donor_metadata_path("d1", namespace), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": 12}, f)

    recipient = FakeRequest("r1", list(range(10)) + [100, 101, 102, 103])
    matched, async_load = connector.get_num_new_matched_tokens(recipient, 0)

    assert (matched, async_load) == (8, False)

    connector.update_state_after_alloc(recipient, FakeBlocks(([10, 11],)), matched)
    metadata = connector.build_connector_meta(FakeSchedulerOutput())
    assert len(metadata.loads) == 1
    assert metadata.loads[0].donor_id == "d1"
    assert metadata.loads[0].token_count == 8
    assert metadata.loads[0].block_ids == ([10, 11],)


def test_request_only_audit_records_hit_advertisement_and_allocation(tmp_path) -> None:
    audit_path = tmp_path / "vllm-semblend-audit.jsonl"
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "request_only_experimental",
                "provider": "local",
                "min_prompt_tokens": 4,
                "min_similarity": 0.3,
                "kv_storage_path": str(tmp_path / "kv"),
                "max_materialized_tokens": 12,
                "audit_path": str(audit_path),
            }
        ),
        cache_config=FakeCacheConfig(block_size=4),
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)

    donor = FakeRequest("d1", list(range(20)))
    connector.request_finished(donor, [0, 1, 2, 3, 4])
    namespace = connector._provider._donors["d1"].namespace  # noqa: SLF001
    os.makedirs(connector._donor_dir("d1", namespace))  # noqa: SLF001
    with open(connector._donor_metadata_path("d1", namespace), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": 12}, f)

    recipient = FakeRequest("r1", list(range(10)) + [100, 101, 102, 103])
    matched, _ = connector.get_num_new_matched_tokens(recipient, 0)
    connector.update_state_after_alloc(recipient, FakeBlocks(([10, 11],)), matched)

    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_event = {event["event"]: event for event in events}

    assert by_event["donor_registered"]["request_id"] == "d1"
    assert by_event["semantic_lookup_hit"]["request_id"] == "r1"
    assert by_event["request_only_load_advertised"]["tokens"] == 8
    assert by_event["load_allocated"]["block_id_count"] == 2


def test_runtime_noop_does_not_emit_materialized_event(tmp_path, monkeypatch) -> None:
    audit_path = tmp_path / "vllm-semblend-audit.jsonl"
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda filename: {}  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "request_only_experimental",
                "provider": "local",
                "audit_path": str(audit_path),
            }
        )
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.WORKER)
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

    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = [event["event"] for event in events]
    assert "runtime_materialization_started" in names
    assert "runtime_materialization_declined" in names
    assert "runtime_materialized" not in names
    assert connector.stats_snapshot["loads_rejected_no_kv_layers"] == 1


def test_request_only_default_caps_non_identical_reuse_to_exact_token_prefix(tmp_path) -> None:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "request_only_experimental",
                "provider": "local",
                "min_prompt_tokens": 4,
                "min_similarity": 0.1,
                "kv_storage_path": str(tmp_path),
                "max_materialized_tokens": 16,
            }
        ),
        cache_config=FakeCacheConfig(block_size=4),
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)

    donor = FakeRequest("d1", list(range(20)))
    connector.request_finished(donor, [0, 1, 2, 3, 4])
    namespace = connector._provider._donors["d1"].namespace  # noqa: SLF001
    os.makedirs(connector._donor_dir("d1", namespace))  # noqa: SLF001
    with open(connector._donor_metadata_path("d1", namespace), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": 16}, f)

    recipient = FakeRequest("r1", [0, 1, 2, 3, 4, 5, 200, 201, 202, 203, 204, 205])
    matched, async_load = connector.get_num_new_matched_tokens(recipient, 0)

    assert (matched, async_load) == (4, False)


def test_request_only_non_identical_reuse_requires_explicit_flag(tmp_path) -> None:
    vllm_config = FakeVllmConfig(
        FakeKvTransferConfig(
            {
                "mode": "request_only_experimental",
                "provider": "local",
                "min_prompt_tokens": 4,
                "min_similarity": 0.1,
                "kv_storage_path": str(tmp_path),
                "max_materialized_tokens": 16,
                "allow_non_identical_request_only": True,
            }
        ),
        cache_config=FakeCacheConfig(block_size=4),
    )
    connector = SemBlendVllmConnector(vllm_config, KVConnectorRole.SCHEDULER)

    donor = FakeRequest("d1", list(range(20)))
    connector.request_finished(donor, [0, 1, 2, 3, 4])
    namespace = connector._provider._donors["d1"].namespace  # noqa: SLF001
    os.makedirs(connector._donor_dir("d1", namespace))  # noqa: SLF001
    with open(connector._donor_metadata_path("d1", namespace), "w", encoding="utf-8") as f:  # noqa: SLF001
        json.dump({"token_count": 16}, f)

    recipient = FakeRequest("r1", [0, 1, 2, 3, 4, 5, 200, 201, 202, 203, 204, 205])
    matched, async_load = connector.get_num_new_matched_tokens(recipient, 0)

    assert (matched, async_load) == (8, False)

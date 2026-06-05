from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector


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
    cache_salt: str | None = None


@dataclass
class FakeBlocks:
    block_ids: tuple[list[int], ...]

    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...]:
        return self.block_ids


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list = field(default_factory=list)


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

from __future__ import annotations

from dataclasses import dataclass, field

from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.types import ReuseMode


@dataclass
class FakeKvTransferConfig:
    kv_connector_extra_config: dict = field(default_factory=dict)


@dataclass
class FakeModelConfig:
    model: str = "test-model"


@dataclass
class FakeVllmConfig:
    kv_transfer_config: FakeKvTransferConfig
    model_config: FakeModelConfig = field(default_factory=FakeModelConfig)


@dataclass
class FakeGetterKvTransferConfig:
    values: dict = field(default_factory=dict)

    def get_from_extra_config(self, key: str, default=None):
        return self.values.get(key, default)


def test_config_defaults_to_discovery_only() -> None:
    cfg = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert cfg.mode == ReuseMode.DISCOVERY_ONLY
    assert cfg.provider == "local"
    assert cfg.model_id == "test-model"


def test_config_reads_extra_config() -> None:
    cfg = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(
            FakeKvTransferConfig(
                {
                    "mode": "exact_prefix",
                    "provider": "semblend",
                    "min_prompt_tokens": 1024,
                    "min_similarity": 0.8,
                    "min_reuse_ratio": 0.4,
                    "embedder_type": "minilm",
                    "chunk_size": 128,
                }
            )
        )
    )
    assert cfg.mode == ReuseMode.EXACT_PREFIX
    assert cfg.provider == "semblend"
    assert cfg.min_prompt_tokens == 1024
    assert cfg.min_similarity == 0.8
    assert cfg.min_reuse_ratio == 0.4
    assert cfg.embedder_type == "minilm"
    assert cfg.chunk_size == 128


def test_config_parses_string_booleans() -> None:
    cfg = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(
            FakeKvTransferConfig(
                {
                    "register_donors": "false",
                    "enable_prompt_text": "true",
                    "log_decisions": "off",
                }
            )
        )
    )
    assert cfg.register_donors is False
    assert cfg.enable_prompt_text is True
    assert cfg.log_decisions is False


def test_config_supports_get_from_extra_config() -> None:
    cfg = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(
            FakeGetterKvTransferConfig(
                {
                    "provider": "semblend",
                    "min_prompt_tokens": 512,
                    "register_donors": "no",
                }
            )
        )
    )
    assert cfg.provider == "semblend"
    assert cfg.min_prompt_tokens == 512
    assert cfg.register_donors is False

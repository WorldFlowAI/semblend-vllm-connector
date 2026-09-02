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


def test_getter_path_carries_every_config_field() -> None:
    """Regression: min_semantic_span was missing from the getter key list,
    so getter-only vllm configs silently reverted it to the default. Guard
    the whole class of bug by exercising every field through the getter path.
    """
    supplied = {
        "mode": "semantic_span_experimental",
        "provider": "semblend",
        "provider_module": "x.mod",
        "provider_class": "XProvider",
        "model_id": "m",
        "min_prompt_tokens": 64,
        "min_semantic_span": 64,
        "min_similarity": 0.91,
        "min_reuse_ratio": 0.33,
        "embedder_type": "minilm",
        "chunk_size": 96,
        "max_donors": 7,
        "register_donors": "false",
        "skip_when_exact_prefix_ratio_at_least": 0.25,
        "lookup_top_k": 3,
        "enable_prompt_text": "true",
        "log_decisions": "false",
        "audit_path": "/tmp/a.jsonl",
        "kv_storage_path": "/tmp/kv",
        "max_materialized_tokens": 2048,
        "allow_non_identical_request_only": "true",
    }
    cfg = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(FakeGetterKvTransferConfig(dict(supplied)))
    )
    assert cfg.mode == ReuseMode.SEMANTIC_SPAN_EXPERIMENTAL
    assert cfg.min_semantic_span == 64
    assert cfg.min_prompt_tokens == 64
    assert cfg.min_similarity == 0.91
    assert cfg.min_reuse_ratio == 0.33
    assert cfg.chunk_size == 96
    assert cfg.max_donors == 7
    assert cfg.register_donors is False
    assert cfg.skip_when_exact_prefix_ratio_at_least == 0.25
    assert cfg.lookup_top_k == 3
    assert cfg.enable_prompt_text is True
    assert cfg.log_decisions is False
    assert cfg.audit_path == "/tmp/a.jsonl"
    assert cfg.kv_storage_path == "/tmp/kv"
    assert cfg.max_materialized_tokens == 2048
    assert cfg.allow_non_identical_request_only is True
    assert cfg.provider_module == "x.mod"
    assert cfg.provider_class == "XProvider"
    assert cfg.model_id == "m"
    assert cfg.embedder_type == "minilm"

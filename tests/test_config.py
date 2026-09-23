from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields

import pytest

from semblend_vllm_connector.config import SemBlendVllmConfig
from semblend_vllm_connector.types import ReuseMode


@pytest.fixture(autouse=True)
def _clean_semblend_env(monkeypatch) -> None:
    # Every test here asserts a default or an explicit value; an operator's
    # shell exporting SEMBLEND_VLLM_* would silently change the answer.
    for name in list(os.environ):
        if name.startswith("SEMBLEND_VLLM_"):
            monkeypatch.delenv(name)


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
                    "boundary_sliced_query_text": "false",
                    "log_decisions": "off",
                }
            )
        )
    )
    assert cfg.register_donors is False
    assert cfg.enable_prompt_text is True
    assert cfg.boundary_sliced_query_text is False
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
        "boundary_sliced_query_text": "false",
        "min_query_text_chars": 128,
        "log_decisions": "false",
        "audit_path": "/tmp/a.jsonl",
        "kv_storage_path": "/tmp/kv",
        "max_materialized_tokens": 2048,
        "allow_non_identical_request_only": "true",
        "capture_served_requests": "true",
        "capture_policy": "sampled",
        "capture_sample_rate": 0.25,
        "capture_hint_key": "donor_please",
        "capture_write_queue_depth": 8,
        "capture_write_close_timeout_s": 2.5,
        "donor_registration_retry_steps": 3,
        "kv_storage_backend": "MEMORY",
        "kv_memory_max_donors": 3,
        "min_boundary_tokens": 16,
        "evict_filled_blocks_from_prefix_cache": "false",
        "lookup_precheck": "false",
        "capture_async_copy": "false",
        "stage_hint_key": "Stage_Please",
        "multi_donor_spans": "false",
    }
    # A field added to the config without a line here is exactly the drift
    # this test exists to catch.
    assert set(supplied) == {f.name for f in fields(SemBlendVllmConfig)}
    cfg = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(FakeGetterKvTransferConfig(dict(supplied)))
    )
    assert cfg.mode == ReuseMode.SEMANTIC_SPAN_EXPERIMENTAL
    assert cfg.min_semantic_span == 64
    assert cfg.lookup_precheck is False
    assert cfg.capture_async_copy is False
    assert cfg.stage_hint_key == "stage_please"
    assert cfg.multi_donor_spans is False
    assert cfg.min_prompt_tokens == 64
    assert cfg.min_similarity == 0.91
    assert cfg.min_reuse_ratio == 0.33
    assert cfg.chunk_size == 96
    assert cfg.max_donors == 7
    assert cfg.register_donors is False
    assert cfg.skip_when_exact_prefix_ratio_at_least == 0.25
    assert cfg.lookup_top_k == 3
    assert cfg.enable_prompt_text is True
    assert cfg.boundary_sliced_query_text is False
    assert cfg.min_query_text_chars == 128
    assert cfg.log_decisions is False
    assert cfg.audit_path == "/tmp/a.jsonl"
    assert cfg.kv_storage_path == "/tmp/kv"
    assert cfg.max_materialized_tokens == 2048
    assert cfg.allow_non_identical_request_only is True
    assert cfg.provider_module == "x.mod"
    assert cfg.provider_class == "XProvider"
    assert cfg.model_id == "m"
    assert cfg.embedder_type == "minilm"
    assert cfg.capture_served_requests is True
    assert cfg.capture_policy == "sampled"
    assert cfg.capture_sample_rate == 0.25
    assert cfg.capture_hint_key == "donor_please"
    assert cfg.capture_write_queue_depth == 8
    assert cfg.kv_storage_backend == "memory"
    assert cfg.kv_memory_max_donors == 3
    assert cfg.min_boundary_tokens == 16
    assert cfg.evict_filled_blocks_from_prefix_cache is False


def test_env_path_defaults_match_dataclass_defaults() -> None:
    """The defaults are written twice (field default and the from_vllm_config
    fallback); a value changed in one place only is a config that reads
    differently depending on how it was built.
    """
    from_env = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert from_env == SemBlendVllmConfig(model_id="test-model")


def test_default_min_prompt_tokens_can_carry_a_span() -> None:
    """A default gap between min_prompt_tokens and min_semantic_span admits
    prompts to lookup that block_align_spans can never serve.
    """
    cfg = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert cfg.min_prompt_tokens == 512
    assert cfg.min_prompt_tokens >= cfg.min_semantic_span
    assert cfg.validation_warnings() == ()


def test_semantic_span_mode_warns_when_min_prompt_tokens_is_below_min_semantic_span(
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING, logger="semblend_vllm_connector"):
        cfg = SemBlendVllmConfig.from_vllm_config(
            FakeVllmConfig(
                FakeKvTransferConfig(
                    {
                        "mode": "semantic_span_experimental",
                        "min_prompt_tokens": 256,
                        "min_semantic_span": 512,
                    }
                )
            )
        )
    # Legal but self-defeating: the config is built, the operator is told.
    assert cfg.min_prompt_tokens == 256
    (message,) = cfg.validation_warnings()
    assert "min_prompt_tokens=256" in message
    assert "min_semantic_span=512" in message
    logged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(message in r.getMessage() for r in logged), "inconsistency was not logged"


def test_min_prompt_tokens_gap_is_silent_where_min_semantic_span_is_inert(caplog) -> None:
    """min_semantic_span only gates the semantic-span path; the same gap in
    exact_prefix mode costs nothing and must not nag."""
    with caplog.at_level(logging.WARNING, logger="semblend_vllm_connector"):
        cfg = SemBlendVllmConfig.from_vllm_config(
            FakeVllmConfig(
                FakeKvTransferConfig(
                    {"mode": "exact_prefix", "min_prompt_tokens": 256, "min_semantic_span": 512}
                )
            )
        )
    assert cfg.validation_warnings() == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_min_boundary_tokens_defaults_off() -> None:
    cfg = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert cfg.min_boundary_tokens == 0


def test_min_boundary_tokens_reads_extra_config() -> None:
    cfg = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(FakeKvTransferConfig({"min_boundary_tokens": "16"}))
    )
    assert cfg.min_boundary_tokens == 16


def test_min_boundary_tokens_reads_env_and_extra_config_wins(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_VLLM_MIN_BOUNDARY_TOKENS", "32")
    from_env = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert from_env.min_boundary_tokens == 32

    explicit = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(FakeKvTransferConfig({"min_boundary_tokens": 16}))
    )
    assert explicit.min_boundary_tokens == 16


def test_min_boundary_tokens_unparseable_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_VLLM_MIN_BOUNDARY_TOKENS", "one-block")
    cfg = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert cfg.min_boundary_tokens == 0


def test_prefix_cache_eviction_defaults_on() -> None:
    """The safe behaviour is the default: an operator who never heard of the
    knob gets the contamination mitigation."""
    cfg = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert cfg.evict_filled_blocks_from_prefix_cache is True


def test_prefix_cache_eviction_parses_bool_and_string_forms() -> None:
    for raw in (False, "false", "off", "no", "0"):
        cfg = SemBlendVllmConfig.from_vllm_config(
            FakeVllmConfig(FakeKvTransferConfig({"evict_filled_blocks_from_prefix_cache": raw}))
        )
        assert cfg.evict_filled_blocks_from_prefix_cache is False, raw
    for raw in (True, "true", "on", "yes", "1"):
        cfg = SemBlendVllmConfig.from_vllm_config(
            FakeVllmConfig(FakeKvTransferConfig({"evict_filled_blocks_from_prefix_cache": raw}))
        )
        assert cfg.evict_filled_blocks_from_prefix_cache is True, raw


def test_prefix_cache_eviction_reads_env_and_extra_config_wins(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_VLLM_EVICT_FILLED_BLOCKS_FROM_PREFIX_CACHE", "false")
    from_env = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert from_env.evict_filled_blocks_from_prefix_cache is False

    explicit = SemBlendVllmConfig.from_vllm_config(
        FakeVllmConfig(FakeKvTransferConfig({"evict_filled_blocks_from_prefix_cache": True}))
    )
    assert explicit.evict_filled_blocks_from_prefix_cache is True


def test_prefix_cache_eviction_unparseable_falls_back_to_the_safe_default(
    monkeypatch,
) -> None:
    """A typo must not silently leave approximate KV exact-matchable."""
    monkeypatch.setenv("SEMBLEND_VLLM_EVICT_FILLED_BLOCKS_FROM_PREFIX_CACHE", "nope")
    cfg = SemBlendVllmConfig.from_vllm_config(FakeVllmConfig(FakeKvTransferConfig()))
    assert cfg.evict_filled_blocks_from_prefix_cache is True

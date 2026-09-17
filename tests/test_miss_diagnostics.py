"""A recorded miss says WHY it missed, and reports only what was measured.

Of 450 requests that carried a reusable donor in phase-0 stream B, 17 produced
a semantic match. The loss could not be diagnosed from the collected runs,
because ``semantic_lookup_miss`` recorded latency, boundary and query length
and nothing about the decision. The provider protocol answers with a result or
None, so the reason travels beside it.

A first draft splatted every field of the pipeline's result into the row. On a
real ``no_donor_match`` the pipeline returns a FRESH result carrying only the
reason, so ``similarity`` read 0.0, ``fuzzy_confidence`` 1.0 and
``confidence_tier`` "exact" on every miss -- dataclass defaults audited as
measurements. These tests use the real result type where it matters and pin
that a value the pipeline did not compute is None, never a default.
"""

from __future__ import annotations

import json
from enum import Enum

import numpy as np
import pytest
from test_boundary_query_text import (
    _DOCUMENT_TOKENS,
    _PREAMBLE_TOKENS,
    _events,
    _WordTokenizer,
)
from test_connector_discovery import (
    FakeCacheConfig,
    FakeKvTransferConfig,
    FakeRequest,
    FakeVllmConfig,
)

from semblend_vllm_connector._vllm_compat import KVConnectorRole
from semblend_vllm_connector.connector import SemBlendVllmConnector, _json_scalar
from semblend_vllm_connector.providers.semblend import (
    MISS_DIAGNOSTIC_KEYS,
    SemBlendPipelineProvider,
    _miss_diagnostics,
)
from semblend_vllm_connector.types import MaterializationKind, SemanticLookupResult

# --- connector side -----------------------------------------------------------


class _SilentProvider:
    """Answers lookups, explains nothing. The pre-0.2.7 protocol."""

    def __init__(self) -> None:
        self.result: SemanticLookupResult | None = None

    def register_donor(self, donor) -> None:
        return None

    def lookup(self, request):
        return self.result

    def clear_donors(self) -> None:
        return None


class _ExplainingProvider(_SilentProvider):
    def __init__(self) -> None:
        super().__init__()
        self.diagnostics: dict | None = None
        self.raises = False

    def last_lookup_diagnostics(self):
        if self.raises:
            raise RuntimeError("provider is unwell")
        return self.diagnostics


def _connector(tmp_path, provider):
    cfg = {
        "mode": "discovery_only",
        "provider": "local",
        "min_prompt_tokens": 64,
        "enable_prompt_text": True,
        "skip_when_exact_prefix_ratio_at_least": 1.0,
        "audit_path": str(tmp_path / "audit.jsonl"),
        "kv_storage_path": str(tmp_path / "kv"),
    }
    connector = SemBlendVllmConnector(
        FakeVllmConfig(FakeKvTransferConfig(cfg), cache_config=FakeCacheConfig(block_size=16)),
        KVConnectorRole.SCHEDULER,
    )
    connector._provider = provider  # noqa: SLF001
    connector._prompt_tokenizer = _WordTokenizer()  # noqa: SLF001
    return connector


def _look_up(connector, request_id: str = "r1"):
    recipient = FakeRequest(request_id, [*_PREAMBLE_TOKENS, *_DOCUMENT_TOKENS])
    return connector.get_num_new_matched_tokens(recipient, len(_PREAMBLE_TOKENS))


def test_the_reason_a_lookup_missed_reaches_the_audit(tmp_path) -> None:
    provider = _ExplainingProvider()
    provider.diagnostics = {
        "rejection_reason": "fuzzy_low_reuse",
        "donor_count": 1236,
        "donor_count_scope": "namespace",
        "best_similarity": 0.93,
        "reuse_ratio": 0.12,
    }
    connector = _connector(tmp_path, provider)

    _look_up(connector)

    miss = _events(tmp_path, "semantic_lookup_miss")[0]
    assert miss["miss_rejection_reason"] == "fuzzy_low_reuse"
    assert miss["miss_best_similarity"] == 0.93
    assert miss["miss_reuse_ratio"] == 0.12
    assert miss["miss_donor_count"] == 1236
    assert miss["miss_donor_count_scope"] == "namespace"


def test_a_provider_that_explains_nothing_still_records_a_miss(tmp_path) -> None:
    connector = _connector(tmp_path, _SilentProvider())
    _look_up(connector)
    miss = _events(tmp_path, "semantic_lookup_miss")[0]
    assert miss["event"] == "semantic_lookup_miss"
    assert not [key for key in miss if key.startswith("miss_")]


def test_a_provider_whose_explanation_raises_still_records_a_miss(tmp_path) -> None:
    provider = _ExplainingProvider()
    provider.raises = True
    connector = _connector(tmp_path, provider)
    _look_up(connector)
    miss = _events(tmp_path, "semantic_lookup_miss")[0]
    assert miss["event"] == "semantic_lookup_miss"
    assert not [key for key in miss if key.startswith("miss_")]


class _Tier(Enum):
    LOW = "low"


def test_numpy_and_enum_values_do_not_cost_the_row(tmp_path) -> None:
    """The audit writer is plain json.dumps and a failure there drops the WHOLE
    event; a cosine kernel hands back numpy floats."""
    provider = _ExplainingProvider()
    provider.diagnostics = {
        "rejection_reason": "no_donor_match",
        "donor_count": np.int64(12),
        "donor_count_scope": "namespace",
        "best_similarity": np.float32(0.41),
        "reuse_ratio": np.bool_(False),
        "tier": _Tier.LOW,
    }
    connector = _connector(tmp_path, provider)

    _look_up(connector)

    line = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    row = json.loads(line)
    assert row["event"] == "semantic_lookup_miss"
    assert row["miss_donor_count"] == 12
    assert abs(row["miss_best_similarity"] - 0.41) < 1e-6
    assert row["miss_reuse_ratio"] is False
    assert row["miss_tier"] == "low"
    assert connector.stats_snapshot.get("audit_errors_total", 0) == 0


def test_json_scalar_handles_the_shapes_a_provider_returns() -> None:
    assert _json_scalar(np.float32(0.5)) == 0.5
    assert _json_scalar(np.int64(3)) == 3
    assert _json_scalar(np.bool_(True)) is True
    assert _json_scalar(_Tier.LOW) == "low"
    assert _json_scalar(None) is None
    assert _json_scalar("x") == "x"
    assert _json_scalar(object()).startswith("<object")


# --- adapter side, against the REAL pipeline result -------------------------

semblend_pipeline = pytest.importorskip("semblend_core.pipeline")
PipelineResult = semblend_pipeline.PipelineResult


class _Store:
    size = 1236

    def visible_donors(self, key):
        return {"vllm:tenant-a": 7}.get(key, 0)


class _StoreWithoutScope:
    size = 1236


class _Pipeline:
    def __init__(self, store) -> None:
        self._donor_store = store
        self.result = None

    def find_donor(self, **kwargs):
        return self.result


class _Config:
    lookup_top_k = 5


class _Lookup:
    token_ids = [1, 2, 3]
    prompt_text = "text"
    namespace = "vllm:tenant-a"


def _adapter(store, result):
    provider = object.__new__(SemBlendPipelineProvider)
    provider._config = _Config()  # noqa: SLF001
    provider._last_miss = None  # noqa: SLF001
    provider._pipeline = _Pipeline(store)  # noqa: SLF001
    provider._pipeline.result = result  # noqa: SLF001
    return provider


def test_every_miss_carries_the_same_keys() -> None:
    for reason in ("no_donor_match", "fuzzy_low_reuse", "low_consumable_coverage"):
        diagnostics = _miss_diagnostics(
            PipelineResult(found=False, rejection_reason=reason), _Pipeline(_Store()), "vllm:tenant-a"
        )
        assert tuple(diagnostics) == MISS_DIAGNOSTIC_KEYS
    assert tuple(_miss_diagnostics(None, _Pipeline(_Store()), "vllm:tenant-a")) == MISS_DIAGNOSTIC_KEYS


def test_a_total_miss_reports_no_similarity_rather_than_a_default() -> None:
    """The pipeline returns a fresh result for no_donor_match: similarity 0.0,
    fuzzy_confidence 1.0, confidence_tier 'exact' are all defaults. None of
    them is a measurement and none may be audited as one."""
    result = PipelineResult(found=False, rejection_reason="no_donor_match")
    assert result.similarity == 0.0 and result.fuzzy_confidence == 1.0  # the trap

    diagnostics = _miss_diagnostics(result, _Pipeline(_Store()), "vllm:tenant-a")

    assert diagnostics["rejection_reason"] == "no_donor_match"
    assert diagnostics["best_similarity"] is None
    assert diagnostics["reuse_ratio"] is None
    assert "fuzzy_confidence" not in diagnostics
    assert "confidence_tier" not in diagnostics


def test_a_candidate_that_scored_and_was_cut_reports_its_numbers() -> None:
    """semblend >= 0.3.24 sets these on the decline paths that held a candidate."""
    result = PipelineResult(
        found=False, rejection_reason="fuzzy_low_reuse", similarity=0.93, reuse_ratio=0.12
    )

    diagnostics = _miss_diagnostics(result, _Pipeline(_Store()), "vllm:tenant-a")

    assert diagnostics["best_similarity"] == 0.93
    assert diagnostics["reuse_ratio"] == 0.12


def test_the_donor_count_is_the_namespace_this_lookup_could_see() -> None:
    """A tenant whose namespace holds no donors misses everything while the
    whole store reports a thousand; the reader must not be told a thousand."""
    result = PipelineResult(found=False, rejection_reason="no_donor_match")

    seen = _miss_diagnostics(result, _Pipeline(_Store()), "vllm:tenant-a")
    empty = _miss_diagnostics(result, _Pipeline(_Store()), "vllm:tenant-nobody")

    assert (seen["donor_count"], seen["donor_count_scope"]) == (7, "namespace")
    assert (empty["donor_count"], empty["donor_count_scope"]) == (0, "namespace")


def test_a_store_without_a_scoped_count_says_the_number_is_the_whole_store() -> None:
    result = PipelineResult(found=False, rejection_reason="no_donor_match")
    diagnostics = _miss_diagnostics(result, _Pipeline(_StoreWithoutScope()), "vllm:tenant-a")
    assert (diagnostics["donor_count"], diagnostics["donor_count_scope"]) == (1236, "store")


def test_exception_text_does_not_reach_the_audit() -> None:
    """The pipeline reports internal failures as 'pipeline_error: <message>'
    and a message can carry prompt text; only the class of failure travels."""
    result = PipelineResult(
        found=False, rejection_reason="pipeline_error: alignment failed for 'Patient MRN 5512'"
    )
    diagnostics = _miss_diagnostics(result, _Pipeline(_Store()), "vllm:tenant-a")
    assert diagnostics["rejection_reason"] == "pipeline_error"
    assert "5512" not in json.dumps(diagnostics)


def test_the_real_adapter_records_a_reason_and_clears_it_on_a_hit() -> None:
    provider = _adapter(_Store(), PipelineResult(found=False, rejection_reason="no_donor_match"))

    assert provider.lookup(_Lookup()) is None
    assert provider.last_lookup_diagnostics()["rejection_reason"] == "no_donor_match"
    assert provider.last_lookup_diagnostics()["donor_count"] == 7

    provider._pipeline.result = PipelineResult(found=True, donor_id="d1", similarity=0.97)  # noqa: SLF001
    assert provider.lookup(_Lookup()) is not None
    assert provider.last_lookup_diagnostics() is None


def test_a_failing_diagnostic_does_not_turn_a_miss_into_a_provider_error() -> None:
    class _ExplodingStore:
        @property
        def size(self):
            raise RuntimeError("remote store unreachable")

        def visible_donors(self, key):
            raise RuntimeError("remote store unreachable")

    provider = _adapter(
        _ExplodingStore(), PipelineResult(found=False, rejection_reason="no_donor_match")
    )

    assert provider.lookup(_Lookup()) is None  # a miss, not an exception
    diagnostics = provider.last_lookup_diagnostics()
    assert diagnostics["rejection_reason"] == "no_donor_match"
    assert diagnostics["donor_count"] is None


def test_the_hit_result_type_is_unchanged() -> None:
    provider = _adapter(_Store(), PipelineResult(found=True, donor_id="d1", similarity=0.97))
    result = provider.lookup(_Lookup())
    assert result.materialization_kind is MaterializationKind.DISCOVERY_ONLY
    assert result.similarity == 0.97

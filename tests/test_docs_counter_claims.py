"""What the shipped docs say about the prefix-cache counters has to be true.

These counters are read by people who never open the source: the changelog and
the connector contract are the whole interface. Two claims in them were checked
against the code and did not hold, so each one is pinned here against the
behaviour it describes.
"""

from __future__ import annotations

from pathlib import Path

from test_b6_contamination import (
    PROMPT_TOKENS,
    FakeRequest,
    _audit_events,
    _bound_connector,
)
from test_b6_eviction_knob import _run_fill_script

REPO_ROOT = Path(__file__).resolve().parent.parent


def _doc(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _prose(name: str) -> str:
    """One doc, line wrapping removed, so a claim is one searchable string."""
    return " ".join(_doc(name).split())


def _events(root, name: str) -> list[dict]:
    return [event for event in _audit_events(root) if event["event"] == name]


def _both_arms(tmp_path):
    """The same fill script on each arm, audited into its own directory."""
    on, on_pool = _bound_connector(tmp_path / "on")
    off, off_pool = _bound_connector(tmp_path / "off", evict_filled_blocks_from_prefix_cache=False)
    _run_fill_script(on, on_pool, FakeRequest("r1", list(range(PROMPT_TOKENS))))
    _run_fill_script(off, off_pool, FakeRequest("r1", list(range(PROMPT_TOKENS))))
    return on, off


def test_an_event_is_not_written_for_every_fill_step(tmp_path) -> None:
    """A step that finds nothing newly cached writes nothing, and docs say so.

    ``_run_fill_script`` drives four scheduling steps: an admission, two
    chunked-prefill continuations that allocate nothing, and a decode step that
    appends a block. The continuations scan a tail this connector has already
    settled, so they write no event. A scorer counting events as a proxy for
    fill steps -- or asserting the mitigation ran by finding one event per step
    -- would be wrong, so the changelog must not promise one per step.
    """
    root = tmp_path / "off"
    connector, pool = _bound_connector(root, evict_filled_blocks_from_prefix_cache=False)

    steps = _run_fill_script(connector, pool, FakeRequest("r1", list(range(PROMPT_TOKENS))))

    events = [
        event
        for event in _audit_events(root)
        if event["event"] == "prefix_cache_blocks_left_cached"
    ]
    assert len(steps) == 4
    assert len(events) == 2

    changelog = _doc("CHANGELOG.md")
    assert "audit event per fill step" not in changelog
    assert "prefix_cache_blocks_left_cached" in changelog


def test_the_contract_does_not_claim_readmission_lands_on_new_blocks() -> None:
    """A readmitted request is routinely handed its own just-freed blocks back.

    The contract used to justify the two counters' different scopes with the
    claim that readmission re-hashes the prefix onto *new* physical blocks. It
    does not have to, and under the memory pressure that caused the preemption
    it usually does not (see tests/test_b6_arm_comparability.py). The scope
    difference is real, so the contract has to state it without that rationale
    and name the pair that is comparable across arms.
    """
    contract = _doc("docs/VLLM_CONNECTOR_CONTRACT.md")

    assert "onto new physical blocks" not in contract
    assert "prefix_cache_distinct_blocks_evicted" in contract
    assert "prefix_cache_distinct_blocks_left_cached" in contract


def test_the_docs_do_not_claim_the_control_arm_re_finds_every_block(tmp_path) -> None:
    """The arms' per-pass counts match while the block table is stable.

    The changelog and the contract justified the distinct counters with a
    mechanism the code does not have: that with eviction off *every* later pass
    finds every block again. Both arms act only on filled blocks that are cached
    and not yet settled, and both settle what they acted on, so a request that
    keeps one block table reports each block once on either arm -- as this run
    shows, by producing the same per-pass sequence twice. The arms diverge only
    where the table is replaced (tests/test_b6_arm_comparability.py). A reader
    who believed the old sentence would expect the count to repeat every step
    and read a correct run as broken instrumentation.
    """
    _both_arms(tmp_path)

    evicted = [e["blocks_evicted"] for e in _events(tmp_path / "on", "prefix_cache_blocks_evicted")]
    left = [
        e["blocks_left_cached"]
        for e in _events(tmp_path / "off", "prefix_cache_blocks_left_cached")
    ]
    assert evicted == left == [97, 1]

    for name in ("docs/VLLM_CONNECTOR_CONTRACT.md", "CHANGELOG.md"):
        prose = _prose(name)
        assert "every later pass finds every block again" not in prose, name
        assert "found by every later pass" not in prose, name
        # What actually decides it, named where the old claim stood.
        assert "same block table" in prose, name


def test_the_contract_documents_the_only_event_that_names_block_ids(tmp_path) -> None:
    """``block_ids_left_cached`` is load-bearing, so the contract has to name it.

    The control arm's event carries the ids it found cached, and the union of
    them over a run is that arm's whole result. The eviction event deliberately
    carries no counterpart -- after its pass the set is empty by construction --
    so the field table has to say which event has ids and the contract must not
    promise a field that is never written.
    """
    _both_arms(tmp_path)

    left = _events(tmp_path / "off", "prefix_cache_blocks_left_cached")[0]
    evicted = _events(tmp_path / "on", "prefix_cache_blocks_evicted")[0]
    assert left["block_ids_left_cached"] == list(range(103, 200))
    assert [key for key in evicted if key.startswith("block_ids")] == []

    contract = _prose("docs/VLLM_CONNECTOR_CONTRACT.md")
    assert "`block_ids_left_cached`" in contract
    assert "block_ids_evicted" not in contract

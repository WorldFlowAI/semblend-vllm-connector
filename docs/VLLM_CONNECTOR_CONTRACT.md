# vLLM Connector Contract

This connector is intentionally narrow. It should behave like a well-formed
vLLM `KVConnectorBase_V1` implementation before it behaves like a semantic KV
experiment.

## Loading Contract

The connector must be loadable out of tree:

```json
{
  "kv_connector": "SemBlendVllmConnector",
  "kv_connector_module_path": "semblend_vllm_connector.connector",
  "kv_role": "kv_both"
}
```

All SemBlend-specific configuration belongs in `kv_connector_extra_config` or
documented `SEMBLEND_VLLM_*` environment variables.

## Scheduler Contract

`get_num_new_matched_tokens()` may be called multiple times for the same request.
It should avoid irreversible side effects until vLLM has allocated destination
state through `update_state_after_alloc()`.

The returned token count is a load promise, not a hint. Return a positive count
only when the connector can load those tokens into vLLM-owned KV state.

When returning zero tokens, the async-load boolean must be `False`.

## Worker Contract

Worker-side loading should only consume metadata produced by the scheduler-side
connector. It should validate:

- namespace;
- model/tokenizer/cache compatibility;
- donor freshness;
- block shape and count;
- destination ownership;
- supported attention backend and position handling.

If any validation fails, inference should continue through normal prefill when
vLLM is configured with `kv_load_failure_policy="recompute"`.

## Exact Prefix Compatibility

Exact vLLM prefix caching remains authoritative and should run first.

Semantic donor discovery must not insert non-identical KV blocks into vLLM's
exact prefix cache under recipient prompt hashes. Non-identical reuse requires
request-local semantics or an explicit upstream cache-commit policy.

## Materialization Modes

| Mode | Connector behavior |
| --- | --- |
| `discovery_only` | Lookup/register donors, return zero external tokens. |
| `exact_prefix` | Future mode for exact-equivalent, prefix-shaped, block-valid loads. |
| `request_only_experimental` | Block-aligned prefix materialization for isolated validation; run with vLLM prefix caching disabled so non-identical KV is not committed as exact cache state. |
| `segmented_experimental` | Requires segmented/sparse execution support and backend validation. |

## Metrics Contract

The connector should separate:

- semantic discovery;
- materialization advertised;
- backend materialization confirmed;
- backend decline;
- fallback/error;
- lookup latency.

Only backend-confirmed materialization should count as confirmed KV reuse.

## Audit Contract

When `audit_path` / `SEMBLEND_VLLM_AUDIT_PATH` is set, the connector emits
JSONL events with `schema_version=2`.

Benchmark and product gates should treat:

- `semantic_lookup_hit` as discovery evidence only;
- `request_only_load_advertised` / `exact_prefix_load_advertised` as a load
  promise only;
- `runtime_materialized` as backend-confirmed KV materialization;
- `runtime_materialization_declined` as a declined load with a reason;
- `prefix_cache_blocks_evicted` as connector-filled blocks removed from vLLM's
  exact prefix cache, and `prefix_cache_blocks_left_cached` as connector-filled
  blocks deliberately left in it. The second one is written only when
  `evict_filled_blocks_from_prefix_cache` is false, which is a measurement-only
  setting: it leaves approximate KV matchable by a later exact hit, so a run
  that emits it is a contaminated control, not a product configuration. One
  event is written per pass that finds filled blocks cached, which is not the
  same as one per scheduling step: a pass whose scan finds nothing writes
  nothing.

Do not count semantic hits or advertised loads as materialized KV reuse unless a
matching `runtime_materialized` event exists and negative controls remain at
zero.

### Which prefix-cache number to compare across arms

Both events carry the request's join key and two units, because the two arms
cannot be compared in the first unit:

| Field | Unit |
| --- | --- |
| `blocks_evicted` / `blocks_left_cached` | blocks this pass acted on |
| `blocks_evicted_cumulative` / `blocks_left_cached_cumulative` | the same, running, for the current admission; restarts on readmission |
| `distinct_blocks_evicted` / `distinct_blocks_left_cached` | of those, the ones this request had not been counted for yet |
| `distinct_blocks_evicted_cumulative` / `distinct_blocks_left_cached_cumulative` | distinct physical blocks for the whole request, readmissions included |
| `block_ids_left_cached` | the block ids this pass found cached -- on the left-cached event only, see below |

The per-pass counts are arm-dependent, but not on every pass. On both arms a
pass acts only on the filled blocks that are cached *and not yet settled*, and
both arms settle what they acted on, so for as long as a request keeps the same
block table each of its blocks is reported once on either arm and the two arms'
per-pass sequences are identical.

The arms part when the block table is replaced. A preemption and readmission
re-hashes the prefix, and what was settled on the old table is a claim about
blocks this request may no longer hold, so the connector clears it and scans
the table again. With eviction off the blocks were never removed and are all
re-found. With eviction on they are gone from the hash table unless the engine
re-hashed them -- which it does when the request is handed its own just-freed
blocks back, the common case, since memory pressure is why it was preempted --
and those are evicted a second time. `prefix_cache_blocks_evicted` and
`prefix_cache_blocks_left_cached` therefore diverge across arms for identical
contamination: they are eviction work and cache residency respectively, not a
matched pair.

**For the A/B, use the distinct pair**: `prefix_cache_distinct_blocks_evicted`
against `prefix_cache_distinct_blocks_left_cached`, or the per-request maximum
of `distinct_blocks_evicted_cumulative` against
`distinct_blocks_left_cached_cumulative`. Both count distinct physical blocks
per request, on both arms, so identical contamination gives identical totals.
They are per request and not per admission: they survive a readmission,
including one the scheduler drops before the span is served.

Only `prefix_cache_blocks_left_cached` names block ids. The union of
`block_ids_left_cached` over a run is the set of blocks that stayed
exact-matchable, which is the control arm's whole result. The eviction event
has no counterpart field: after its pass that set is empty by construction, and
listing every evicted id on every pass of the default path would inflate the
audit log of every deployment for a number nothing joins on. The arms are
compared on the distinct counters above, not on ids, and the eviction arm's
block set is deliberately not recoverable from the log.

### Join Key

Every event that names a request carries four join fields:

| Field | Meaning |
| --- | --- |
| `connector_id` | The emitting connector instance. Both roles append to the same file and number requests independently. |
| `request_id` | The vLLM request id. |
| `request_seq` | The request's arrival number within that connector instance, stamped at first sight and never re-derived. vLLM reuses request ids, so `request_id` alone does not separate two uses of one id. |
| `event_seq` | The event's position within that request, so events written in the same second can still be ordered. |

`request_seq` and `event_seq` are per connector instance, so a cross-role join
(a scheduler advertise against a worker materialization) is on `request_id`
alone; within one role, join on `connector_id` plus `request_seq`.

An event that names no request (`connector_initialized`, and the once-per-
connector shape warnings) carries `connector_id` only.

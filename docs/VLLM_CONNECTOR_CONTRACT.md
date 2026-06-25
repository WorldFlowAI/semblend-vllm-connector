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
JSONL events with `schema_version=1`.

Benchmark and product gates should treat:

- `semantic_lookup_hit` as discovery evidence only;
- `request_only_load_advertised` / `exact_prefix_load_advertised` as a load
  promise only;
- `runtime_materialized` as backend-confirmed KV materialization;
- `runtime_materialization_declined` as a declined load with a reason.

Do not count semantic hits or advertised loads as materialized KV reuse unless a
matching `runtime_materialized` event exists and negative controls remain at
zero.

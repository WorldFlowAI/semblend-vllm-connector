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

### Lookup skip reasons

A request that never reaches the provider leaves one of these instead of a hit
or a miss. Each carries the join key, `attempt`, `prompt_tokens`, and the
numbers the decision was taken on, so a lookup rate is a ratio against a
denominator that is written down rather than inferred. All of them are decided
before any embedding is computed.

| Event | Skipped because | Extra fields |
| --- | --- | --- |
| `lookup_skipped_incompatible_engine` | the startup gate declined this engine | `reason` |
| `lookup_skipped_short_prompt` | the prompt is below `min_prompt_tokens` | `min_prompt_tokens` |
| `lookup_skipped_exact_ratio` | vLLM's own prefix cache already covers enough of the prompt | `boundary`, `exact_ratio`, `threshold` |
| `lookup_skipped_below_min_boundary` | the local prefix-cache boundary is below `min_boundary_tokens` | `boundary`, `min_boundary_tokens`, `block_size` |
| `lookup_skipped_remaining_below_min_span` | fewer tokens are left after the boundary than `min_semantic_span` | `boundary`, `remaining`, `min_semantic_span`, `block_size` |

The last one is semantic-span mode only, and it is not a variant of the
exact-ratio gate: a deployment that sets `skip_when_exact_prefix_ratio_at_least`
to 1.0 to keep every near-exact prompt eligible still cannot serve a tail
shorter than its own span floor, because `block_align_spans` and the floor
re-applied after the clamps both drop it. Without this gate that tail costs an
embedding and a provider round trip per scheduling attempt and can only end in
`semantic_span_boundary_missed`.

### Donor registration skip reasons

`donor_registered` is written when a finished request joins the donor pool;
`donor_registration_skipped` with a `reason` is written when it does not.

| `reason` | Meaning |
| --- | --- |
| `incompatible_engine` | the startup gate declined this engine, so its capture layout is unaddressable |
| `register_donors_disabled` | `register_donors` is off |
| `no_provider` | no provider is loaded |
| `short_prompt` | the prompt is below `min_prompt_tokens` |
| `no_captured_kv` | this request's KV capture was skipped, so it holds nothing to supply |
| `capture_not_durable` | this connector opened a capture for the request and the donor's metadata record does not exist yet, so its bytes are still queued or never landed |

`no_captured_kv` also carries `capture_skip_reason`, the reason from the
request's own `capture_skipped` event. A donor with no captured KV can never be
served from, but it would still be embedded, indexed and ranked: a request
served by this connector skips its own capture, so a prompt re-issued verbatim
produces a family of donors identical to each other that take every slot in the
provider's candidate cut and push the one donor that does hold KV out of it.
Providers must therefore drop donors whose `DonorRegistration.has_captured_kv`
is false *before* the top-k cut, not after.

### What a lookup embedded

A provider that matches on prompt text embeds that text with a sentence
encoder, and a sentence encoder truncates: a MiniLM model sees roughly the
first 256 tokens of what it is handed, and SemBlend caps the text by characters
(`SEMBLEND_EMBED_MAX_CHARS`, 4000 by default) before that. So only the *head*
of the text decides the match, and if the head is a wrapper every request
behind that wrapper embeds alike.

The connector therefore hands the provider the prompt text from the request's
block-aligned exact-prefix boundary onward rather than from token 0. The
boundary is where vLLM's own prefix cache stops, which is where the tokens this
connector is trying to serve begin and where the request stops looking like
every other request behind the same wrapper. Donors are registered the same way
-- from the boundary they were admitted at -- so both sides of a match embed the
content and not the wrapper. `boundary_sliced_query_text=false` restores
embedding from token 0 for an A/B.

The character offset comes from decoding the *prefix* tokens and taking their
length, not from an offset mapping over the whole prompt: the prefix is the
short part and the prompt is not, and the token ids already name the position.
The decode drops special tokens and strips, so the offset can be a few
characters off; that moves where the embedded window starts by at most a word.

`semantic_lookup_hit` and `semantic_lookup_miss` carry what was embedded:

| Field | Meaning |
| --- | --- |
| `query_text_offset_tokens` | the token position the embedded text starts at; 0 means the whole prompt |
| `query_text_chars` | the length in characters of the text handed to the provider |
| `query_text_fallback` | `null` when the text is the slice, otherwise why it is not |

`donor_registered` carries the same three as `donor_text_offset_tokens`,
`donor_text_chars` and `donor_text_fallback`.

| `*_text_fallback` | Meaning |
| --- | --- |
| `no_prompt_text` | no text was available at all (`enable_prompt_text` off, or no decode) |
| `disabled_by_config` | `boundary_sliced_query_text` is off |
| `offset_unavailable` | the prefix tokens could not be decoded, so there is no character offset |
| `offset_past_prompt_text` | the decoded prefix is longer than the prompt text itself |
| `below_min_chars` | the slice is shorter than `min_query_text_chars` and identifies nothing |

Every fallback sends the full prompt text, which is what earlier releases
always sent. A run whose misses all carry `query_text_offset_tokens=0` on
wrapped prompts is the phase-0 failure reproducing, not a provider problem.

### Capture cost

Donor capture is not free, and the part of it that is still on the critical
path is now the copy alone. `save_kv_layer` (connector.py:3035) is called by
vLLM from inside the forward pass, once per layer per prefill chunk, and
`_capture_layer` (connector.py:3073) does two things on that thread:

1. gathers the window's slots out of the paged KV cache
   (`_extract_kv_from_layer`, connector.py:1154);
2. copies it to host memory with a blocking `.cpu()` on pageable memory
   (connector.py:3104) -- no pinned staging buffer, no side stream, no
   `non_blocking=True`, so the copy synchronizes the device.

Everything after that is handed to the connector's capture writer
(`_submit_captured_layer`, connector.py:3261) and runs on a single background
thread: the safetensors write, the donor's metadata record, and the directory
work behind them. `wait_for_save` (connector.py:3539) is the join it is
documented to be.

Under chunked prefill a resumed capture still reads the previous chunk back
and concatenates it (connector.py:3094), so a donor captured in *n* chunks
copies its early tokens *n* times. The read is served from the writer's
staging map when those bytes are still in flight, so a chunk never appends to
a half-written file.

The copy is the whole prompt's KV. For Qwen2.5-7B-Instruct at fp16 -- 28
layers, 4 KV heads, head_dim 128 -- that is 2 x 28 x 4 x 128 x 2 = 57,344 bytes
per token, so ~1.2 GB at a 21.3K-token prompt and ~1.8 GB at 32K.

**What this change is answering.** Measured phase-0, 2026-09-14, stock vLLM
0.29 on an A10G, `register_donors=true` with capture on every request, `disk`
tier on a local NVMe ext4 volume:

| Run | Prompt tokens (median) | Stock + prefix cache TTFT (median) | Connector arm TTFT (median) | Delta |
| --- | --- | --- | --- | --- |
| 32K stream, 232 requests | 21,283 | 6.20 s | 14.80 s | +8.60 s |
| 4K smoke, 24 requests | 3,706 | 0.83 s | 1.00 s | +0.17 s |

and per captured request, from `donor_capture_cost` in the same run (medians):

| Field | Median |
| --- | --- |
| `capture_ms` | 14,535 |
| `copy_ms` | 119 |
| `write_ms` | 6,726 |
| `copy_bytes` | 1,254 MB |

The copy is 119 ms of a 14.5 s capture. The write is 6.7 s of it, and the
remainder is the per-layer `metadata.json` rewrite and the gather. So:

- **overlapped with the rest of the forward pass now**: the safetensors
  write, the metadata record and the directory work leave `save_kv_layer`, so
  `write_ms` on the captures that follow this change is the handoff to the
  writer and nothing else, unless the queue was full, in which case
  `write_blocked_ms` names the wait. **Whatever does not overlap is still
  paid inside `execute_model`**, at the `wait_for_save` join, and it is
  reported as `flush_ms`. vLLM calls both hooks inside the same step
  (`vllm/v1/worker/kv_connector_model_runner_mixin.py`: "this context manager
  ... encapsulates the entire KV connector lifecycle within execute_model",
  with `wait_for_save` in the `finally`), so the expected win is **pipelining,
  not elimination**: a step with nothing to overlap against pays the same
  wall time as before. A post-change phase-0 number has to be compared as
  `capture_ms + flush_ms` against the old `capture_ms`, never against
  `capture_ms` alone -- `write_ms` collapsing to ~0 is the work moving, not
  the work going away;
- **still fully on it**: the gather out of the paged cache and the blocking
  `.cpu()` (`copy_ms`, a measured 119 ms median), and the concatenation a
  chunked capture does. Taking those off needs a pinned host buffer on a side
  CUDA stream and real GPU validation; it is not in this change.

The delta above landed on every request whether or not it was served -- in
that run 3 of 231 lookups hit -- which is what `capture_policy` is for (below).
Lookup latency in the same audit was a median of 19 ms, so the lookup was
never the cost.

#### The ordering invariant

**Registration with the provider happens only after the writer has durably
written every layer and the metadata record for that donor.**

That is the hazard a deferred write creates, and it is enforced in two places:

- the writer's queue is FIFO with one consumer, and a donor's metadata record
  is submitted only after every layer job of the step has already landed
  (`_flush_captures`, connector.py:3407, joins the writer and drains its
  failures *before* queuing any record);
- the record is the donor's visibility. `_has_stored_donor` reads its
  presence and nothing else -- not the donor directory, which exists from the
  first layer written into it -- and `_register_donor_on_finish` holds back a
  request this connector captured until it exists.

**The record is looked for under the namespace the capture was opened under,
not under one recomputed at finish.** The two are not always the same digest:
the scheduler's capture hook is handed vLLM's `NewRequestData`, which has no
`cache_salt` field, while `request_finished` is handed a `Request`, which
does, and the namespace folds `cache_salt` in. A gate keyed on the finish-side
namespace stats a directory the worker never writes to, so on any salted
deployment every donor would be declined forever with its bytes on disk under
the other digest. The capture's own namespace is recorded when the capture
opens and the gate reads that.

**A donor that is not readable yet is not the same as one that never will
be.** vLLM's `Scheduler._free_request` calls `request_finished` and only then
adds the id to `finished_req_ids`, which ships in the *next* `SchedulerOutput`
-- so a request that finishes while its capture is still open has that capture
closed by the worker one step *after* the registration hook already ran. Such a
donor is held, not declined: it is retried for `donor_registration_retry_steps`
scheduler steps (default 8) at every scheduler-role hook that recurs -- the
per-step metadata build and each request's finish -- and once more at
`shutdown`, which is the last pass and therefore terminal. The budget is
counted in steps and nothing else: a step that finishes a whole batch of
requests runs this pass once per finishing request, and none of those calls
ages anybody, because the worker gets no chance to publish until the next
step. It leaves `donor_registration_deferred` when it is held, and registers
with `deferred_steps` naming how many steps it waited. Only a record that
never appears is a decline.

A write that fails ends in that decline rather than a silent one: the donor is
never published, the worker writes `donor_capture_write_failed` with
`capture_write_errors_total`, and the scheduler role's registration declines
with `capture_not_durable` once the retries are spent. Join the two on
`request_id` -- they are written by different connector instances in different
processes.

**A capture nobody published is deleted by the worker, not by the scheduler.**
The disk tier has no eviction of its own, so an unpublished capture's layer
files -- ~1.2 GB at a 21.3K-token prompt -- would otherwise sit on the volume
forever, unreferenced and unmatchable. The deletion belongs to the role that
owns those files: the worker, whose finish hook joins the capture writer
before it looks, so past that join a missing record means no record is coming.
The scheduler role only declines. It is another process; a capture it cannot
see a record for may still be in the worker's write queue, and deleting a
layer there leaves the writer publishing that donor's record over a hole. The
deletion leaves `capture_orphan_discarded`, and a publish checks its own
layers one last time before writing the record, so a record never names a
layer that is not on the store.

The metadata record is written **once per donor**, when the capture completes,
not once per layer per chunk as it was through 0.2.5. A capture the scheduler
has not closed is therefore not discoverable at all: a partially captured
donor can no longer be matched mid-prefill.

#### Which requests are captured

`capture_policy` decides, before anything is copied:

| Value | Captures |
| --- | --- |
| `all` (default) | every request that passes the length, served and block gates -- the behaviour every earlier release had, unchanged, because the phase-0 manifests predict donor selection from it |
| `sampled` | a deterministic fraction, `capture_sample_rate`, keyed on `sha256(request_id)` so a rerun of the same trace captures the same donors |
| `hinted` | only requests the caller marked |

A router sets the hint one of three ways, all read by `_capture_hinted`: on
the request's sampling params, `extra_args={"semblend_capture": true}` (vLLM's
OpenAI server forwards `vllm_xargs` into `extra_args`, so a client can send
`"vllm_xargs": {"semblend_capture": true}`); as request metadata or a header,
`x-semblend-capture: 1`; or as an attribute of that name on the request
object. The key is `capture_hint_key`, `semblend_capture` by default. A value
that is not a recognized truthy spelling reads as "not set".

The policy is deliberately *not* "capture only on a lookup hit": a seed never
hits, so that policy captures nothing worth having and the donor pool never
starts.

Requests the policy declines leave `capture_skipped` with `reason`
`not_sampled` or `not_hinted`, carrying `capture_policy` and
`capture_sample_rate`, and a `donor_registration_skipped` with
`reason="no_captured_kv"` at finish, exactly as any other skip does.

#### What the events carry

`donor_capture_cost` is written by the worker role once per captured request,
when the worker sees it finish:

| Field | Meaning |
| --- | --- |
| `capture_ms` | wall time inside the per-layer save hooks for this request, all layers and chunks |
| `copy_ms` | of that, the device-to-host copies |
| `write_ms` | of that, what the write cost the forward pass -- after this change the handoff to the writer, not the write |
| `write_blocked_ms` | of `write_ms`, time spent waiting on a full writer queue |
| `flush_ms` | the `wait_for_save` joins this request's writes were part of, still inside `execute_model`: the part of the write the forward pass did not overlap away. Per step, not per request, so two requests captured in the same step each report the whole join rather than a share of it |
| `copy_bytes` | bytes that crossed to host memory, counted before any concatenation with earlier chunks |
| `layers` | layers this request captured |
| `store_tier` | `kv_storage_backend` |

It is the worker's event, not a field on `donor_registered`, because the copy
is the worker's: `request_finished` is scheduler-role and that connector is a
different instance in a different process, which has never seen a tensor. Join
the two on `request_id`. `donor_registered` and `capture_skipped` carry
`store_tier` so the tier a cost was paid to -- or avoided -- is on both sides.

Two more worker events come out of the writer:

| Event | Written when | Fields |
| --- | --- | --- |
| `capture_write_queue_full` | the forward pass waited for room in the writer's queue; once per request, however many layers waited | `blocked_ms`, `queue_depth`, `store_tier` |
| `donor_capture_write_failed` | a queued write raised; the donor is never published | `kind`, `layer`, `error_type`, `store_tier` |

Two scheduler-role events come out of the durability gate:

| Event | Written when | Fields |
| --- | --- | --- |
| `donor_registration_deferred` | the donor's record had not appeared when its request finished; it is retried per step | `reason`, `prompt_tokens`, `retry_steps`, `store_tier` |
| `capture_orphan_discarded` | the worker deleted the files of a capture it never published, so they do not sit on a tier with no eviction | `files`, `store_tier` |

Every counter the capture path keeps, and what it means:

| Counter | Counts |
| --- | --- |
| `capture_ms_total` | `capture_ms` over all captured requests |
| `capture_copy_ms_total` | `copy_ms` over all captured requests |
| `capture_write_ms_total` | `write_ms` over all captured requests |
| `capture_write_blocked_ms_total` | `write_blocked_ms` over all captured requests |
| `capture_flush_ms_total` | wall time in the `wait_for_save` join, added once per join rather than once per request, so it is the unduplicated total |
| `capture_bytes_total` | bytes copied to host |
| `capture_layers_total` | per-layer save hooks that captured something |
| `capture_layer_writes_total` | layer payloads the writer landed |
| `capture_metadata_writes_total` | donor records the writer published -- one per donor |
| `capture_write_queue_blocked_total` | submissions that waited for queue room |
| `capture_write_errors_total` | queued writes that raised |
| `capture_metadata_skipped_evicted` | records not published because the memory tier evicted the donor while its record was queued behind its own layers; publishing would advertise tensors that are gone |
| `capture_finalize_skipped_write_failed` | records not queued because a layer of that donor failed to land |
| `capture_orphans_discarded_total` | unpublished captures whose files the worker deleted |
| `capture_orphan_discard_errors` | those deletions that failed |
| `capture_closed_short_at_decode` | captures closed at decode still short of the prompt's cacheable prefix; the donor is published at the shorter length |
| `capture_disabled_steps` | scheduler steps where materialization was off, so nothing was captured |
| `capture_unclamped_no_schedule_info` | capture ends taken as the whole prompt because the scheduler output carried no `num_scheduled_tokens`; stock vLLM always carries it, so this counts a stub engine |
| `capture_metadata_unexpected_type_layers` | save-hook calls that found another connector's metadata bound, so every capture the scheduler queued for that step was dropped |

and on the registration side, `donor_registration_deferred_total` and
`donor_registration_skipped_not_durable`.

**Still proposed, not implemented: defer the copy.** Copy into a pinned host
buffer on a side CUDA stream with `non_blocking=True`, record an event, and
let the writer thread wait on the event rather than the forward pass. The
donor's slots stay valid for the life of the request, so a late read is safe,
but the pinned-buffer pool, the per-request event bookkeeping and the
interaction with the memory tier's LRU eviction are all new failure surfaces.
It should land behind `capture_async=false` and be measured against the same
phase-0 stream before the default moves. At a measured 119 ms median it is
also the smallest of the three wins.

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

## Tenant key

Two different keys travel with every request, and they are not
interchangeable.

**The namespace** (`namespace_for_request`) is engine-local. It digests the
model, tokenizer, block size, dtype, `cache_salt` and adapter, so it moves with
values that are not request fields at all. It is what isolates a donor inside
this engine: a lookup sees donors registered under the identical namespace and
nothing else. Nothing outside this process can reproduce it, by design.

**The tenant key** is derived from the request's raw `cache_salt` alone:

```
tenant_key := "semblend:tenant:v1:" + sha256(cache_salt utf-8)[:32]
```

with the sentinel `semblend:tenant:v1:none` when the request carries no salt.
Because it is a function of the salt and nothing else, a gateway that sets one
salt per tenant can compute the same string, which is what makes tenant-scoped
placement possible; because it is a digest, the salt itself never reaches the
wire or anyone's logs.

The connector does not compute the tenant key. It reads the raw salt
(`cache_salt_for_request`) and carries it on the records it hands the provider
— `SemanticLookupRequest.cache_salt` and `DonorRegistration.cache_salt` — so a
provider never reaches back into vLLM's objects for it. The SemBlend provider
forwards it to `SemBlendPipeline.register_donor(cache_salt=...)`, and SemBlend
publishes it in the `DonorRegistered` event at
`namespace.extra.tenant_key`, beside the engine-local key at
`namespace.extra.cache_salt`.

The contract is SemBlend's, versioned **tenant key v1**, and is written down
once in `semblend-release/docs/tenant-key-v1.md`. The shared test vector is
`tests/tenant_key_v1_vector.json` — identical bytes in every repository that
implements it.

Notes:

- A SemBlend release from before the contract has no `cache_salt` argument.
  The provider checks the signature and omits it rather than raising, so the
  donor is registered without a tenant key rather than lost.
- A vLLM request type that has no `cache_salt` field at all is a version
  mismatch, not an unsalted deployment: every donor is then published with the
  no-tenant sentinel and no placement can select it.
  `cache_salt_for_request` warns once per process in that case (the request
  carrying the field with no salt set is silent, because that is a real
  answer).
- The salt is hashed by SemBlend exactly as the request carried it — no
  trimming, no case folding — so a caller that wants two requests to share one
  tenant key must set byte-identical salts.
- The raw salt is tenant-identifying and is never written to the audit trail
  or a log line; only values derived from it are.

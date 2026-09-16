# Changelog

All notable changes to this project will be documented here.

This project uses pre-1.0 semantic versioning. Breaking behavior may change
between minor releases while the vLLM semantic KV interface is experimental.

## Unreleased

Donor capture no longer writes on the engine's prefill thread. Measured in the
same phase-0 pass of 2026-09-14 on stock vLLM 0.29 with an A10G: per captured
request the medians were `capture_ms` 14,535, `copy_ms` 119, `write_ms` 6,726
and `copy_bytes` 1,254 MB, while engine TTFT went from 6.20 s to 14.80 s at
21.3K-token prompts and from 0.83 s to 1.00 s at 3.7K. The device-to-host copy
was never the cost -- 119 ms of a 14.5 s capture -- so what moves here is
everything behind it.

- The safetensors write, the donor's metadata record and the directory work
  behind them now run on one background writer owned by the connector, fed by
  a bounded FIFO queue (`capture_write_queue_depth`, 64 by default). The
  per-layer host copy stays on the forward thread, because it reads the paged
  KV cache that thread owns. `wait_for_save` is the join it was always
  documented to be, and it is also where a completed donor is published, so
  the record can never overtake the tensors it describes. When the queue is
  full the forward pass waits rather than dropping a layer -- a dropped layer
  would leave a donor advertising bytes nobody wrote -- and the wait is
  counted (`capture_write_queue_blocked_total`, `write_blocked_ms` on
  `donor_capture_cost`, one `capture_write_queue_full` row per request). The
  writer is stopped in the connector's `shutdown` and again at interpreter
  exit. Every wait in that stop -- handing the writer its stop signal
  included, because a wedged store leaves the queue full and an unbounded put
  there never returns -- is inside one deadline
  (`capture_write_close_timeout_s`, 5 s by default), split so that queueing
  the pending records cannot spend the drain's half of it; a store that has
  stopped responding costs that much at teardown and warns, rather than
  hanging the process. Whatever that deadline leaves in the queue is handed
  back as a failed write (`donor_capture_write_failed`,
  `capture_write_errors_total`) rather than left outstanding, so the
  `wait_for_save` of a step that ran after a teardown returns instead of
  joining work nobody will run. `write_ms` keeps its meaning -- what the write
  cost the forward pass -- and loses its old content: it is now the handoff.
- **The write is overlapped, not eliminated.** vLLM calls `wait_for_save`
  inside the same `execute_model` as the save hooks, so whatever the writer
  has not finished when the step ends is still paid on the forward thread. It
  is now measured: `flush_ms` on `donor_capture_cost` and
  `capture_flush_ms_total`. A step with nothing to overlap against pays the
  same wall time as before; the win is per-layer pipelining. Compare a
  post-change phase-0 capture as `capture_ms + flush_ms` against the old
  `capture_ms`, never against `capture_ms` alone.
- `metadata.json` is written once per donor, when the capture completes,
  rather than once per layer per prefill chunk. That record is now the donor's
  visibility: `_has_stored_donor` reads its presence and not the donor
  directory, which exists from the first layer written into it. A partially
  captured donor is therefore no longer matchable mid-prefill.
- **A donor is never discoverable before its bytes are readable.**
  Registration with the provider happens only after the writer has durably
  written every layer and the metadata record for that donor. The record is
  looked for under the namespace the *capture* was opened with, not one
  recomputed at finish: vLLM hands the capture hook a `NewRequestData`, which
  has no `cache_salt` field, and the finish hook a `Request`, which does, and
  the namespace digest folds the salt in -- so a gate keyed on the finish side
  would stat a directory the worker never writes to and decline every donor on
  any tenant-salted deployment. A write that failed leaves
  `donor_capture_write_failed` and `capture_write_errors_total` on the worker
  side and never publishes the donor at all. The two roles are different
  connector instances in different processes: join them on `request_id`.
- **A capture that is not readable yet is retried, not declined.** vLLM's
  `Scheduler._free_request` calls `request_finished` before the id reaches
  `finished_req_ids`, so a request that finishes while its capture is still
  open has that capture closed by the worker one step *after* the registration
  hook ran. Such a donor is held for `donor_registration_retry_steps`
  *scheduler steps* (8 by default) -- a budget aged on the step clock alone,
  so a batch of finishes inside one step cannot spend it before the worker has
  a step to publish in -- retried at every recurring scheduler-role hook and
  once more at `shutdown`, which is terminal; it leaves
  `donor_registration_deferred` and registers with `deferred_steps` naming the
  wait. Only a record that never appears is a decline --
  `donor_registration_skipped reason="capture_not_durable"` with
  `donor_registration_skipped_not_durable`.
- **A capture nobody published is deleted by the worker that owns it**
  (`capture_orphan_discarded`, `capture_orphans_discarded_total`): the disk
  tier has no eviction of its own, so layer files with no record would sit
  unreferenced and unmatchable for the life of the volume. The worker's finish
  hook joins its own writer before it looks, so past that join a missing
  record means no record is coming. The scheduler role never deletes: it is
  another process, a capture it sees no record for may still be in the
  worker's queue, and taking a layer there leaves the writer publishing that
  donor's record over a hole. Publishing re-checks the layers the record is
  about to vouch for as a last line, so a record never names a layer that is
  not on the store.
- `capture_policy` chooses which requests become donors: `all` (the default,
  and byte-for-byte the selection every earlier release made, because the
  phase-0 manifests predict donors from it), `sampled` at
  `capture_sample_rate` keyed on `sha256(request_id)` so a rerun captures the
  same set, or `hinted`, which captures only requests a router marked --
  through `extra_args={"semblend_capture": true}` on the sampling params
  (`vllm_xargs` from the OpenAI server), an `x-semblend-capture` header or
  metadata key, or an attribute of that name. The key is `capture_hint_key`.
  Declines are audited as `capture_skipped` with `not_sampled` / `not_hinted`.
  Capturing only on a lookup hit is deliberately not offered: a seed never
  hits, so that policy would never fill a donor pool.
- `shutdown` submits the pending metadata records before it stops the writer,
  so a donor whose layers all landed at teardown is discoverable rather than
  durable-and-invisible, and it keeps the closed writer on the connector: a
  `save_kv_layer` after shutdown is a named write failure instead of a
  silently restarted thread.

## 0.2.5 - 2026-09-14

The quickstart now says to run the semantic-span mode with prefix caching
enabled: the eviction of connector-filled blocks was measured live in
phase-0 E6 on vLLM 0.29 and removes the exact-cache contamination it was
written for.

Both entries come from the phase-0 LongBench-v2 pass of 2026-09-14 on stock
vLLM 0.29: 232 requests, prompts with a median of 21.3K tokens, each document
sent once as a seed and once again behind a 512-token operator preamble.

- A lookup now embeds the prompt text from the request's block-aligned
  exact-prefix boundary onward instead of from token 0, and a donor is
  registered with the text from its own boundary onward. In the measured run
  228 of 231 lookups missed although every recipient duplicated a document a
  donor already held. The provider embeds prompt text with a sentence encoder,
  which truncates to its model window -- roughly the first 256 tokens for
  MiniLM, and SemBlend caps the text by characters before that -- so with a
  512-token preamble in front of the document the entire embedded window was
  preamble. Every recipient embedded alike, and none of them embedded like the
  seed it duplicated. The boundary is where vLLM's own prefix cache stops,
  which is both where the tokens this connector wants to serve begin and where
  the request stops looking like every other request behind the same wrapper.
  The character offset is taken by decoding the prefix tokens and measuring
  them, not by re-tokenizing the whole prompt for an offset mapping.
  `semantic_lookup_hit` and `semantic_lookup_miss` gained
  `query_text_offset_tokens`, `query_text_chars` and `query_text_fallback`, and
  `donor_registered` the same three under `donor_text_*`, so a run says what it
  embedded rather than leaving it to be inferred. A slice that is empty,
  undecodable or shorter than `min_query_text_chars` (default 64) falls back to
  the full prompt text and names the reason in that field.
  `boundary_sliced_query_text=false` restores the previous behaviour for an A/B.

- Donor capture is now instrumented, because it was the arm's dominant cost and
  nothing in the audit said so. Median TTFT was 14.80 s on the connector arm
  against 6.20 s on stock vLLM with prefix caching -- +8.6 s on every request,
  served or not, where only 3 of 231 lookups hit -- while the same
  configuration at 3.7K-token prompts cost +0.17 s. That scaling is a per-token
  KV copy, not a lookup: lookup latency in the same audit had a median of 19 ms,
  and at 21.3K tokens the capture copies ~1.2 GB of KV out of the GPU inside
  the forward pass. The worker now writes one `donor_capture_cost` event per
  captured request carrying `capture_ms`, `copy_ms`, `write_ms`, `copy_bytes`,
  `layers` and `store_tier`, with `capture_ms_total`, `capture_copy_ms_total`,
  `capture_write_ms_total`, `capture_bytes_total` and `capture_layers_total` as
  the connector totals; `donor_registered` and `capture_skipped` gained
  `store_tier`. The event is the worker's and not a field on `donor_registered`
  because the copy is the worker's, and the scheduler-role connector that
  writes that event is a different instance in a different process; join them
  on `request_id`.
  Behaviour is unchanged: the copy and the write are still synchronous on the
  forward pass. docs/VLLM_CONNECTOR_CONTRACT.md gained a "Capture cost" section
  with the measured numbers, the line-level read of the path, and the staged
  proposal for taking it off the critical path -- capture fewer requests first,
  then defer the write, then the copy.

## 0.2.4 - 2026-09-14

Both entries are fixes for behaviour measured live in phase-0 E6
(2026-09-14) on stock vLLM 0.29, where a prompt served from a donor was
re-issued verbatim five times.

- A request whose KV capture was skipped is no longer registered as a donor.
  It holds nothing any recipient could be served from, but it was still
  embedded, indexed and ranked against every later lookup — and a request
  this connector serves skips its own capture, so a verbatim re-issue
  produced a donor identical to the previous re-issue. In the measured run
  the fifth repeat found the recipient and the four repeats before it sitting
  at similarity 1.0 in a `lookup_top_k` of five, and the seed they were all
  served from no longer made the cut: the reuse stopped, and the audit showed
  a plain miss with no cause in it. The skip is now audited as
  `donor_registration_skipped` with `reason=no_captured_kv` and the
  `capture_skip_reason` it inherits, and counted as
  `donor_registration_skipped_no_captured_kv`. `register_donors` and
  `capture_served_requests` are unchanged: with capture on, a served request
  is captured and stays a donor.
  Defensively, the ranking this connector owns now drops donors that hold no
  captured KV *before* the top-k cut rather than after it, so such a donor
  cannot take a slot in the cut from one that can supply KV.
  `DonorRegistration` gained a `has_captured_kv` field (default `True`, every
  existing construction site unchanged) to carry that fact to a provider.

- The connector no longer runs a lookup that cannot end in a load. In
  semantic-span mode, when fewer tokens are left after the block-aligned
  boundary than `min_semantic_span`, no span can clear the floor
  (`block_align_spans` drops it, and so does the floor re-applied after the
  clamps), so the lookup is skipped with
  `lookup_skipped_remaining_below_min_span` — the fields of its sibling skips
  plus `remaining` and `min_semantic_span` — and counted as
  `skipped_remaining_below_min_span_per_attempt`. In the measured run the
  re-issues hit vLLM's own exact prefix cache for 3408 of 3411 tokens; with
  `skip_when_exact_prefix_ratio_at_least` at 1.0 that ratio kept them
  eligible, and each repeat paid an embedding plus a ~26 ms provider round
  trip for three tokens before ending in `semantic_span_boundary_missed`. The
  gate is scoped to semantic-span mode, where `min_semantic_span` is the only
  place that floor applies.

## 0.2.3 - 2026-09-14

Runs on stock vLLM 0.29 with prefix caching enabled: the eviction of
connector-filled blocks from the engine's exact prefix cache was measured
live in phase-0 E5 (2026-09-14), where a verbatim re-issue of a served
prompt was re-served through the connector from the clean donor rather
than from the poisoned exact cache. The quickstart's prefix-caching
guidance flips once E6 has measured the contamination arm.

- The request's raw `cache_salt` now reaches SemBlend, which publishes it as a
  **tenant key** (`semblend:tenant:v1:<sha256(salt)[:32]>`, or the sentinel
  `semblend:tenant:v1:none`) in the `DonorRegistered` event at
  `namespace.extra.tenant_key`. The connector's per-request namespace, still
  sent as the engine-local isolation key at `namespace.extra.cache_salt`,
  digests the model, tokenizer, block size, dtype, salt and adapter, so no
  router holding only the request could ever reproduce it; the tenant key is a
  function of the salt alone, so whoever set the salt can compute it and scope
  placement to one tenant. `SemanticLookupRequest` and `DonorRegistration`
  gained a `cache_salt` field (default `None`, every existing construction site
  unchanged), so a provider reads the salt off the record instead of reaching
  back into vLLM's objects. The raw salt is never logged or audited. A request
  type that carries no `cache_salt` field at all — a version mismatch rather
  than an unsalted deployment — warns once per process, because it would
  otherwise publish every donor tenant-less and silently. Contract:
  `docs/VLLM_CONNECTOR_CONTRACT.md`, "Tenant key"; shared vector:
  `tests/tenant_key_v1_vector.json`.

- Blocks the connector fills are evicted from vLLM's exact prefix cache on
  the step they are filled, and the same pass runs again on every later step
  the request is scheduled, so the blocks vLLM hashes as the request
  continues -- the rest of the prompt under chunked prefill, then every
  decode block -- go the same way. Approximate KV can no longer be served to
  a later request through the engine's own exact match. Evictions are
  counted and audited per load.
  This is what allows the semantic-span mode to run with prefix caching
  enabled; the quickstart still says to disable it until the change has
  been measured on a GPU.
- `min_boundary_tokens` is enforced ahead of the provider lookup, in every
  mode: set it to the block size so unserved requests prime the shared
  prefix cleanly.
- `evict_filled_blocks_from_prefix_cache` (default `true`) turns that
  eviction off for a contaminated control run. With it off nothing is
  evicted -- approximate KV stays servable to a later exact match -- but the
  connector still tracks the blocks it filled and writes a
  `prefix_cache_blocks_left_cached` audit event whenever a pass finds
  filled blocks cached, with the count and the request's join key, so what
  the eviction removes can be measured rather than asserted. Measurement
  only; the default behaviour is unchanged.
- Both prefix-cache audit events also carry a distinct-block count and a
  request-scoped distinct total (`prefix_cache_distinct_blocks_evicted` /
  `prefix_cache_distinct_blocks_left_cached`), and that is the pair to
  compare across arms: it counts distinct physical blocks per request and
  survives a readmission, including one the scheduler drops before the span
  is served. The per-pass counts beside them are not comparable. Both arms
  report a block once while the request keeps the same block table; a
  preemption replaces that table, and the blocks are then re-found on the
  contaminated arm whether or not the engine re-hashed them, while the
  eviction arm meets again only the ones it did. See the audit contract.

## 0.2.2 - 2026-09-13

Correctness release for the semantic-span path on stock vLLM 0.29. Every
item below was found by reading the engine source or by tests that fail
against 0.2.1; no behavior was measured on a GPU in this release.

- Span loads at a non-zero boundary wrote donor KV over the request's
  shared prefix blocks and left the credited window uninitialized. The
  write now lands at the target offset on both the semantic-span and the
  exact-prefix paths, and the donor offset travels with it.
- Three guards never fired: the compressed-attention check tested a type
  the engine never passes, the rope check skipped its own bail-out on
  scaled-rope models, and nothing refused backends that pack attention
  heads differently. Each wrote silently wrong KV; each now declines.
- An advertised span that reached the prompt end drove the engine's
  remaining work to zero and tripped an engine assertion. Advertised
  counts are now capped so at least one token is always left to compute.
- The pending load is built once allocation has succeeded, not inside the
  match hook the engine documents as side-effect free.
- Counters are per request unless their name says per attempt; a request
  re-queued under memory pressure was inflating every reported rate.
- Donor storage: eviction retracts the donor's advertised length and its
  files, both read paths count as a use, and a donor missing at load time
  is a counted decline. Non-span kinds hand their blocks back through the
  engine's load-error hook so vLLM recomputes them under the recompute
  failure policy; the span kind still fails loudly.
- Donor capture is clamped to what the engine actually scheduled and
  continues across later prefill chunks, so a donor's recorded length is
  what was written rather than the whole prompt.
- `min_prompt_tokens` defaults to 512 to match `min_semantic_span`, with a
  warning when they disagree in span mode. `min_boundary_tokens` is added
  for a later release.
- A compatibility suite that drives all four connector hooks against real
  tensors, and loads vLLM's own type module when a source tree is
  available (`SEMBLEND_VLLM_SOURCE`), skipping loudly otherwise.
- Publishing moves to PyPI trusted publishing; no API token secret.

Known: the semantic-span mode still requires prefix caching off, which
limits it to spans that begin at the prompt start. Lifting that is the
next release.

## 0.2.1 - 2026-09-03

- Recipients that received a semantic load are no longer captured as
  donors by default (`capture_served_requests` restores the old behavior).
  Capture was ~230 ms of the hit path at 3.5K tokens.
- `kv_storage_backend=memory` keeps donor layers in the worker's host RAM
  under an LRU cap (`kv_memory_max_donors`), so loads never touch disk.
- Loads are materialized on vLLM's no-forward scheduling steps instead of
  raising; under concurrent long prefills that raise took the engine down.

## 0.2.0 - 2026-09-03

- `semantic_span_experimental` mode: block-aligned donor spans advertised at
  the scheduler's computed boundary and realized with K re-rotation into the
  recipient's blocks. Verified paraphrase whole-span serve works on stock vLLM
  0.26; interior spans use the scheduler re-consult patch.
- vLLM 0.26 worker-registered KV caches consumed via `register_kv_caches`;
  a semantic-span load that materializes zero layers fails loudly.
- Spans trim to the captured donor window (chunked prefill captures the
  donor's first scheduled chunk).
- Rope parameters resolved from transformers 5.x `rope_parameters`;
  non-default rope types decline the load; rotation tables built on the
  donor K device.
- Extra-config keys derived from the config dataclass so the getter path can
  never silently drop a key.
- `confidence_tier` surfaced in the lookup-hit audit event.
- Stock-vLLM quickstart (`docs/QUICKSTART_VLLM.md`).
- Initial discovery-only vLLM out-of-tree connector scaffold.
- Local deterministic provider for unit and integration testing.
- Lazy SemBlend provider adapter.
- SemBlend provider and validation docs.

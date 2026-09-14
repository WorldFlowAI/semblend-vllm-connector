# Changelog

All notable changes to this project will be documented here.

This project uses pre-1.0 semantic versioning. Breaking behavior may change
between minor releases while the vLLM semantic KV interface is experimental.

## Unreleased

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

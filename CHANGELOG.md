# Changelog

All notable changes to this project will be documented here.

This project uses pre-1.0 semantic versioning. Breaking behavior may change
between minor releases while the vLLM semantic KV interface is experimental.

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

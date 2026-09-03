# Changelog

All notable changes to this project will be documented here.

This project uses pre-1.0 semantic versioning. Breaking behavior may change
between minor releases while the vLLM semantic KV interface is experimental.

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

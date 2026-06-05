# Architecture

## Goal

Provide a production-shaped vLLM connector that lets SemBlend
experiment with semantic KV donor discovery without weakening vLLM exact
prefix-cache semantics.

## Component Boundary

```text
vLLM
  exact prefix cache
  scheduler
  paged KV blocks
  KVConnector V1 lifecycle

SemBlend vLLM Connector
  request extraction
  namespace construction
  provider lookup lifecycle
  donor registration lifecycle
  materialization gating
  connector metrics

SemBlend
  in-process semantic donor discovery
  embedding/search/alignment/quality signals
```

## Safety Model

The connector treats semantic search as evidence, not authority.

`get_num_new_matched_tokens()` is a materialization promise in vLLM. A provider
may discover a semantic donor, but the connector returns positive matched tokens
only when it can load the advertised KV into vLLM-owned blocks. Until request-only
cache commit and segmented materialization are available, default behavior is
discovery-only.

## Rollout Stages

1. **Discovery-only OOT connector**
   - register donors after request finish;
   - lookup donors on future requests;
   - return zero external tokens;
   - emit hit/miss/reject telemetry.

2. **Exact-equivalent materialization**
   - return positive external tokens only when provider returns engine-valid block
     refs for an exact-equivalent prefix-shaped span;
   - compare outputs against cold vLLM;
   - fail closed through `kv_load_failure_policy="recompute"`.

3. **Request-local experimental materialization**
   - store block-aligned donor KV through vLLM's connector callbacks;
   - let SemBlend choose a donor, then load a prefix-shaped span into recipient
     blocks;
   - use only in isolated validation runs with vLLM prefix caching disabled.

4. **vLLM cache-commit policy**
   - upstream a generic `EXACT_COMMIT` vs `REQUEST_ONLY` external KV policy;
   - prevent non-identical semantic donor KV from entering exact prefix cache.

5. **Segmented external-KV plan**
   - target spans;
   - donor spans;
   - compute-only gaps;
   - halo recompute ranges;
   - cache publication mask.

5. **Sparse/partial attention path**
   - backend allowlist;
   - RoPE/position handling;
   - selective recomputation;
   - quality gates and materialization feedback.

## Repository Scope

This repo is intentionally limited to the vLLM connector surface.

It includes:

- connector lifecycle;
- provider protocol;
- safe modes;
- SemBlend provider adapter;
- generic validation harnesses.

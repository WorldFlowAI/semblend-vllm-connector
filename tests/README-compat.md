# Which compatibility guards actually ran

The connector falls back to an in-repo shim for `KVConnectorBase_V1` when vLLM
is absent (`semblend_vllm_connector._vllm_compat`), so a suite run on a machine
without vLLM checks the startup gate against this repo's own stand-ins — the
version drift the gate exists to catch would sail straight through. Anything
that claims vLLM compatibility has to say which of the two it did.

**Look at the pytest header.** Every run prints one of:

```
real-vLLM type guards: ACTIVE via installed vllm 0.29.0 [vllm.v1.kv_cache_interface]
```

```
real-vLLM type guards: SKIPPED (vLLM is not installed and no usable source tree
was named by $SEMBLEND_VLLM_SOURCE)
  the compatibility guards run against this repo's shim only; see tests/README-compat.md
```

`tests/conftest.py` decides this once per session and skips the real-type tests
in `tests/test_e2_compat_guards.py` (the `test_real_*` ones) when it has to.

## Running them

Highest fidelity, and what CI should do — a Linux box with the wheel installed:

```bash
pip install 'vllm==0.29.0'
pytest tests/
```

Where the wheel cannot be installed, point the suite at a read-only source
checkout of the same release:

```bash
SEMBLEND_VLLM_SOURCE=/path/to/vllm-0.29.0 pytest tests/
```

Keep that checkout somewhere durable. On 2026-09-16 the armed suite had been
pointed at a path under the system temp directory; the tree was cleaned out
from under it, and because the loader falls back rather than failing, a run
against a different vLLM release reported three failures that looked like
regressions and were not. A shallow checkout is enough and costs about 160 MB:

```bash
git clone --depth 1 --filter=blob:none --branch v0.29.0 \
  https://github.com/vllm-project/vllm.git ~/dev/worldflowai/vllm-0290
```

Set `SEMBLEND_REQUIRE_VLLM_GUARDS=1` alongside it so a source tree that cannot
be loaded fails the run instead of quietly skipping the guards.

A job whose point is the compatibility check should not depend on a human
reading that header. Set `SEMBLEND_REQUIRE_VLLM_GUARDS=1` and the run fails
outright rather than skipping:

```bash
SEMBLEND_REQUIRE_VLLM_GUARDS=1 pytest tests/
```

`.github/workflows/ci.yml` installs `.[dev]`, which does not include vLLM, so
CI today runs the SKIPPED path on every push. Adding a job that installs the
pinned vLLM and sets that variable is what turns these guards into a gate.

## What the source-tree route can and cannot reach

`vllm/__init__.py` imports the compiled extension and the engine, so the
conftest registers a package whose `__path__` is the checkout and imports the
submodules underneath it directly. The classes that come back are vLLM's own.

Reachable that way on a machine with an older torch than the release pins
(vLLM 0.29 pins `torch==2.13.0`):

- `vllm.v1.kv_cache_interface` — `FullAttentionSpec`, `MLAAttentionSpec`,
  `SlidingWindowSpec`, `KVCacheGroupSpec`, `KVCacheConfig`,
  `is_full_attention_spec`, `num_head_slots`. This is what the startup gate
  reads, and what the `test_real_*` guards are built from.

Not reachable, and why:

- `vllm.v1.core.kv_cache_utils` (`resolve_kv_cache_block_sizes`) — imports
  `vllm.config`, which needs `torch._inductor.custom_graph_pass` and `cbor2`.
  The prefix-hash block-size leg of the gate therefore records
  `compat_check_incomplete_block_size_resolution_unavailable` instead of
  running. `test_resolve_kv_cache_block_sizes_keeps_the_signature_the_gate_calls`
  checks that helper's name and signature against vLLM's source text rather
  than by calling it, so a rename still fails the suite.
- `vllm.distributed.kv_transfer.kv_connector.v1.base` (`KVConnectorBase_V1`) —
  needs `torch.distributed._symmetric_memory`. The connector keeps using its
  shim base class, so the suite exercises the connector's own logic, not vLLM's
  dispatch into it.
- `vllm.transformers_utils.config` (`is_rope_parameters_nested`) — vLLM 0.29
  requires transformers v5 and refuses to import under v4. The connector
  already treats that as "decide the rope shape locally"; the local answer is
  what `tests/test_e2_compat_guards.py` covers.

One local compromise, reported in the header when it applies: vLLM's
`vllm.utils.torch_utils` imports `torch.library.infer_schema` at module scope,
which older torch releases do not define. The conftest fills that one name in
with a function that **raises** if anything calls it, so the import proceeds and
any code that genuinely needs the real symbol fails loudly instead of running
against a stand-in. Nothing in the KV-cache spec dataclasses calls it.

Only the wheel exercises the whole set. A source-tree run is a real check of
the types the gate reads, not a substitute for installing vLLM.

# SemBlend vLLM Connector

[![CI](https://github.com/worldflowai/semblend-vllm-connector/actions/workflows/ci.yml/badge.svg)](https://github.com/worldflowai/semblend-vllm-connector/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

vLLM KVConnector for SemBlend-backed semantic KV donor discovery.

This repo is the open-source adapter layer between vLLM and
[SemBlend](https://github.com/WorldFlowAI/semblend).

SemBlend is a semantic KV reuse research library. It exists to evaluate when
similar prompts may safely reuse or blend previously computed KV state. This
connector exposes that work through vLLM's `KVConnectorBase_V1`
lifecycle.

## Status

Experimental.

Default behavior is **discovery-only**:

- exact vLLM prefix caching remains authoritative;
- semantic lookup runs only after exact prefix coverage is insufficient;
- the connector records donor hits, misses, and rejection reasons;
- it returns `(0, False)` from `get_num_new_matched_tokens()` unless a future
  materialization mode can prove that the KV can be loaded safely;
- normal vLLM execution continues on every provider error or unsupported case.


## Install

Development:

```bash
pip install -e ".[dev]"
```

With SemBlend provider support:

```bash
pip install -e ".[semblend,dev]"
```

Run local checks:

```bash
make check
```

## vLLM Configuration

Discovery-only mode:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "SemBlendVllmConnector",
    "kv_connector_module_path": "semblend_vllm_connector.connector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "mode": "discovery_only",
      "provider": "local",
      "min_prompt_tokens": 256,
      "min_similarity": 0.70
    }
  }'
```

SemBlend provider mode:

```json
{
  "kv_connector": "SemBlendVllmConnector",
  "kv_connector_module_path": "semblend_vllm_connector.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "mode": "discovery_only",
    "provider": "semblend",
    "min_prompt_tokens": 256,
    "min_similarity": 0.70,
    "min_reuse_ratio": 0.50,
    "embedder_type": "minilm",
    "model_id": "meta-llama/Llama-3.1-8B-Instruct"
  }
}
```

Equivalent JSON examples live in [`examples/`](examples/).

## Modes

| Mode | Positive matched tokens? | Purpose |
| --- | --- | --- |
| `discovery_only` | No | Safe telemetry and workload qualification. |
| `exact_prefix` | Only with engine-valid exact block refs | Future safe materialization path. |
| `request_only_experimental` | Yes, block-aligned prefix only | Isolated validation mode; run with vLLM prefix caching disabled. |
| `segmented_experimental` | Not enabled in this repo yet | Requires segmented/sparse execution support. |

## Safety Rules

The connector must not:

- weaken exact prefix-cache semantics;
- report semantic hits as computed tokens unless KV can actually be loaded;
- publish non-identical semantic donor KV into vLLM's exact prefix cache;
- cross model, tokenizer, adapter, or cache-salt namespaces;
- fail inference because semantic lookup failed.

## Repository Layout

```text
src/semblend_vllm_connector/
  connector.py        vLLM KVConnectorBase_V1 implementation
  config.py           config/env parsing
  provider.py         provider protocol + local deterministic provider
  providers/
    semblend.py       lazy SemBlendPipeline adapter
  types.py            shared dataclasses/enums
  namespace.py        vLLM request namespace extraction

docs/
  ARCHITECTURE.md     detailed architecture and rollout plan
  SEMBLEND_PROVIDER.md
  VLLM_CONNECTOR_CONTRACT.md
  VALIDATION.md       validation matrix

examples/
  discovery_kv_transfer_config.json
  semblend_discovery_kv_transfer_config.json
```

## Open Source Posture

This project follows the dynamic connector pattern used by mature vLLM KV cache
projects: vLLM loads the connector from a Python module path, connector-specific
settings live in `kv_connector_extra_config`, and unsafe materialization cases
fail closed to normal vLLM prefill.

See:

- [SEMBLEND_PROVIDER.md](docs/SEMBLEND_PROVIDER.md)
- [VLLM_CONNECTOR_CONTRACT.md](docs/VLLM_CONNECTOR_CONTRACT.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)

# SemBlend Provider

[SemBlend](https://github.com/WorldFlowAI/semblend) is a semantic KV reuse
research library. It exists to evaluate when similar prompts may safely reuse or
blend previously computed KV state while preserving explicit quality and safety
gates.

`semblend-vllm-connector` is the vLLM adapter for that work. It maps vLLM
request lifecycle events into a small provider interface:

- register donor metadata after a request finishes;
- look up possible donors for later requests;
- return discovery telemetry in safe modes;
- advertise materialized KV only when vLLM can load it safely.

## Current Behavior

The default provider mode is discovery-only. In this mode the connector may find
semantic donors, but it returns zero external KV tokens to vLLM. This allows
users to validate workload fit without changing model outputs.

## Optional SemBlend Dependency

Install with SemBlend support:

```bash
pip install "semblend-vllm-connector[semblend]"
```

Configure the provider:

```json
{
  "kv_connector": "SemBlendVllmConnector",
  "kv_connector_module_path": "semblend_vllm_connector.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "mode": "discovery_only",
    "provider": "semblend",
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "embedder_type": "minilm"
  }
}
```

## Provider Boundary

SemBlend can propose donor candidates. vLLM remains responsible for request
scheduling, KV allocation, block ownership, and exact prefix-cache semantics.
The connector must reject any donor that cannot be represented safely in
vLLM-owned KV state.

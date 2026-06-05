# Examples

These examples are intentionally discovery-only.

## Local Provider

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config "$(cat examples/discovery_kv_transfer_config.json)"
```

## SemBlend Provider

```bash
pip install -e ".[semblend]"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config "$(cat examples/semblend_discovery_kv_transfer_config.json)"
```

Both examples return zero external KV tokens today. They are for donor discovery,
workload qualification, and integration validation.

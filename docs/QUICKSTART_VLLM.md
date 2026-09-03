# Quickstart: semantic KV reuse on stock vLLM

This runs the SemBlend semantic KV connector against an unmodified vLLM
0.26 server on one GPU, seeds a donor request, sends a paraphrase of it,
and shows the reuse in the audit log. No fork, no external KV store, no
gateway: the connector and SemBlend run inside the vLLM process next to
the engine.

What you get on stock vLLM: **verified paraphrase reuse**. A request
whose content is a fact-equivalent rewording of an earlier request
(different tokens, same facts) is served the earlier request's KV for
the whole matched span at zero prefill recompute, gated by a fail-closed
fact check. Interior-span reuse (same document under a different
wrapper) additionally needs a 24-line scheduler patch that lets vLLM
re-consult the connector at chunk boundaries; see "Beyond stock vLLM".

## 1. Install

```bash
pip install "vllm==0.26.0"
pip install "semblend>=0.3.18" "semblend-vllm-connector>=0.2.0" sentence-transformers rapidfuzz
# Until those versions are on PyPI, install both straight from GitHub:
pip install "git+https://github.com/WorldFlowAI/semblend@feat/events-plus-paraphrase" \
            "git+https://github.com/WorldFlowAI/semblend-vllm-connector" \
            sentence-transformers rapidfuzz
```

`semblend` provides the matching pipeline (MiniLM embeddings + token
alignment + paraphrase verification); the connector implements vLLM's
`KVConnectorBase_V1`.

## 2. Serve

```bash
cat > kvcfg.json << 'JSON'
{
  "kv_connector": "SemBlendVllmConnector",
  "kv_connector_module_path": "semblend_vllm_connector.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "mode": "semantic_span_experimental",
    "provider": "semblend",
    "min_prompt_tokens": 256,
    "min_similarity": 0.7,
    "min_semantic_span": 512,
    "embedder_type": "minilm",
    "enable_prompt_text": true,
    "register_donors": true,
    "log_decisions": true,
    "audit_path": "/tmp/semblend-audit.jsonl",
    "kv_storage_path": "/tmp/semblend-kv"
  }
}
JSON

SEMBLEND_PARAPHRASE_SERVE=1 SEMBLEND_CHUNK_FAST_PATH=0 \
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --kv-transfer-config "$(cat kvcfg.json | tr -d '\n')"
```

`SEMBLEND_PARAPHRASE_SERVE=1` enables the verified paraphrase tier.
`SEMBLEND_CHUNK_FAST_PATH=0` disables the token-overlap fast path, which
otherwise preempts the paraphrase tier on low-overlap paraphrases (fixed
in a coming semblend release).

## 3. Seed a donor, then send a paraphrase

```bash
python - << 'PY'
import json, time, urllib.request

def complete(prompt, max_tokens=32):
    body = json.dumps({"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.loads(urllib.request.urlopen(req).read())
    return r["choices"][0]["text"], time.time() - t0

# Long, fact-bearing report; the paraphrase keeps every fact and rewords every sentence.
donor = "you are a triage assistant. review the following service report.\n" + " ".join(
    f"service checkout-77aa recorded {200 + i % 40} requests in window {i} while replica set alpha kept latency at {90 + i % 30} ms."
    for i in range(110)) + "\nsummarize the health of checkout-77aa."
warm = "you are a triage assistant. review the following service report.\n" + " ".join(
    f"during window {i} the checkout-77aa service held latency to {90 + i % 30} ms across replica set alpha as it logged {200 + i % 40} requests."
    for i in range(110)) + "\nsummarize the health of checkout-77aa."

text, t = complete(donor); print(f"donor  {t:.2f}s {text[:60]!r}")
time.sleep(5)  # donor registers on completion
text, t = complete(warm);  print(f"warm   {t:.2f}s {text[:60]!r}")
PY
```

## 4. Read the audit trail

```bash
grep -E "semantic_lookup_hit|semantic_span_load_advertised|runtime_materialized" /tmp/semblend-audit.jsonl
```

You should see, for the warm request: a `semantic_lookup_hit` with
`"confidence_tier": "paraphrase_verified"`, a
`semantic_span_load_advertised` at `"boundary": 0`, and a
`runtime_materialized` with `layers_materialized` equal to the model's
layer count. **Only `runtime_materialized` counts as reuse.** A hit that
is advertised but not materialized is not a win, and the audit keeps
them separate on purpose.

Configuration knobs that matter for an overhead study:

| Knob | Effect |
|---|---|
| `min_prompt_tokens` | requests below this never touch the layer (no lookup, no capture) |
| `register_donors: false` | lookup-only mode: no donor capture cost, reuse only from donors registered elsewhere |
| `min_semantic_span` | smallest span worth materializing |
| `max_materialized_tokens` | cap on served tokens per request |

## 5. Measure with SemBench

[SemBench](https://github.com/WorldFlowAI/sembench)
replays donor/recipient manifests against any OpenAI-compatible endpoint
with streamed TTFT and attribution from this audit log. The
`examples/overhead-benefit/` directory holds the exact manifests and
scripts behind our published overhead-vs-benefit table:

```bash
pip install "git+https://github.com/WorldFlowAI/sembench" transformers aiohttp
python -m sembench run-live-gateway --manifest hit.jsonl --output hit-run.jsonl \
  --gateway-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct
python -m sembench roi --histogram "512:0.6,2048:0.3,8192:0.1" --hit-rate-by-bucket "512:0,2048:0.1,8192:0.35"
```

The `roi` model charges lookup and capture to every eligible request and
reports the hit rate at which the layer breaks even for your traffic
mix; replace its default coefficients with the numbers from your run.

## Beyond stock vLLM

- **Interior spans** (the same long document under a different
  question or system prompt): needs vLLM to re-consult the connector at
  chunked-prefill boundaries. The patch is 24 lines in
  `vllm/v1/core/sched/scheduler.py`
  ([WorldFlowAI/vllm](https://github.com/WorldFlowAI/vllm), branch
  `feat/connector-mid-request-matching-v0260`); stock connectors see no
  behavior change. This lane is under active validation.
- **Fleet routing**: with several vLLM workers, donors live where they
  were computed. Synapse adds semantic-KV-affinity placement (an llm-d
  EPP scorer, or a Dynamo provider) so recipients land on the worker
  holding their donor. The connector publishes donor events over NATS
  when `SEMBLEND_NATS_URL` is set.
- **Tenancy**: donors and lookups are namespaced by model, tokenizer,
  block size, dtype, LoRA, and the request's `cache_salt` (vLLM's own
  prefix-cache isolation key). Set `cache_salt` per tenant and a request
  never sees another tenant's donors.

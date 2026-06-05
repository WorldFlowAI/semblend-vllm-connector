# Contributing

Thanks for contributing to `semblend-vllm-connector`.

This repo is an out-of-tree vLLM connector. Keep changes small, reviewable, and
consistent with vLLM connector lifecycle semantics.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

With SemBlend provider support:

```bash
pip install -e ".[semblend,dev]"
```

## Pull Requests

Before opening a PR:

- run `make check`;
- add or update tests for behavior changes;
- update docs for user-visible behavior changes;
- keep discovery and materialization behavior clearly separated;
- keep the repo focused on the vLLM connector surface.

Use one of these PR title prefixes when practical:

- `[Bugfix]`
- `[Connector]`
- `[Config]`
- `[Docs]`
- `[Test]`
- `[CI]`

## vLLM Connector Rules

- Do not weaken exact prefix-cache semantics.
- Do not return positive matched tokens unless KV can actually be loaded.
- Do not publish non-identical semantic donor KV into the exact prefix cache.
- Validate model, tokenizer, namespace, cache salt, adapter, and block shape.
- Fail closed to normal vLLM prefill on unsupported cases.

## Contribution Rights

By submitting a contribution, you agree that your contribution is provided under
the Apache-2.0 license used by this repository and that you have the right to
submit it. Maintainers may request a `Signed-off-by` line for larger changes.

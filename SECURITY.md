# Security Policy

## Supported Versions

This project is pre-1.0. Security fixes target the `main` branch until the first
tagged release plan is published.

## Reporting A Vulnerability

Do not open a public issue for suspected vulnerabilities.

Email security reports to `research@worldflowai.com` with:

- affected version or commit;
- reproduction steps;
- impact;
- whether the issue can expose prompts, token IDs, request metadata, or KV data;
- any suggested mitigation.

We will acknowledge credible reports within 5 business days when possible.

## Sensitive Data

This connector should avoid retaining prompt text by default. Keep
`enable_prompt_text=false` unless a deployment explicitly opts in and has the
right data-handling controls.

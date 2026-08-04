# OpenAlex research discovery

This workflow is the secondary discovery source. Google Scholar email alerts remain the primary daily source.

## Schedule

GitHub Actions checks the workflow once per day. A persisted 44-hour state gate makes external discovery run approximately every two days while avoiding month-boundary problems associated with day-of-month cron expressions. Manual runs can set `force` to bypass the gate.

## Query policy

Queries are research-topic phrases defined in `config/openalex_discovery.yaml`. Venue names are not used as discovery filters. The configured themes cover optical and photonic neural networks, free-space and diffractive systems, physical training, optical nonlinear activation, general zeroth-order optimization, LLM zeroth-order optimization, and optical zeroth-order training.

## Acceptance policy

Every returned work passes deterministic project recognition before persistence. A work is appended only when it matches the project assigned to the query. No LLM is involved.

Cross-source duplicate checks run in this order:

1. OpenAlex work ID
2. DOI
3. normalized title plus publication year
4. content fingerprint

Accepted records are appended to `data/paper_registry.jsonl`. Existing Scholar records are never overwritten.

## Source-scoped mandatory rules

Tier 1 venue mandatory summaries and the optical-ZO mandatory override apply only to Google Scholar alert records. OpenAlex discoveries remain eligible for scoring, but they do not bypass the normal relevance and daily-budget stages.

## Data handling

The workflow stores only normalized candidate fields, discovery state, and aggregate manifests. It does not persist raw OpenAlex responses and does not download full text.

## State outputs

- `state/openalex_discovery_state.json`
- `state/openalex_discovery_manifest.json`
- updated `data/paper_registry.jsonl`
- refreshed enrichment, recognition, routing manifest, and queues

The workflow uses the repository secret `OPENALEX_API_KEY`.

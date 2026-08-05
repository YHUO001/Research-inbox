# OpenAlex research discovery

OpenAlex is the secondary discovery source, but it now runs inside the same daily workflow as Google Scholar email ingestion. Both sources append to the same immutable registry and then share one deduplication, enrichment, routing, scoring, and budget-selection path.

## Schedule

`Daily Research Inbox` runs at 08:17 Asia/Singapore. Its 20:47 schedule is recovery-only and runs only when the complete morning pipeline did not record success on the same local date.

OpenAlex has a persisted 20-hour provider gate. This permits one normal scan per day while preventing the evening recovery run from repeating an already successful morning OpenAlex scan. A manual workflow run can set `force_openalex` to bypass the provider gate.

The unified success marker is written only when all components succeed:

1. Google Scholar ingestion
2. OpenAlex discovery or a valid provider interval skip
3. deterministic routing
4. metadata enrichment
5. scoring and queue construction

If any component fails, partial diagnostics and accepted registry records are persisted, the workflow fails, and the evening recovery run retries the incomplete path.

## Query policy

Queries are research-topic phrases defined in `config/openalex_discovery.yaml`. Venue names are not used as discovery filters. The configured themes cover optical and photonic neural networks, free-space and diffractive systems, physical training, optical nonlinear activation, general zeroth-order optimization, LLM zeroth-order optimization, and optical zeroth-order training.

## Acceptance policy

Every returned work passes deterministic project recognition before persistence. A work is appended only when it matches the project assigned to the query. No LLM is involved.

Cross-source duplicate checks run in this order:

1. OpenAlex work ID
2. DOI
3. normalized title plus publication year
4. content fingerprint

Accepted records are appended to `data/paper_registry.jsonl`. Existing Scholar records are never overwritten. After both discovery sources finish, the entire unified registry is re-enriched, re-routed, and rescored.

## Source-scoped mandatory rules

Tier 1 venue mandatory summaries and the optical-ZO mandatory override apply only to Google Scholar alert records. OpenAlex discoveries remain eligible for scoring, but they do not bypass the normal relevance and daily-budget stages.

## Data handling

The workflow stores only normalized candidate fields, discovery state, and aggregate manifests. It does not persist raw OpenAlex responses and does not download full text during discovery.

## State outputs

- `state/unified_discovery_state.json`
- `state/openalex_discovery_state.json`
- `state/openalex_discovery_manifest.json`
- updated `data/paper_registry.jsonl`
- refreshed enrichment, recognition, routing, scoring, and candidate queues

The workflow uses the repository secret `OPENALEX_API_KEY`.

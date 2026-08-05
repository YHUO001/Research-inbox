# Fully automatic daily research pipeline

## Schedule (Asia/Singapore)

- 08:17 — one unified discovery run: Google Scholar alerts and OpenAlex feed the same registry, deduplication, enrichment, routing, scoring, and selection path.
- 20:47 — recovery run only when the complete unified morning pipeline was not successful on the same local date.
- 21:30 — one automatic summary batch, limited to three validated summaries.
- 22:15 — one aggregate daily email, sent only when a completed non-empty digest exists.

OpenAlex retains a 20-hour provider interval inside the unified workflow. Therefore, the recovery run does not repeat a successful morning OpenAlex scan, but it can retry OpenAlex when the provider step itself failed.

## Unified discovery transaction

Both discovery sources append normalized candidates to `data/paper_registry.jsonl`. Cross-source identity checks use OpenAlex ID, DOI, normalized title plus year, and content fingerprint. After both collectors finish, the whole registry is deterministically routed, enriched through Crossref/OpenAlex, rescored, and reduced to the common LLM candidate queue.

`state/unified_discovery_state.json` advances only when Scholar ingestion, OpenAlex discovery, routing, enrichment, and scoring all succeed. Partial accepted records and diagnostics are persisted before a failed job exits, allowing the evening recovery run to continue without losing work or duplicating already accepted papers.

## Automatic summary transaction

The summary workflow filters `completed_candidate_ids` before any provider call. The remaining `summary_slot` candidates are enriched with temporary open-access method context, summarized with DeepSeek Pro, and checked locally for schema validity, Chinese narrative, method depth, numerical grounding, and deterministic ONN architecture evidence.

Only a complete all-or-nothing batch can update `state/summary_history.json`. A failed or partial batch leaves every candidate incomplete so that it can be retried on a later day. Human review workflows are not part of the production path.

## Long-term knowledge base

Every automatically completed paper is added idempotently to:

- `data/knowledge_base/papers.jsonl` — canonical searchable paper and summary records.
- `data/knowledge_base/index.json` — indexes by candidate ID, DOI, project, year, venue, digest date, and method tags.
- `data/knowledge_base/index.md` — human-readable project index.
- `state/knowledge_base_manifest.json` — counts and file hashes.

Public full-text method context remains temporary runner data and is never written to the knowledge base.

## Daily email

The email workflow reads only a `completed_automatic` digest and sends at most one message for a local date. Empty or failed digest days are skipped. Delivery is protected by both a deterministic RFC Message-ID search in Gmail Sent Mail and `state/email_delivery_state.json`, so workflow retries do not send duplicate messages.

The configured recipient is allowlisted in `config/email_delivery.yaml`. Gmail OAuth must include both `gmail.readonly` and `gmail.send`; no Gmail modify or full-mailbox scope is requested.

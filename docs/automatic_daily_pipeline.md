# Fully automatic daily research pipeline

## Schedule (Asia/Singapore)

- 08:17 — primary Google Scholar alert ingestion.
- 20:47 — recovery ingestion only when no successful Scholar run was recorded on the same local date.
- 21:30 — one automatic summary batch, limited to three validated summaries.
- 22:15 — one aggregate daily email, sent only when a completed non-empty digest exists.
- OpenAlex discovery retains its independent 44-hour minimum interval.

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

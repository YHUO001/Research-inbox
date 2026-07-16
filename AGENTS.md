# Research Inbox Agent Instructions

## Scope

This repository implements a low-token academic paper discovery, filtering, summarization, and archival pipeline.

## Operating principles

- Prefer deterministic scripts for collection, normalization, deduplication, scoring, and storage.
- Use an LLM only after deterministic filtering has reduced the candidate set.
- Process only new records since the last successful checkpoint.
- Do not scan an entire mailbox, paper library, or historical dataset unless explicitly requested.
- Do not download or analyze full-text PDFs during routine automated runs.
- Never fabricate DOI, author, venue, date, citation, or bibliographic metadata.
- Mark unresolved metadata explicitly rather than guessing.
- Do not commit credentials, API keys, email contents, downloaded PDFs, or private research data.
- Keep automated outputs concise and structured.
- Respect all limits in `config/pipeline.yaml`.
- Keep `llm.enabled: false` until collection, normalization, and scoring have been tested.

## Change policy

- Make changes on a feature branch and submit them through a pull request.
- Avoid modifying scientific conclusions automatically.
- Add tests for parsing, normalization, deduplication, or scoring logic when those modules are introduced.

# Metadata enrichment foundation

This stage enriches the immutable Scholar-derived registry without modifying
`data/paper_registry.jsonl`.

## Providers

- Crossref is used for DOI lookup and conservative bibliographic matching.
  Authentication is not required. `RESEARCH_INBOX_CONTACT_EMAIL` is optional
  and enables Crossref's polite identification mode.
- OpenAlex is used for DOI lookup, abstracts, open-access locations, and
  citation counts. OpenAlex currently requires a free API key. The workflow
  reads it only from the `OPENALEX_API_KEY` GitHub Actions secret.
- No full text is downloaded.
- Raw API responses are never persisted. Only normalized, Schema-validated
  metadata and a normalized cache are written to `automation-state`.

## Matching policy

Exact DOI lookup is preferred. Title searches are accepted only when:

1. normalized title similarity is at least 0.96;
2. year or first-author surname corroborates the match;
3. year and first-author evidence do not conflict; and
4. the top result is not ambiguous with another result.

Unresolved records retain their Scholar values and are not guessed.

## Manual backfill

Run **Actions → Enrich Paper Registry → Run workflow → main**.

The workflow writes:

- `data/enriched_paper_registry.jsonl`
- `state/enrichment_manifest.json`
- `state/enrichment_cache.json`

Re-running with unchanged source data and unexpired cache entries is
idempotent and does not create a new state commit.

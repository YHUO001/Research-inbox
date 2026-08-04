# Scripts

Deterministic collection and processing scripts will live here.

Planned order:

1. Parse Google Scholar alert emails into raw records.
2. Normalize records into `schemas/paper_record.schema.json`.
3. Enrich metadata through OpenAlex and Crossref.
4. Deduplicate papers by DOI, source IDs, and normalized title.
5. Score relevance locally before any LLM call.
6. Export selected records to the archival layer.

Scripts must read limits from `config/pipeline.yaml` and update checkpoints only after successful completion.

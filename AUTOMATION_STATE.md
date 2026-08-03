# Automation State Branch

This branch stores only lightweight, non-secret runtime state for the scheduled research inbox.

Allowed content:

- Gmail message IDs and checkpoint timestamps;
- deterministic paper metadata extracted from public Scholar alerts;
- candidate fingerprints and routing queues;
- sanitized failure counters and error classes.

Forbidden content:

- OAuth client secrets or refresh tokens;
- raw MIME, complete email bodies, or private correspondence;
- downloaded PDFs or full paper text;
- local databases and execution logs.

The scheduled workflow is the only expected writer. Changes are serialized with a GitHub Actions concurrency group.

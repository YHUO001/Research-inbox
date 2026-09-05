"""Summary-layer adapter for deterministic author metadata.

The summary pipeline can call this adapter before rendering user-facing output.
It deliberately does not infer author background or academic reputation.
"""

from __future__ import annotations

from typing import Any

from scripts.enrich.author_metadata import build_author_context


def attach_author_context(
    summary: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    """Attach metadata-derived author context without changing the summary text."""
    enriched = dict(summary)
    enriched["author_context"] = build_author_context(source)
    return enriched

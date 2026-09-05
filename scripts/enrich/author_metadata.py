"""Author metadata enrichment for selected summary candidates.

This module intentionally handles deterministic metadata only. It does not ask the
LLM to infer authorship roles.
"""

from __future__ import annotations

from typing import Any


def normalize_author_record(author: dict[str, Any], position: str) -> dict[str, Any]:
    return {
        "name": author.get("name") or "",
        "position": position,
        "institutions": list(author.get("institutions") or []),
        "orcid": author.get("orcid"),
    }


def build_author_context(record: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic context consumed by the summary layer.

    Existing metadata providers may expose different fields. Missing information
    is represented explicitly instead of being guessed.
    """
    authors = list(record.get("authors") or [])
    corresponding = list(record.get("corresponding_authors") or [])

    return {
        "first_author": normalize_author_record(authors[0], "first_author") if authors else None,
        "corresponding_authors": [
            normalize_author_record(author, "corresponding_author")
            for author in corresponding
        ],
        "source": "metadata_only",
    }

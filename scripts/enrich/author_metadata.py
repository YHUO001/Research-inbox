"""Author metadata enrichment for selected summary candidates.

This module intentionally handles deterministic metadata only. It does not ask the
LLM to infer authorship roles.
"""

from __future__ import annotations

from typing import Any


def _institutions(value: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in value or []:
        if isinstance(item, dict):
            name = str(item.get("display_name") or item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            output.append(name)
    return output


def normalize_author_record(author: Any, position: str) -> dict[str, Any]:
    if isinstance(author, dict):
        name = str(author.get("name") or author.get("display_name") or "").strip()
        institutions = _institutions(author.get("institutions") or author.get("affiliations"))
        orcid = author.get("orcid")
        openalex_id = author.get("openalex_id") or author.get("id")
        is_corresponding = author.get("is_corresponding") is True
        source_position = str(author.get("position") or author.get("author_position") or position)
    else:
        name = str(author or "").strip()
        institutions = []
        orcid = None
        openalex_id = None
        is_corresponding = False
        source_position = position
    return {
        "name": name,
        "position": source_position,
        "institutions": institutions,
        "orcid": orcid,
        "openalex_id": openalex_id,
        "is_corresponding": is_corresponding,
    }


def build_author_context(record: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic context consumed by the summary layer.

    Existing metadata providers may expose different fields. Missing information
    is represented explicitly instead of being guessed. In particular, the last
    author is never treated as corresponding author unless metadata says so.
    """
    raw_authors = list(record.get("authors") or [])
    authors = [
        normalize_author_record(author, "first" if index == 0 else "additional")
        for index, author in enumerate(raw_authors)
    ]
    authors = [author for author in authors if author.get("name")]

    explicit_corresponding = list(record.get("corresponding_authors") or [])
    if explicit_corresponding:
        corresponding = [
            normalize_author_record(author, "corresponding")
            for author in explicit_corresponding
        ]
    else:
        corresponding = [author for author in authors if author.get("is_corresponding") is True]
    corresponding = [author for author in corresponding if author.get("name")]

    first_author = next(
        (author for author in authors if author.get("position") in {"first", "first_author"}),
        authors[0] if authors else None,
    )

    institutions: list[str] = []
    seen: set[str] = set()
    for author in authors:
        for institution in author.get("institutions") or []:
            key = str(institution).casefold()
            if key not in seen:
                seen.add(key)
                institutions.append(str(institution))

    return {
        "authors": authors,
        "first_author": first_author,
        "corresponding_authors": corresponding,
        "institutions": institutions,
        "team_scope": (
            "multi_institution"
            if len(institutions) > 1
            else "single_institution"
            if len(institutions) == 1
            else "unknown"
        ),
        "corresponding_author_status": (
            "explicit" if corresponding else "not_explicitly_available"
        ),
        "source": "metadata_only",
    }

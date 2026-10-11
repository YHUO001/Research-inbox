"""Selected-paper author and affiliation enrichment.

This module runs only after daily selection (normally at most three papers).  It
uses OpenAlex work metadata to recover author affiliations and explicit
corresponding-author flags without asking the summary LLM to infer them.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from scripts.enrich.author_metadata import build_author_context

OPENALEX_BASE_URL = "https://api.openalex.org"


def _institution_names(authorship: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for institution in authorship.get("institutions") or []:
        if not isinstance(institution, dict):
            continue
        name = str(institution.get("display_name") or institution.get("name") or "").strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def context_from_openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic author context from an OpenAlex work response."""
    authors: list[dict[str, Any]] = []
    for authorship in work.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        if not isinstance(author, dict):
            continue
        name = str(author.get("display_name") or "").strip()
        if not name:
            continue
        authors.append(
            {
                "name": name,
                "orcid": author.get("orcid"),
                "openalex_id": author.get("id"),
                "institutions": _institution_names(authorship),
                "position": str(authorship.get("author_position") or ""),
                "is_corresponding": authorship.get("is_corresponding") is True,
            }
        )

    first = next((item for item in authors if item.get("position") == "first"), None)
    if first is None and authors:
        first = authors[0]
    corresponding = [item for item in authors if item.get("is_corresponding") is True]

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
        "first_author": first,
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
        "source": "openalex_selected_metadata",
    }


def _work_identifier(source: dict[str, Any]) -> str | None:
    openalex_id = str(source.get("openalex_id") or "").strip()
    if openalex_id:
        return openalex_id.rstrip("/").rsplit("/", 1)[-1]
    doi = str(source.get("doi") or "").strip()
    if doi:
        doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return quote(f"https://doi.org/{doi}", safe=":/")
    return None


def fetch_openalex_author_context(
    source: dict[str, Any],
    *,
    api_key: str | None,
    timeout_seconds: float = 15.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any] | None:
    identifier = _work_identifier(source)
    if not identifier or not api_key:
        return None
    query = urlencode({"api_key": api_key, "select": "id,authorships"})
    url = f"{OPENALEX_BASE_URL}/works/{identifier}?{query}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ResearchInbox/0.4 (selected author metadata)",
        },
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return context_from_openalex_work(payload)


def enrich_selected_author_context(
    source: dict[str, Any],
    *,
    api_key: str | None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Return best available deterministic context for one selected paper.

    Network failure is non-fatal: the daily digest still receives the author names
    already present in the selected candidate and explicitly marks unavailable
    corresponding-author or affiliation metadata instead of guessing.
    """
    context = fetch_openalex_author_context(source, api_key=api_key, opener=opener)
    if context and context.get("authors"):
        context["lookup_status"] = "enriched"
        return context
    fallback = build_author_context(source)
    fallback["lookup_status"] = "fallback_metadata"
    return fallback

"""Selected-paper author and affiliation enrichment.

This module runs only after daily selection (normally at most three papers). It
uses public work metadata to recover author affiliations and explicit
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
CROSSREF_BASE_URL = "https://api.crossref.org"


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


def _finalize_context(authors: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
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
        "source": source,
    }


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
    return _finalize_context(authors, source="openalex_selected_metadata")


def context_from_crossref_message(message: dict[str, Any]) -> dict[str, Any]:
    """Build author/affiliation context from Crossref without inferring roles."""
    authors: list[dict[str, Any]] = []
    for index, author in enumerate(message.get("author") or []):
        if not isinstance(author, dict):
            continue
        name = " ".join(
            value
            for value in (
                str(author.get("given") or "").strip(),
                str(author.get("family") or "").strip(),
            )
            if value
        ).strip()
        if not name:
            continue
        institutions = []
        seen: set[str] = set()
        for affiliation in author.get("affiliation") or []:
            if not isinstance(affiliation, dict):
                continue
            institution = str(affiliation.get("name") or "").strip()
            key = institution.casefold()
            if institution and key not in seen:
                seen.add(key)
                institutions.append(institution)
        authors.append(
            {
                "name": name,
                "orcid": author.get("ORCID"),
                "openalex_id": None,
                "institutions": institutions,
                "position": "first" if index == 0 else "additional",
                "is_corresponding": False,
            }
        )
    return _finalize_context(authors, source="crossref_selected_metadata")


def _work_identifier(source: dict[str, Any]) -> str | None:
    openalex_id = str(source.get("openalex_id") or "").strip()
    if openalex_id:
        return openalex_id.rstrip("/").rsplit("/", 1)[-1]
    doi = str(source.get("doi") or "").strip()
    if doi:
        doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return quote(f"https://doi.org/{doi}", safe=":/")
    return None


def _doi(source: dict[str, Any]) -> str | None:
    value = str(source.get("doi") or "").strip()
    if not value:
        return None
    return value.removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def _read_json(request: Request, *, timeout_seconds: float, opener: Callable[..., Any]) -> dict[str, Any] | None:
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def fetch_openalex_author_context(
    source: dict[str, Any],
    *,
    api_key: str | None,
    timeout_seconds: float = 15.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any] | None:
    identifier = _work_identifier(source)
    if not identifier:
        return None
    params: dict[str, str] = {"select": "id,authorships"}
    if api_key:
        params["api_key"] = api_key
    query = urlencode(params)
    request = Request(
        f"{OPENALEX_BASE_URL}/works/{identifier}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "ResearchInbox/0.4 (selected author metadata)",
        },
        method="GET",
    )
    payload = _read_json(request, timeout_seconds=timeout_seconds, opener=opener)
    if not payload:
        return None
    context = context_from_openalex_work(payload)
    return context if context.get("authors") else None


def fetch_crossref_author_context(
    source: dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any] | None:
    doi = _doi(source)
    if not doi:
        return None
    request = Request(
        f"{CROSSREF_BASE_URL}/works/{quote(doi, safe='')}",
        headers={
            "Accept": "application/json",
            "User-Agent": "ResearchInbox/0.4 (selected author metadata)",
        },
        method="GET",
    )
    payload = _read_json(request, timeout_seconds=timeout_seconds, opener=opener)
    message = (payload or {}).get("message")
    if not isinstance(message, dict):
        return None
    context = context_from_crossref_message(message)
    return context if context.get("authors") else None


def enrich_selected_author_context(
    source: dict[str, Any],
    *,
    api_key: str | None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Return best available deterministic context for one selected paper.

    OpenAlex is preferred because it may expose explicit corresponding-author
    flags and structured institutions. Crossref supplies a public affiliation
    fallback. Network failure is non-fatal and never triggers role inference.
    """
    context = fetch_openalex_author_context(source, api_key=api_key, opener=opener)
    if context and context.get("authors"):
        context["lookup_status"] = "enriched"
        return context
    context = fetch_crossref_author_context(source, opener=opener)
    if context and context.get("authors"):
        context["lookup_status"] = "enriched"
        return context
    fallback = build_author_context(source)
    fallback["lookup_status"] = "fallback_metadata"
    return fallback

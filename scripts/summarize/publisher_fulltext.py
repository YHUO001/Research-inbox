from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import urlsplit

from scripts.summarize.fulltext_methods import MethodContext, collect_method_context
from scripts.summarize.springer_openaccess import collect_springer_openaccess_context


SPRINGER_DOI_PREFIXES = (
    "10.1007/",
    "10.1038/",
    "10.1057/",
    "10.1134/",
    "10.1140/",
    "10.1186/",
    "10.1245/",
    "10.1385/",
    "10.2165/",
    "10.3758/",
)
SPRINGER_HOST_SUFFIXES = (
    "springer.com",
    "springernature.com",
    "nature.com",
    "biomedcentral.com",
)
SPRINGER_VENUE_MARKERS = (
    "springer",
    "nature ",
    "nature communications",
    "scientific reports",
    "bmc ",
    "biomed central",
)
NON_RETRYABLE_SPRINGER_HTTP = (401, 403)


def _normalized_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^doi\s*:\s*", "", text)
    if text.startswith(("http://", "https://")):
        parsed = urlsplit(text)
        if parsed.netloc.lower() in {"doi.org", "dx.doi.org", "www.doi.org"}:
            text = parsed.path.lstrip("/")
    return text


def _host(value: Any) -> str:
    text = str(value or "").strip()
    if not text.startswith(("http://", "https://")):
        return ""
    return (urlsplit(text).hostname or "").lower()


def _host_matches(host: str, suffixes: tuple[str, ...]) -> bool:
    return bool(host) and any(
        host == suffix or host.endswith(f".{suffix}") for suffix in suffixes
    )


def is_springer_nature_source(source: dict[str, Any]) -> bool:
    doi = _normalized_doi(source.get("doi"))
    if any(doi.startswith(prefix) for prefix in SPRINGER_DOI_PREFIXES):
        return True

    for value in (source.get("open_access_url"), source.get("landing_page")):
        if _host_matches(_host(value), SPRINGER_HOST_SUFFIXES):
            return True

    venue = f" {str(source.get('venue') or '').strip().lower()} "
    return any(marker in venue for marker in SPRINGER_VENUE_MARKERS)


def _valid_public_url(value: Any, *, allow_doi: bool = False) -> str | None:
    text = str(value or "").strip()
    if not text.startswith(("http://", "https://")):
        return None
    host = _host(text)
    if not host:
        return None
    if _host_matches(host, SPRINGER_HOST_SUFFIXES):
        return None
    if host in {"doi.org", "dx.doi.org", "www.doi.org"} and not allow_doi:
        return None
    return text


def arxiv_public_urls(doi: Any) -> list[str]:
    normalized = _normalized_doi(doi)
    prefix = "10.48550/arxiv."
    if not normalized.startswith(prefix):
        return []
    identifier = normalized[len(prefix) :]
    if not identifier:
        return []
    return [
        f"https://arxiv.org/html/{identifier}",
        f"https://arxiv.org/pdf/{identifier}",
    ]


def routed_open_source(
    source: dict[str, Any], *, include_doi_fallback: bool
) -> dict[str, Any] | None:
    urls: list[str] = []
    for value in arxiv_public_urls(source.get("doi")):
        if value not in urls:
            urls.append(value)

    for value in (source.get("open_access_url"), source.get("landing_page")):
        candidate = _valid_public_url(value)
        if candidate and candidate not in urls:
            urls.append(candidate)

    if not urls and include_doi_fallback:
        doi = _normalized_doi(source.get("doi"))
        if doi:
            urls.append(f"https://doi.org/{doi}")

    if not urls:
        return None
    return {
        "candidate_id": source.get("candidate_id") or source.get("id"),
        "open_access_url": urls[0],
        "landing_page": urls[1] if len(urls) > 1 else None,
        "doi": None,
    }


def _auth_failure(context: MethodContext) -> MethodContext:
    error = str(context.error or "")
    if not any(f"HTTP {code}" in error for code in NON_RETRYABLE_SPRINGER_HTTP):
        return context
    return MethodContext(
        candidate_id=context.candidate_id,
        status="authentication_failed",
        source_url=context.source_url,
        media_type=context.media_type,
        section_headings=list(context.section_headings),
        text="",
        error=error,
    )


def _combine_failures(
    source: dict[str, Any],
    primary: MethodContext,
    secondary: MethodContext | None,
) -> MethodContext:
    if secondary is None:
        return primary
    errors = "; ".join(
        value for value in (primary.error, secondary.error) if value
    )
    return MethodContext(
        candidate_id=str(source.get("candidate_id") or source.get("id") or "unknown"),
        status=(
            primary.status
            if primary.status == "authentication_failed"
            else secondary.status
        ),
        source_url=secondary.source_url or primary.source_url,
        media_type=secondary.media_type or primary.media_type,
        section_headings=list(secondary.section_headings),
        text="",
        error=errors[:1000] or None,
    )


def collect_publisher_routed_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    api_key: str | None = None,
    api_fetcher: Callable[..., bytes] | None = None,
    direct_loader: Callable[..., MethodContext] = collect_method_context,
) -> MethodContext:
    candidate_id = str(source.get("candidate_id") or source.get("id") or "unknown")
    allow_direct = bool(config.get("allow_non_springer_open_urls", True))

    if not is_springer_nature_source(source):
        alternative = routed_open_source(source, include_doi_fallback=True)
        if alternative is None or not allow_direct:
            return MethodContext(
                candidate_id,
                "not_available",
                None,
                None,
                [],
                "",
                "No eligible publisher or repository open URL; fell back to abstract",
            )
        return direct_loader(alternative, config=config)

    kwargs: dict[str, Any] = {
        "config": config,
        "api_key": api_key,
    }
    if api_fetcher is not None:
        kwargs["fetcher"] = api_fetcher
    official = _auth_failure(collect_springer_openaccess_context(source, **kwargs))
    if official.status == "used":
        return official

    if not allow_direct:
        return official

    alternative = routed_open_source(source, include_doi_fallback=False)
    if alternative is None:
        return official
    direct = direct_loader(alternative, config=config)
    if direct.status == "used":
        return direct
    return _combine_failures(source, official, direct)

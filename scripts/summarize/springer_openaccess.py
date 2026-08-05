from __future__ import annotations

import os
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from scripts.summarize.fulltext_methods import (
    MethodContext,
    collect_method_context,
    is_method_heading,
    normalize_space,
    text_windows,
    unique_paragraphs,
)


BLOCKED_DIRECT_HOST_SUFFIXES = (
    "doi.org",
    "nature.com",
    "springer.com",
    "springernature.com",
)


def normalize_doi(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.IGNORECASE)
    if text.startswith(("http://", "https://")):
        parsed = urlsplit(text)
        if parsed.netloc.lower() not in {"doi.org", "dx.doi.org", "www.doi.org"}:
            return None
        text = parsed.path.lstrip("/")
    text = text.strip()
    if not re.fullmatch(r"10\.\d{4,9}/\S+", text, flags=re.IGNORECASE):
        return None
    return text


def api_audit_url(endpoint: str, doi: str) -> str:
    return f"{endpoint}?{urlencode({'q': f'doi:{doi}', 'p': 1})}"


def api_request_url(endpoint: str, doi: str, api_key: str) -> str:
    return f"{endpoint}?{urlencode({'q': f'doi:{doi}', 'p': 1, 'api_key': api_key})}"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _element_text(element: ElementTree.Element) -> str:
    return normalize_space(" ".join(element.itertext()))


def extract_jats_method_context(
    payload: bytes,
    *,
    maximum_characters: int,
) -> tuple[str, list[str]]:
    root = ElementTree.fromstring(payload)
    records = [
        element
        for element in root.iter()
        if _local_name(element.tag) in {"article", "book-part"}
    ]
    if not records:
        return "", []

    paragraphs: list[str] = []
    headings: list[str] = []
    for record in records:
        for section in record.iter():
            if _local_name(section.tag) != "sec":
                continue
            title = ""
            for child in list(section):
                if _local_name(child.tag) == "title":
                    title = _element_text(child)
                    break
            section_type = normalize_space(str(section.attrib.get("sec-type") or ""))
            if not is_method_heading(title) and not is_method_heading(section_type):
                continue
            heading = title or section_type or "Methods"
            headings.append(heading)
            for item in section.iter():
                if _local_name(item.tag) in {"p", "list-item"}:
                    paragraphs.append(_element_text(item))

    blocks: list[str] = []
    used = 0
    for paragraph in unique_paragraphs(paragraphs):
        remaining = maximum_characters - used
        if remaining <= 0:
            break
        value = paragraph[:remaining]
        blocks.append(value)
        used += len(value) + 2
    if blocks:
        return "\n\n".join(blocks), list(dict.fromkeys(headings))

    body_text = "\n".join(
        _element_text(element)
        for record in records
        for element in record.iter()
        if _local_name(element.tag) in {"title", "p"}
    )
    return text_windows(body_text, maximum_characters=maximum_characters)


def default_api_fetch(
    url: str,
    *,
    timeout_seconds: float,
    maximum_bytes: int,
) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "ResearchInbox/0.6 (+https://github.com/YHUO001/Research-inbox)",
            "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ValueError("Springer Open Access response exceeded byte limit")
        return payload


def collect_springer_openaccess_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    api_key: str | None = None,
    fetcher: Callable[..., bytes] = default_api_fetch,
) -> MethodContext:
    candidate_id = str(source.get("candidate_id") or source.get("id") or "unknown")
    doi = normalize_doi(source.get("doi"))
    endpoint = str(
        config.get("springer_openaccess_endpoint")
        or "https://api.springernature.com/openaccess/jats"
    ).rstrip("?")
    if not doi:
        return MethodContext(
            candidate_id,
            "not_available",
            None,
            None,
            [],
            "",
            "DOI unavailable for Springer Open Access lookup",
        )

    key = api_key or os.environ.get(
        str(config.get("springer_api_key_env") or "SPRINGER_NATURE_API_KEY")
    )
    audit_url = api_audit_url(endpoint, doi)
    if not key:
        return MethodContext(
            candidate_id,
            "not_available",
            audit_url,
            "application/xml+jats",
            [],
            "",
            "Springer Open Access API key unavailable; fell back to abstract",
        )

    try:
        payload = fetcher(
            api_request_url(endpoint, doi, key),
            timeout_seconds=float(config.get("timeout_seconds") or 10),
            maximum_bytes=int(config.get("maximum_download_bytes") or 15_000_000),
        )
        text, headings = extract_jats_method_context(
            payload,
            maximum_characters=int(config.get("maximum_method_characters") or 12_000),
        )
    except HTTPError as error:
        message = f"Springer Open Access HTTP {error.code}; fell back to abstract"
    except URLError as error:
        message = f"Springer Open Access network error: {type(error.reason).__name__}; fell back to abstract"
    except Exception as error:
        message = f"Springer Open Access {type(error).__name__}; fell back to abstract"
    else:
        if text:
            return MethodContext(
                candidate_id,
                "used",
                audit_url,
                "application/xml+jats",
                headings,
                text,
            )
        message = "Springer Open Access returned no method-oriented JATS content; fell back to abstract"

    return MethodContext(
        candidate_id,
        "not_available",
        audit_url,
        "application/xml+jats",
        [],
        "",
        message,
    )


def _direct_url_allowed(value: Any) -> bool:
    text = str(value or "").strip()
    if not text.startswith(("http://", "https://")):
        return False
    host = (urlsplit(text).hostname or "").lower()
    return bool(host) and not any(
        host == suffix or host.endswith(f".{suffix}")
        for suffix in BLOCKED_DIRECT_HOST_SUFFIXES
    )


def non_springer_open_source(source: dict[str, Any]) -> dict[str, Any] | None:
    urls: list[str] = []
    for value in (source.get("open_access_url"), source.get("landing_page")):
        text = str(value or "").strip()
        if _direct_url_allowed(text) and text not in urls:
            urls.append(text)
    if not urls:
        return None
    return {
        "candidate_id": source.get("candidate_id") or source.get("id"),
        "open_access_url": urls[0],
        "landing_page": urls[1] if len(urls) > 1 else None,
        "doi": None,
    }


def collect_official_or_open_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    api_key: str | None = None,
    api_fetcher: Callable[..., bytes] = default_api_fetch,
    direct_loader: Callable[..., MethodContext] = collect_method_context,
) -> MethodContext:
    official = collect_springer_openaccess_context(
        source,
        config=config,
        api_key=api_key,
        fetcher=api_fetcher,
    )
    if official.status == "used":
        return official

    alternative = non_springer_open_source(source)
    if alternative is not None and bool(config.get("allow_non_springer_open_urls", True)):
        direct = direct_loader(alternative, config=config)
        if direct.status == "used":
            return direct
        errors = "; ".join(
            value for value in (official.error, direct.error) if value
        )
        return MethodContext(
            str(source.get("candidate_id") or source.get("id") or "unknown"),
            direct.status,
            direct.source_url or official.source_url,
            direct.media_type or official.media_type,
            direct.section_headings,
            "",
            errors[:1000] or None,
        )
    return official

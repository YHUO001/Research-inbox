from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

PARSER_VERSION = 3

SCHOLAR_RESULT_URL = re.compile(r"https?://scholar\.google\.[^/]+/scholar_url\?", re.I)
RESOURCE_PREFIX_RE = re.compile(r"^\s*(?:\[(?:HTML|PDF|TXT|DOC|DOCX)\]\s*)+", re.I)
MARKDOWN_LINK = re.compile(
    r"^(?:\[(?:HTML|PDF|TXT|DOC|DOCX)\]\s*)?"
    r"\[(?P<title>.+?)\]"
    r"\((?P<href>https?://scholar\.google\.[^/]+/scholar_url\?[^\n]+)\)\s*$",
    re.I | re.M,
)
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
ARXIV_RE = re.compile(r"(?i)(?:arxiv[:\s]*)?(\d{4}\.\d{4,5})(?:v\d+)?")
PMID_RE = re.compile(r"(?i)PMID[:\s]+(\d{6,9})")
FOOTER_MARKERS = (
    "[保存]",
    "[Save]",
    "[Twitter]",
    "Google 学术搜索发送此邮件",
    "Google Scholar email alert",
    "列出快讯",
    "取消快讯",
)
SHARE_TEXT = {"保存", "save", "twitter", "linkedin", "facebook"}
RESOURCE_TEXT = {
    "[html]", "[pdf]", "[txt]", "[doc]", "[docx]",
    "html", "pdf", "txt", "doc", "docx",
}
SHARE_HREF_MARKERS = ("/citations?", "/scholar_share?", "/scholar_alerts?")
DOCUMENT_COVER_PATTERNS = (
    re.compile(r"\bmaster of science\b.*\ball rights reserved\b", re.I),
    re.compile(r"\bdoctor of philosophy\b.*\ball rights reserved\b", re.I),
    re.compile(r"^\s*copyright\s*(?:©|\(c\))", re.I),
    re.compile(r"^\s*(?:table of contents|references|bibliography)\s*$", re.I),
)


@dataclass(frozen=True)
class SourceContext:
    message_id: str
    received_at: str
    sender: str
    subject: str
    thread_id: str | None = None
    spf: str | None = None
    dkim: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", title)
    return normalize_spaces(RESOURCE_PREFIX_RE.sub("", value))


def normalize_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", title).lower().strip()
    value = re.sub(r"^\s*(?:19|20)\d{2}\s*[-_:]\s*", "", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return normalize_spaces(value)


def sanitize_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    tracking = {"utm_source", "utm_medium", "utm_campaign", "ved", "usg"}
    pairs = []
    for pair in parsed.query.split("&") if parsed.query else []:
        if pair.split("=", 1)[0] not in tracking:
            pairs.append(pair)
    return urlunparse(parsed._replace(query="&".join(pairs), fragment=""))


def unwrap_scholar_url(href: str) -> str | None:
    target = parse_qs(urlparse(href).query).get("url", [None])[0]
    return sanitize_url(unquote(target)) if target else sanitize_url(href)


def clean_doi(value: str) -> str:
    value = value.rstrip(".,;:)]}")
    for suffix in (".short", ".abstract", ".full", ".pdf"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
    return value


def identifier_evidence(value: str | None, source: str | None) -> dict[str, str | None]:
    return {
        "value": value,
        "verification_status": "regex_extracted" if value else "missing",
        "source": source if value else None,
    }


def extract_identifiers(
    title: str,
    metadata: str | None,
    snippet: str | None,
    url: str | None,
) -> dict:
    fields = [
        ("title", title),
        ("metadata_line", metadata or ""),
        ("snippet", snippet or ""),
        ("url", url or ""),
    ]
    values: dict[str, tuple[str | None, str | None]] = {
        "doi": (None, None),
        "arxiv_id": (None, None),
        "pmid": (None, None),
    }
    for source, text in fields:
        if values["doi"][0] is None and (match := DOI_RE.search(text)):
            values["doi"] = (clean_doi(match.group(0)), source)
        if values["arxiv_id"][0] is None and (match := ARXIV_RE.search(text)):
            values["arxiv_id"] = (match.group(1), source)
        if values["pmid"][0] is None and (match := PMID_RE.search(text)):
            values["pmid"] = (match.group(1), source)
    return {
        key: identifier_evidence(value, source)
        for key, (value, source) in values.items()
    }


def normalize_metadata_dash(text: str) -> str:
    return re.sub(r"\s+[–—−]\s+", " - ", normalize_spaces(text))


def parse_metadata_line(
    metadata: str | None,
) -> tuple[list[dict], str | None, int | None]:
    if not metadata:
        return [], None, None

    raw = normalize_metadata_dash(metadata)
    left, separator, right = raw.rpartition(" - ")
    if not separator:
        left, right = "", raw

    years = YEAR_RE.findall(right)
    year = int(years[-1]) if years else None
    venue = right
    if year:
        # Remove only the terminal publication year. A venue may itself start
        # with a year, for example "2026 49th MIPRO ..., 2026".
        venue = re.sub(rf",?\s*{year}\s*$", "", venue).strip(" ,-…")
    venue = venue or None

    authors: list[dict] = []
    for name in left.replace("…", "").split(",") if left else []:
        cleaned = name.strip()
        if cleaned:
            authors.append(
                {
                    "name": cleaned,
                    "orcid": None,
                    "verification_status": "raw_email",
                }
            )
    return authors, venue, year


def extract_alert_name(subject: str, body: str) -> str | None:
    quoted = re.search(r'["“](.+?)["”]\s*-\s*', subject)
    if quoted:
        return quoted.group(1).strip()
    footer = re.search(r'关注了\s*\[?\[?["“](.+?)["”]\]?\]?', body)
    return footer.group(1).strip() if footer else None


def compact_snippet(lines: Iterable[str]) -> str | None:
    cleaned: list[str] = []
    for line in lines:
        text = normalize_spaces(line)
        if not text:
            continue
        if text.lower() in RESOURCE_TEXT:
            continue
        if text.lower() in SHARE_TEXT:
            break
        if any(marker.lower() in text.lower() for marker in FOOTER_MARKERS):
            break
        cleaned.append(text)
    return normalize_spaces(" ".join(cleaned)) or None


def split_metadata_and_snippet(
    fragments: list[str],
) -> tuple[str | None, str | None]:
    metadata_parts: list[str] = []
    for index, fragment in enumerate(fragments[:12], start=1):
        metadata_parts.append(fragment)
        candidate = normalize_metadata_dash(" ".join(metadata_parts))
        if len(candidate) > 600:
            break
        if YEAR_RE.search(candidate) and " - " in candidate:
            return candidate, compact_snippet(fragments[index:])
    return None, compact_snippet(fragments)


def markdown_blocks(
    body: str,
) -> list[tuple[str, str, str | None, str | None]]:
    matches = list(MARKDOWN_LINK.finditer(body))
    blocks: list[tuple[str, str, str | None, str | None]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        useful = [
            line.strip()
            for line in body[start:end].splitlines()
            if line.strip()
        ]
        metadata = useful[0] if useful else None
        snippet = compact_snippet(useful[1:]) if len(useful) > 1 else None
        blocks.append(
            (
                clean_title(match.group("title")),
                match.group("href"),
                metadata,
                snippet,
            )
        )
    return blocks


def anchor_href(anchor: Tag) -> str:
    return str(anchor.get("href", ""))


def is_primary_result_anchor(anchor: Tag) -> bool:
    return bool(
        anchor.name == "a"
        and SCHOLAR_RESULT_URL.search(anchor_href(anchor))
    )


def is_inside(node: NavigableString, ancestor: Tag) -> bool:
    parent = node.parent
    return bool(
        parent is ancestor
        or (isinstance(parent, Tag) and ancestor in parent.parents)
    )


def enclosing_anchor(node: NavigableString) -> Tag | None:
    parent = node.parent
    if isinstance(parent, Tag) and parent.name == "a":
        return parent
    if isinstance(parent, Tag):
        return parent.find_parent("a")
    return None


def skip_html_text_node(
    node: NavigableString,
    title_anchor: Tag,
) -> bool:
    if isinstance(node, Comment) or is_inside(node, title_anchor):
        return True
    text = normalize_spaces(str(node))
    if not text:
        return True
    parent_anchor = enclosing_anchor(node)
    if parent_anchor is None:
        return False
    href = anchor_href(parent_anchor).lower()
    link_text = normalize_spaces(
        parent_anchor.get_text(" ", strip=True)
    ).lower()
    if link_text in SHARE_TEXT or link_text in RESOURCE_TEXT:
        return True
    return any(marker in href for marker in SHARE_HREF_MARKERS)


def html_fragments_between(
    anchor: Tag,
    next_anchor: Tag | None,
) -> list[str]:
    fragments: list[str] = []
    for element in anchor.next_elements:
        if next_anchor is not None and element is next_anchor:
            break
        if (
            isinstance(element, Tag)
            and element is not anchor
            and is_primary_result_anchor(element)
        ):
            break
        if not isinstance(element, NavigableString):
            continue
        if skip_html_text_node(element, anchor):
            continue
        text = normalize_spaces(str(element))
        if text.lower() in RESOURCE_TEXT:
            continue
        if fragments and fragments[-1] == text:
            continue
        fragments.append(text)
    return fragments


def html_blocks(
    body: str,
) -> list[tuple[str, str, str | None, str | None]]:
    soup = BeautifulSoup(body, "html.parser")
    anchors = [
        anchor
        for anchor in soup.find_all("a", href=True)
        if is_primary_result_anchor(anchor)
    ]
    blocks: list[tuple[str, str, str | None, str | None]] = []
    for index, anchor in enumerate(anchors):
        next_anchor = anchors[index + 1] if index + 1 < len(anchors) else None
        fragments = html_fragments_between(anchor, next_anchor)
        metadata, snippet = split_metadata_and_snippet(fragments)
        blocks.append(
            (
                clean_title(anchor.get_text(" ", strip=True)),
                anchor_href(anchor),
                metadata,
                snippet,
            )
        )
    return blocks


def quality_warnings(
    title: str,
    metadata: str | None,
    primary_url: str | None,
) -> list[str]:
    warnings: list[str] = []
    if any(pattern.search(title) for pattern in DOCUMENT_COVER_PATTERNS):
        warnings.append("possible_document_cover_title")
    if re.search(r"(?:…|\.\.\.)\s*$", title):
        warnings.append("truncated_title")

    path = urlparse(primary_url or "").path.lower()
    if not metadata and (
        path.endswith((".pdf", ".txt", ".doc", ".docx"))
        or path.endswith("/document")
        or "/bitstreams/" in path
    ):
        warnings.append("direct_document_without_metadata")
    return warnings


def make_candidate(
    *,
    context: SourceContext,
    position: int,
    title: str,
    href: str,
    metadata: str | None,
    snippet: str | None,
    parser_strategy: str,
    extracted_at: str,
    alert_name: str | None,
) -> dict:
    title = clean_title(title)
    normalized = normalize_title(title)
    primary_url = unwrap_scholar_url(href)
    authors, venue, year = parse_metadata_line(metadata)

    warnings: list[str] = []
    if not metadata:
        warnings.append("missing_metadata_line")
    if not venue:
        warnings.append("missing_venue")
    if not year:
        warnings.append("missing_year")

    review_warnings = quality_warnings(title, metadata, primary_url)
    warnings.extend(review_warnings)
    parse_state = (
        "manual_review"
        if review_warnings
        else ("partial" if warnings else "complete")
    )

    candidate_id = hashlib.sha256(
        f"{context.message_id}:{position}".encode()
    ).hexdigest()[:24]
    fingerprint = hashlib.sha256(
        f"{normalized}|{year or ''}|{venue or ''}".encode()
    ).hexdigest()

    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "source": {
            "source_type": "google_scholar_email",
            "message_id": context.message_id,
            "thread_id": context.thread_id,
            "received_at": context.received_at,
            "sender": context.sender,
            "subject": context.subject,
            "alert_name": alert_name,
            "authentication": {
                "spf": context.spf,
                "dkim": context.dkim,
            },
        },
        "position_in_message": position,
        "title": title,
        "normalized_title": normalized,
        "authors": authors,
        "raw_metadata_line": metadata,
        "venue": {
            "raw": venue,
            "normalized": venue,
            "verification_status": "raw_email" if venue else "missing",
        },
        "year": year,
        "snippet": snippet,
        "identifiers": extract_identifiers(
            title,
            metadata,
            snippet,
            primary_url,
        ),
        "links": {
            "primary_url": primary_url,
            "auxiliary_urls": [],
        },
        "parse_status": {
            "state": parse_state,
            "warnings": warnings,
            "errors": [],
            "parser_strategy": parser_strategy,
        },
        "content_fingerprint": fingerprint,
        "extracted_at": extracted_at,
    }


def parse_alert_body(
    body: str,
    context: SourceContext,
    *,
    content_type: str = "auto",
    extracted_at: str | None = None,
) -> list[dict]:
    extracted_at = extracted_at or utc_now()
    alert_name = extract_alert_name(context.subject, body)
    looks_html = bool(
        re.search(r"<(?:html|body|table|div|a)\b", body, re.I)
    )
    if content_type == "html" or (content_type == "auto" and looks_html):
        blocks = html_blocks(body)
        strategy = "html_nearest_ancestor"
    else:
        blocks = markdown_blocks(body)
        strategy = "plain_text_blocks"

    return [
        make_candidate(
            context=context,
            position=index,
            title=title,
            href=href,
            metadata=metadata,
            snippet=snippet,
            parser_strategy=strategy,
            extracted_at=extracted_at,
            alert_name=alert_name,
        )
        for index, (title, href, metadata, snippet) in enumerate(blocks)
        if title and normalize_title(title)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a Google Scholar alert body"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--message-id", required=True)
    parser.add_argument("--received-at", required=True)
    parser.add_argument(
        "--sender",
        default="scholaralerts-noreply@google.com",
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--content-type",
        choices=["auto", "html", "text"],
        default="auto",
    )
    args = parser.parse_args()

    context = SourceContext(
        message_id=args.message_id,
        received_at=args.received_at,
        sender=args.sender,
        subject=args.subject,
    )
    records = parse_alert_body(
        args.input.read_text(encoding="utf-8"),
        context,
        content_type=args.content_type,
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

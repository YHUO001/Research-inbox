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

from bs4 import BeautifulSoup, Tag

SCHOLAR_RESULT_URL = re.compile(r"https?://scholar\.google\.[^/]+/scholar_url\?", re.I)
MARKDOWN_LINK = re.compile(
    r"^\[(?P<title>.+?)\]\((?P<href>https?://scholar\.google\.[^/]+/scholar_url\?[^\n]+)\)\s*$",
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


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).lower().strip()
    text = re.sub(r"^\s*(?:19|20)\d{2}\s*[-_:]\s*", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def unwrap_scholar_url(href: str) -> str | None:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    target = query.get("url", [None])[0]
    if target:
        return sanitize_url(unquote(target))
    return sanitize_url(href)


def sanitize_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    tracking = {"utm_source", "utm_medium", "utm_campaign", "ved", "usg"}
    kept = []
    for pair in parsed.query.split("&") if parsed.query else []:
        key = pair.split("=", 1)[0]
        if key not in tracking:
            kept.append(pair)
    return urlunparse(parsed._replace(query="&".join(kept), fragment=""))


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


def extract_identifiers(title: str, metadata: str | None, snippet: str | None, url: str | None) -> dict:
    fields = [
        ("title", title),
        ("metadata_line", metadata or ""),
        ("snippet", snippet or ""),
        ("url", url or ""),
    ]

    doi_value = doi_source = None
    arxiv_value = arxiv_source = None
    pmid_value = pmid_source = None

    for source, text in fields:
        if not doi_value and (match := DOI_RE.search(text)):
            doi_value, doi_source = clean_doi(match.group(0)), source
        if not arxiv_value and (match := ARXIV_RE.search(text)):
            arxiv_value, arxiv_source = match.group(1), source
        if not pmid_value and (match := PMID_RE.search(text)):
            pmid_value, pmid_source = match.group(1), source

    return {
        "doi": identifier_evidence(doi_value, doi_source),
        "arxiv_id": identifier_evidence(arxiv_value, arxiv_source),
        "pmid": identifier_evidence(pmid_value, pmid_source),
    }


def parse_metadata_line(metadata: str | None) -> tuple[list[dict], str | None, int | None]:
    if not metadata:
        return [], None, None

    raw = re.sub(r"\s+", " ", metadata).strip()
    left, separator, right = raw.rpartition(" - ")
    if not separator:
        left, right = "", raw

    year_matches = YEAR_RE.findall(right)
    year = int(year_matches[-1]) if year_matches else None
    venue = right
    if year:
        venue = re.sub(rf",?\s*{year}\s*.*$", "", venue).strip(" ,-…")
    venue = venue or None

    authors: list[dict] = []
    if left:
        author_text = left.replace("…", "").strip()
        for name in [part.strip() for part in author_text.split(",")]:
            if name:
                authors.append(
                    {
                        "name": name,
                        "orcid": None,
                        "verification_status": "raw_email",
                    }
                )
    return authors, venue, year


def extract_alert_name(subject: str, body: str) -> str | None:
    quoted = re.search(r'["“](.+?)["”]\s*-\s*', subject)
    if quoted:
        return quoted.group(1).strip()
    footer = re.search(r"关注了\s*\[?\[?[\"“](.+?)[\"”]\]?\]?", body)
    return footer.group(1).strip() if footer else None


def compact_snippet(lines: Iterable[str]) -> str | None:
    cleaned = []
    for line in lines:
        text = re.sub(r"\s+", " ", line).strip()
        if not text:
            continue
        if any(marker.lower() in text.lower() for marker in FOOTER_MARKERS):
            break
        cleaned.append(text)
    if not cleaned:
        return None
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def markdown_blocks(body: str) -> list[tuple[str, str, str | None, str | None]]:
    matches = list(MARKDOWN_LINK.finditer(body))
    blocks: list[tuple[str, str, str | None, str | None]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        chunk_lines = [line.strip() for line in body[start:end].splitlines()]
        useful = [line for line in chunk_lines if line]
        if not useful:
            metadata = snippet = None
        else:
            metadata = useful[0]
            snippet = compact_snippet(useful[1:])
        blocks.append((match.group("title").strip(), match.group("href"), metadata, snippet))
    return blocks


def nearest_single_result_container(anchor: Tag) -> Tag:
    current: Tag = anchor
    best: Tag = anchor.parent if isinstance(anchor.parent, Tag) else anchor
    for _ in range(8):
        parent = current.parent
        if not isinstance(parent, Tag):
            break
        candidate_count = sum(
            1
            for link in parent.find_all("a", href=True)
            if SCHOLAR_RESULT_URL.search(str(link.get("href", "")))
        )
        if candidate_count == 1:
            best = parent
            current = parent
            continue
        break
    return best


def html_blocks(body: str) -> list[tuple[str, str, str | None, str | None]]:
    soup = BeautifulSoup(body, "html.parser")
    anchors = [
        anchor
        for anchor in soup.find_all("a", href=True)
        if SCHOLAR_RESULT_URL.search(str(anchor.get("href", "")))
    ]
    blocks: list[tuple[str, str, str | None, str | None]] = []
    for anchor in anchors:
        title = anchor.get_text(" ", strip=True)
        container = nearest_single_result_container(anchor)
        lines = [
            re.sub(r"\s+", " ", text).strip()
            for text in container.get_text("\n", strip=True).splitlines()
        ]
        lines = [line for line in lines if line and line != title and line.lower() not in SHARE_TEXT]
        metadata_index = next(
            (
                i
                for i, line in enumerate(lines)
                if " - " in line and YEAR_RE.search(line)
            ),
            None,
        )
        metadata = lines[metadata_index] if metadata_index is not None else None
        snippet_lines = lines[metadata_index + 1 :] if metadata_index is not None else lines
        snippet = compact_snippet(snippet_lines)
        blocks.append((title, str(anchor["href"]), metadata, snippet))
    return blocks


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
    title = unicodedata.normalize("NFKC", re.sub(r"\s+", " ", title)).strip()
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
    state = "complete" if not warnings else "partial"

    candidate_id = hashlib.sha256(f"{context.message_id}:{position}".encode()).hexdigest()[:24]
    fingerprint_input = f"{normalized}|{year or ''}|{venue or ''}"
    fingerprint = hashlib.sha256(fingerprint_input.encode()).hexdigest()

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
            "authentication": {"spf": context.spf, "dkim": context.dkim},
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
        "identifiers": extract_identifiers(title, metadata, snippet, primary_url),
        "links": {"primary_url": primary_url, "auxiliary_urls": []},
        "parse_status": {
            "state": state,
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

    looks_html = bool(re.search(r"<(?:html|body|table|div|a)\b", body, re.I))
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
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a Google Scholar alert body")
    parser.add_argument("input", type=Path)
    parser.add_argument("--message-id", required=True)
    parser.add_argument("--received-at", required=True)
    parser.add_argument("--sender", default="scholaralerts-noreply@google.com")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--content-type", choices=["auto", "html", "text"], default="auto")
    args = parser.parse_args()

    body = args.input.read_text(encoding="utf-8")
    context = SourceContext(
        message_id=args.message_id,
        received_at=args.received_at,
        sender=args.sender,
        subject=args.subject,
    )
    records = parse_alert_body(body, context, content_type=args.content_type)
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

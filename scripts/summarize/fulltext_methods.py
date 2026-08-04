from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader


METHOD_KEYWORDS = (
    "method",
    "methods",
    "methodology",
    "materials and methods",
    "experimental setup",
    "experimental methods",
    "experimental implementation",
    "experimental demonstration",
    "implementation",
    "design principle",
    "operating principle",
    "computing principle",
    "underlying physics",
    "system architecture",
    "network architecture",
    "computer architecture",
    "optical computer architecture",
    "optical setup",
    "hardware setup",
    "training",
    "training procedure",
    "fabrication",
    "device design",
    "model architecture",
)


@dataclass(frozen=True)
class MethodContext:
    candidate_id: str
    status: str
    source_url: str | None
    media_type: str | None
    section_headings: list[str]
    text: str
    error: str | None = None

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def content_sha256(self) -> str | None:
        if not self.text:
            return None
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def audit_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "source_url": self.source_url,
            "media_type": self.media_type,
            "section_headings": self.section_headings,
            "character_count": self.character_count,
            "content_sha256": self.content_sha256,
            "error": self.error,
            "text_persisted": False,
        }


def candidate_urls(source: dict[str, Any], limit: int = 3) -> list[str]:
    values: list[str] = []
    for value in (
        source.get("open_access_url"),
        source.get("landing_page"),
        f"https://doi.org/{source['doi']}" if source.get("doi") else None,
    ):
        text = str(value or "").strip()
        if text.startswith(("http://", "https://")) and text not in values:
            values.append(text)
    return values[: max(1, limit)]


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def is_method_heading(value: str) -> bool:
    lowered = normalize_space(value).lower()
    return any(keyword in lowered for keyword in METHOD_KEYWORDS)


def heading_level(tag: Tag) -> int:
    name = str(tag.name or "").lower()
    return int(name[1]) if len(name) == 2 and name.startswith("h") and name[1].isdigit() else 7


def unique_paragraphs(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_space(value)
        key = text.lower()
        if len(text) < 40 or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def derived_nature_pdf_urls(base_url: str) -> list[str]:
    parsed = urlsplit(base_url)
    if not parsed.netloc.lower().endswith("nature.com"):
        return []
    article_path = parsed.path.rstrip("/")
    if not re.fullmatch(r"/articles/[^/]+", article_path):
        return []
    article_url = urlunsplit((parsed.scheme or "https", parsed.netloc, article_path, "", ""))
    return [f"{article_url}.pdf", f"{article_url}_reference.pdf"]


def discover_pdf_urls(soup: BeautifulSoup, base_url: str) -> list[str]:
    values: list[str] = []
    for meta in soup.find_all("meta"):
        name = str(meta.get("name") or meta.get("property") or "").lower()
        if name in {"citation_pdf_url", "dc.identifier", "eprints.document_url"}:
            content = str(meta.get("content") or "")
            if content.lower().endswith(".pdf") or "pdf" in content.lower():
                values.append(urljoin(base_url, content))
    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        label = normalize_space(link.get_text(" ", strip=True)).lower()
        if href.lower().endswith(".pdf") or "pdf" in label:
            values.append(urljoin(base_url, href))
    values.extend(derived_nature_pdf_urls(base_url))
    return list(dict.fromkeys(value for value in values if value.startswith(("http://", "https://"))))


def method_heading_windows(text: str, *, maximum_characters: int) -> tuple[str, list[str]]:
    compact = re.sub(r"[ \t]+", " ", text or "")
    heading_hits: list[tuple[int, str]] = []
    offset = 0
    for raw_line in compact.splitlines(keepends=True):
        line = normalize_space(raw_line)
        looks_like_heading = (
            2 <= len(line) <= 180
            and len(line.split()) <= 24
            and not re.search(r"[.!?;:]$", line)
            and is_method_heading(line)
        )
        if looks_like_heading:
            heading_hits.append((offset, line))
        offset += len(raw_line)
    if not heading_hits:
        return "", []

    windows: list[tuple[int, int]] = []
    headings: list[str] = []
    for position, heading in heading_hits:
        start = max(0, position)
        end = min(len(compact), position + 3600)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
        headings.append(heading)

    output: list[str] = []
    used = 0
    for start, end in windows:
        remaining = maximum_characters - used
        if remaining <= 0:
            break
        value = normalize_space(compact[start:end])[:remaining]
        if value:
            output.append(value)
            used += len(value) + 2
    return "\n\n".join(output), list(dict.fromkeys(headings))


def extract_html_method_context(
    payload: bytes,
    *,
    source_url: str,
    maximum_characters: int,
) -> tuple[str, list[str], list[str]]:
    soup = BeautifulSoup(payload, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    paragraphs: list[str] = []
    headings: list[str] = []
    heading_tags = soup.find_all(re.compile(r"^h[1-6]$"))
    for heading in heading_tags:
        title = normalize_space(heading.get_text(" ", strip=True))
        if not is_method_heading(title):
            continue
        headings.append(title)
        level = heading_level(heading)
        section_values: list[str] = []
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and re.fullmatch(r"h[1-6]", str(sibling.name or "")):
                if heading_level(sibling) <= level:
                    break
            if isinstance(sibling, Tag):
                for item in sibling.find_all(["p", "li"], recursive=True):
                    section_values.append(item.get_text(" ", strip=True))
                if sibling.name in {"p", "li"}:
                    section_values.append(sibling.get_text(" ", strip=True))

        if not unique_paragraphs(section_values):
            section = heading.find_parent("section")
            if section is not None:
                section_values.extend(
                    item.get_text(" ", strip=True)
                    for item in section.find_all(["p", "li"], recursive=True)
                )
        paragraphs.extend(unique_paragraphs(section_values))

    if not paragraphs:
        for section in soup.find_all(["section", "div"]):
            marker = " ".join(
                str(value or "")
                for value in (section.get("id"), " ".join(section.get("class") or []))
            )
            if not is_method_heading(marker):
                continue
            heading = section.find(re.compile(r"^h[1-6]$"))
            if heading:
                headings.append(normalize_space(heading.get_text(" ", strip=True)))
            paragraphs.extend(
                unique_paragraphs(
                    [item.get_text(" ", strip=True) for item in section.find_all(["p", "li"])]
                )
            )

    pdf_urls = discover_pdf_urls(soup, source_url)
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
        return "\n\n".join(blocks), list(dict.fromkeys(headings)), pdf_urls

    article_root = soup.find("article") or soup.find("main") or soup
    fallback_text, fallback_headings = method_heading_windows(
        article_root.get_text("\n", strip=True),
        maximum_characters=maximum_characters,
    )
    return fallback_text, list(dict.fromkeys(headings + fallback_headings)), pdf_urls


def text_windows(text: str, *, maximum_characters: int) -> tuple[str, list[str]]:
    heading_text, headings = method_heading_windows(
        text,
        maximum_characters=maximum_characters,
    )
    if heading_text:
        return heading_text, headings

    compact = re.sub(r"[ \t]+", " ", text or "")
    lowered = compact.lower()
    hits: list[tuple[int, str]] = []
    for keyword in METHOD_KEYWORDS:
        for match in re.finditer(re.escape(keyword), lowered):
            hits.append((match.start(), keyword))
    if not hits:
        return "", []
    hits.sort()
    windows: list[tuple[int, int]] = []
    labels: list[str] = []
    radius = 1800
    for position, keyword in hits:
        start = max(0, position - radius)
        end = min(len(compact), position + radius)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
        labels.append(keyword)
    output: list[str] = []
    used = 0
    for start, end in windows:
        remaining = maximum_characters - used
        if remaining <= 0:
            break
        value = normalize_space(compact[start:end])[:remaining]
        if value:
            output.append(value)
            used += len(value) + 2
    return "\n\n".join(output), list(dict.fromkeys(labels))


def extract_pdf_method_context(
    payload: bytes,
    *,
    maximum_characters: int,
) -> tuple[str, list[str]]:
    reader = PdfReader(io.BytesIO(payload))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return text_windows(text, maximum_characters=maximum_characters)


def default_fetch(url: str, *, timeout_seconds: float, maximum_bytes: int) -> tuple[bytes, str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "ResearchInbox/0.5 (+https://github.com/YHUO001/Research-inbox)",
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ValueError("Full-text response exceeded the configured byte limit")
        media_type = str(response.headers.get_content_type() or "application/octet-stream")
        return payload, media_type.lower(), str(response.geturl())


def collect_method_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    fetcher: Callable[..., tuple[bytes, str, str]] = default_fetch,
) -> MethodContext:
    candidate_id = str(source.get("candidate_id") or source.get("id") or "unknown")
    maximum_characters = int(config.get("maximum_method_characters") or 12000)
    maximum_bytes = int(config.get("maximum_download_bytes") or 15_000_000)
    timeout_seconds = float(config.get("timeout_seconds") or 30)
    urls = candidate_urls(source, int(config.get("candidate_url_limit") or 3))
    if not urls:
        return MethodContext(candidate_id, "not_available", None, None, [], "")

    errors: list[str] = []
    queued = list(urls)
    attempted: set[str] = set()
    while queued:
        url = queued.pop(0)
        if url in attempted:
            continue
        attempted.add(url)
        try:
            payload, media_type, final_url = fetcher(
                url,
                timeout_seconds=timeout_seconds,
                maximum_bytes=maximum_bytes,
            )
            if media_type in {"text/html", "application/xhtml+xml"}:
                text, headings, pdf_urls = extract_html_method_context(
                    payload,
                    source_url=final_url,
                    maximum_characters=maximum_characters,
                )
                if text:
                    return MethodContext(
                        candidate_id,
                        "used",
                        final_url,
                        media_type,
                        headings,
                        text,
                    )
                for pdf_url in reversed(pdf_urls[:2]):
                    if pdf_url not in attempted:
                        queued.insert(0, pdf_url)
                errors.append(f"{final_url}: no method-oriented HTML section found")
            elif media_type == "application/pdf" or final_url.lower().endswith(".pdf"):
                text, headings = extract_pdf_method_context(
                    payload,
                    maximum_characters=maximum_characters,
                )
                if text:
                    return MethodContext(
                        candidate_id,
                        "used",
                        final_url,
                        "application/pdf",
                        headings,
                        text,
                    )
                errors.append(f"{final_url}: no method-oriented PDF text found")
            else:
                errors.append(f"{final_url}: unsupported media type {media_type}")
        except Exception as error:  # Network and parser failures are non-fatal fallbacks.
            errors.append(f"{url}: {type(error).__name__}: {str(error)[:240]}")

    return MethodContext(
        candidate_id,
        "not_available",
        urls[0],
        None,
        [],
        "",
        "; ".join(errors)[:1000] or None,
    )
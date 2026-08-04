from __future__ import annotations

import hashlib
import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
PROVIDER_FIELDS = (
    "title",
    "doi",
    "openalex_id",
    "arxiv_id",
    "venue",
    "publication_date",
    "year",
    "abstract",
    "landing_page",
    "open_access_url",
    "cited_by_count",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", value or "").lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    return text.rstrip(".,;:)]}").lower() or None


def title_similarity(left: str | None, right: str | None) -> float:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def surname(value: str | None) -> str | None:
    words = normalize_text(value).split()
    return words[-1] if words else None


def candidate_doi(candidate: dict[str, Any]) -> str | None:
    return normalize_doi(
        ((candidate.get("identifiers") or {}).get("doi") or {}).get("value")
    )


def candidate_first_author(candidate: dict[str, Any]) -> str | None:
    authors = candidate.get("authors") or []
    if not authors:
        return None
    first = authors[0]
    return str(first.get("name") or "") if isinstance(first, dict) else str(first)


def stable_key(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class HttpRequestError(RuntimeError):
    def __init__(self, reason: str, status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


class JsonHttpClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float,
        max_attempts: int,
        min_interval_seconds: float,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.opener = opener
        self.sleeper = sleeper
        self.monotonic = monotonic
        self._last_request_at: float | None = None

    def get_json(
        self,
        base_url: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = urlencode(
            {
                key: value
                for key, value in (params or {}).items()
                if value is not None and value != ""
            },
            doseq=True,
        )
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

        last_error: HttpRequestError | None = None
        for attempt in range(self.max_attempts):
            if self._last_request_at is not None:
                elapsed = self.monotonic() - self._last_request_at
                delay = self.min_interval_seconds - elapsed
                if delay > 0:
                    self.sleeper(delay)
            request = Request(url, headers=headers, method="GET")
            try:
                self._last_request_at = self.monotonic()
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    payload = response.read().decode("utf-8")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise HttpRequestError("invalid_json_shape")
                return value
            except HTTPError as error:
                reason = f"http_{error.code}"
                last_error = HttpRequestError(reason, error.code)
                retryable = error.code in RETRYABLE_STATUS
            except (URLError, TimeoutError, OSError):
                last_error = HttpRequestError("network_error")
                retryable = True
            except (UnicodeDecodeError, json.JSONDecodeError):
                last_error = HttpRequestError("invalid_json")
                retryable = False

            if not retryable or attempt + 1 >= self.max_attempts:
                break
            self.sleeper(float(2**attempt))

        raise last_error or HttpRequestError("unknown_http_error")


class NormalizedCache:
    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self.now = now
        value = data if isinstance(data, dict) else {}
        entries = value.get("entries")
        self.entries: dict[str, dict[str, Any]] = (
            entries if isinstance(entries, dict) else {}
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        now: Callable[[], datetime] = utc_now,
    ) -> "NormalizedCache":
        if not path.exists():
            return cls(now=now)
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(value, now=now)

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self.entries.get(key)
        if not isinstance(entry, dict):
            return None
        expires_at = entry.get("expires_at")
        if not isinstance(expires_at, str) or parse_datetime(expires_at) <= self.now():
            return None
        return entry

    def put(
        self,
        key: str,
        *,
        status: str,
        record: dict[str, Any] | None,
        ttl_days: int,
    ) -> dict[str, Any]:
        retrieved_at = self.now()
        entry = {
            "status": status,
            "record": record,
            "retrieved_at": isoformat(retrieved_at),
            "expires_at": isoformat(retrieved_at + timedelta(days=ttl_days)),
        }
        self.entries[key] = entry
        return entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "entries": dict(sorted(self.entries.items())),
        }


@dataclass(frozen=True)
class Attempt:
    provider: str
    method: str
    status: str
    confidence: str
    cache_hit: bool
    retrieved_at: str | None
    record: dict[str, Any] | None = None
    reason: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "method": self.method,
            "status": self.status,
            "confidence": self.confidence,
            "cache_hit": self.cache_hit,
            "retrieved_at": self.retrieved_at,
            "reason": self.reason,
        }


def crossref_date(message: dict[str, Any]) -> tuple[str | None, int | None]:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = ((message.get(key) or {}).get("date-parts") or [])
        if not parts or not parts[0]:
            continue
        values = [int(value) for value in parts[0][:3]]
        year = values[0]
        if len(values) == 1:
            return f"{year:04d}", year
        if len(values) == 2:
            return f"{year:04d}-{values[1]:02d}", year
        return f"{year:04d}-{values[1]:02d}-{values[2]:02d}", year
    return None, None


def clean_abstract(value: str | None, maximum: int) -> str | None:
    if not value:
        return None
    text = BeautifulSoup(html.unescape(value), "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maximum] or None


def normalize_crossref(message: dict[str, Any], maximum_abstract: int) -> dict[str, Any]:
    date, year = crossref_date(message)
    authors = []
    for author in message.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part
            for part in [str(author.get("given") or ""), str(author.get("family") or "")]
            if part
        ).strip()
        if name:
            authors.append({"name": name, "orcid": author.get("ORCID")})
    titles = message.get("title") or []
    containers = message.get("container-title") or []
    doi = normalize_doi(message.get("DOI"))
    landing = str(message.get("URL") or "") or (
        f"https://doi.org/{doi}" if doi else None
    )
    return {
        "provider": "crossref",
        "provider_id": doi,
        "title": str(titles[0]) if titles else None,
        "authors": authors,
        "doi": doi,
        "openalex_id": None,
        "arxiv_id": None,
        "venue": str(containers[0]) if containers else None,
        "publication_date": date,
        "year": year,
        "abstract": clean_abstract(message.get("abstract"), maximum_abstract),
        "landing_page": landing,
        "open_access_url": None,
        "cited_by_count": message.get("is-referenced-by-count"),
    }


def reconstruct_abstract(index: dict[str, Any] | None, maximum: int) -> str | None:
    if not isinstance(index, dict) or not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, values in index.items():
        if not isinstance(values, list):
            continue
        positions.extend((int(position), str(word)) for position in values)
    if not positions:
        return None
    text = " ".join(word for _, word in sorted(positions))
    return text[:maximum] or None


def normalize_openalex(work: dict[str, Any], maximum_abstract: int) -> dict[str, Any]:
    authors = []
    for authorship in work.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = str(author.get("display_name") or "")
        if name:
            authors.append({"name": name, "orcid": author.get("orcid")})
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    best_oa = work.get("best_oa_location") or {}
    ids = work.get("ids") or {}
    doi = normalize_doi(work.get("doi") or ids.get("doi"))
    openalex_id = str(work.get("id") or ids.get("openalex") or "") or None
    return {
        "provider": "openalex",
        "provider_id": openalex_id,
        "title": work.get("title") or work.get("display_name"),
        "authors": authors,
        "doi": doi,
        "openalex_id": openalex_id,
        "arxiv_id": ids.get("arxiv"),
        "venue": source.get("display_name"),
        "publication_date": work.get("publication_date"),
        "year": work.get("publication_year"),
        "abstract": reconstruct_abstract(
            work.get("abstract_inverted_index"), maximum_abstract
        ),
        "landing_page": primary.get("landing_page_url")
        or (f"https://doi.org/{doi}" if doi else None),
        "open_access_url": best_oa.get("landing_page_url")
        or best_oa.get("pdf_url")
        or (work.get("open_access") or {}).get("oa_url"),
        "cited_by_count": work.get("cited_by_count"),
    }


def conservative_match(
    candidate: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    minimum: float,
    ambiguity_margin: float,
    maximum_year_difference: int,
    require_confirmation: bool,
) -> dict[str, Any] | None:
    candidate_title = str(candidate.get("title") or "")
    candidate_year = candidate.get("year")
    candidate_author = surname(candidate_first_author(candidate))
    accepted: list[tuple[float, float, dict[str, Any]]] = []
    for record in records:
        similarity = title_similarity(candidate_title, record.get("title"))
        if similarity < minimum:
            continue
        record_year = record.get("year")
        year_match = (
            candidate_year is not None
            and record_year is not None
            and abs(int(candidate_year) - int(record_year)) <= maximum_year_difference
        )
        if candidate_year is not None and record_year is not None and not year_match:
            continue
        record_authors = record.get("authors") or []
        record_author = surname(record_authors[0].get("name")) if record_authors else None
        author_match = bool(
            candidate_author and record_author and candidate_author == record_author
        )
        if candidate_author and record_author and not author_match:
            continue
        if require_confirmation and not (year_match or author_match):
            continue
        corroboration = (0.02 if year_match else 0.0) + (
            0.02 if author_match else 0.0
        )
        accepted.append((similarity + corroboration, similarity, record))
    accepted.sort(key=lambda item: item[0], reverse=True)
    if not accepted:
        return None
    if len(accepted) > 1 and accepted[0][1] - accepted[1][1] < ambiguity_margin:
        return None
    return accepted[0][2]


class CrossrefProvider:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        client: JsonHttpClient,
        cache: NormalizedCache,
        contact_email: str | None,
        maximum_abstract: int,
        exact_ttl_days: int,
        search_ttl_days: int,
        matching: dict[str, Any],
    ) -> None:
        self.config = config
        self.client = client
        self.cache = cache
        self.contact_email = contact_email
        self.maximum_abstract = maximum_abstract
        self.exact_ttl_days = exact_ttl_days
        self.search_ttl_days = search_ttl_days
        self.matching = matching

    def _cached_or_fetch(
        self,
        key: str,
        *,
        method: str,
        ttl_days: int,
        fetch: Callable[[], dict[str, Any] | None],
        confidence: str,
    ) -> Attempt:
        cached = self.cache.get(key)
        if cached:
            return Attempt(
                "crossref",
                method,
                str(cached["status"]),
                confidence if cached["status"] == "found" else "unresolved",
                True,
                str(cached["retrieved_at"]),
                cached.get("record"),
                None if cached["status"] == "found" else "not_found",
            )
        try:
            record = fetch()
        except HttpRequestError as error:
            if error.status == 404:
                record = None
            else:
                return Attempt(
                    "crossref",
                    method,
                    "error",
                    "unresolved",
                    False,
                    None,
                    None,
                    error.reason,
                )
        entry = self.cache.put(
            key,
            status="found" if record else "not_found",
            record=record,
            ttl_days=ttl_days,
        )
        return Attempt(
            "crossref",
            method,
            str(entry["status"]),
            confidence if record else "unresolved",
            False,
            str(entry["retrieved_at"]),
            record,
            None if record else "not_found",
        )

    def by_doi(self, doi: str) -> Attempt:
        key = f"crossref:doi:{doi}"

        def fetch() -> dict[str, Any] | None:
            payload = self.client.get_json(
                self.config["base_url"],
                f"works/{quote(doi, safe='')}",
                params={"mailto": self.contact_email},
            )
            message = payload.get("message")
            return (
                normalize_crossref(message, self.maximum_abstract)
                if isinstance(message, dict)
                else None
            )

        return self._cached_or_fetch(
            key,
            method="doi",
            ttl_days=self.exact_ttl_days,
            fetch=fetch,
            confidence="exact",
        )

    def by_title(self, candidate: dict[str, Any]) -> Attempt:
        title = str(candidate.get("title") or "")
        author = candidate_first_author(candidate) or ""
        year = str(candidate.get("year") or "")
        key = (
            "crossref:title:"
            f"{stable_key(normalize_text(title), normalize_text(author), year)}"
        )

        def fetch() -> dict[str, Any] | None:
            bibliographic = " ".join(part for part in (title, author, year) if part)
            payload = self.client.get_json(
                self.config["base_url"],
                "works",
                params={
                    "query.bibliographic": bibliographic,
                    "rows": self.matching["maximum_search_results_per_provider"],
                    "mailto": self.contact_email,
                },
            )
            items = ((payload.get("message") or {}).get("items") or [])
            records = [
                normalize_crossref(item, self.maximum_abstract)
                for item in items
                if isinstance(item, dict)
            ]
            return conservative_match(
                candidate,
                records,
                minimum=float(self.matching["title_similarity_minimum"]),
                ambiguity_margin=float(self.matching["ambiguity_margin"]),
                maximum_year_difference=int(
                    self.matching["maximum_year_difference"]
                ),
                require_confirmation=bool(
                    self.matching["require_year_or_first_author_confirmation"]
                ),
            )

        return self._cached_or_fetch(
            key,
            method="title_year_author",
            ttl_days=self.search_ttl_days,
            fetch=fetch,
            confidence="high",
        )


class OpenAlexProvider:
    SELECT = (
        "id,doi,title,publication_year,publication_date,authorships,"
        "primary_location,best_oa_location,open_access,abstract_inverted_index,"
        "cited_by_count,ids"
    )

    def __init__(
        self,
        config: dict[str, Any],
        *,
        client: JsonHttpClient,
        cache: NormalizedCache,
        api_key: str | None,
        maximum_abstract: int,
        exact_ttl_days: int,
        search_ttl_days: int,
        matching: dict[str, Any],
    ) -> None:
        self.config = config
        self.client = client
        self.cache = cache
        self.api_key = api_key
        self.maximum_abstract = maximum_abstract
        self.exact_ttl_days = exact_ttl_days
        self.search_ttl_days = search_ttl_days
        self.matching = matching

    def _missing_key(self, method: str) -> Attempt:
        return Attempt(
            "openalex",
            method,
            "skipped",
            "unresolved",
            False,
            None,
            None,
            "missing_api_key",
        )

    def _cached_or_fetch(
        self,
        key: str,
        *,
        method: str,
        ttl_days: int,
        fetch: Callable[[], dict[str, Any] | None],
        confidence: str,
    ) -> Attempt:
        if not self.api_key:
            return self._missing_key(method)
        cached = self.cache.get(key)
        if cached:
            return Attempt(
                "openalex",
                method,
                str(cached["status"]),
                confidence if cached["status"] == "found" else "unresolved",
                True,
                str(cached["retrieved_at"]),
                cached.get("record"),
                None if cached["status"] == "found" else "not_found",
            )
        try:
            record = fetch()
        except HttpRequestError as error:
            if error.status == 404:
                record = None
            else:
                return Attempt(
                    "openalex",
                    method,
                    "error",
                    "unresolved",
                    False,
                    None,
                    None,
                    error.reason,
                )
        entry = self.cache.put(
            key,
            status="found" if record else "not_found",
            record=record,
            ttl_days=ttl_days,
        )
        return Attempt(
            "openalex",
            method,
            str(entry["status"]),
            confidence if record else "unresolved",
            False,
            str(entry["retrieved_at"]),
            record,
            None if record else "not_found",
        )

    def by_doi(self, doi: str) -> Attempt:
        key = f"openalex:doi:{doi}"
        external_id = quote(f"https://doi.org/{doi}", safe=":/")

        def fetch() -> dict[str, Any] | None:
            payload = self.client.get_json(
                self.config["base_url"],
                f"works/{external_id}",
                params={"api_key": self.api_key, "select": self.SELECT},
            )
            return normalize_openalex(payload, self.maximum_abstract)

        return self._cached_or_fetch(
            key,
            method="doi",
            ttl_days=self.exact_ttl_days,
            fetch=fetch,
            confidence="exact",
        )

    def by_title(self, candidate: dict[str, Any]) -> Attempt:
        title = str(candidate.get("title") or "")
        author = candidate_first_author(candidate) or ""
        year = str(candidate.get("year") or "")
        key = (
            "openalex:title:"
            f"{stable_key(normalize_text(title), normalize_text(author), year)}"
        )

        def fetch() -> dict[str, Any] | None:
            payload = self.client.get_json(
                self.config["base_url"],
                "works",
                params={
                    "search": title,
                    "per_page": self.matching[
                        "maximum_search_results_per_provider"
                    ],
                    "select": self.SELECT,
                    "api_key": self.api_key,
                },
            )
            results = payload.get("results") or []
            records = [
                normalize_openalex(item, self.maximum_abstract)
                for item in results
                if isinstance(item, dict)
            ]
            return conservative_match(
                candidate,
                records,
                minimum=float(self.matching["title_similarity_minimum"]),
                ambiguity_margin=float(self.matching["ambiguity_margin"]),
                maximum_year_difference=int(
                    self.matching["maximum_year_difference"]
                ),
                require_confirmation=bool(
                    self.matching["require_year_or_first_author_confirmation"]
                ),
            )

        return self._cached_or_fetch(
            key,
            method="title_year_author",
            ttl_days=self.search_ttl_days,
            fetch=fetch,
            confidence="high",
        )


def source_scalar(
    value: Any,
    *,
    source: str,
    confidence: str,
    retrieved_at: str | None,
) -> dict[str, Any]:
    return {
        "value": value,
        "source": source,
        "confidence": confidence,
        "retrieved_at": retrieved_at,
    }


def unavailable_scalar() -> dict[str, Any]:
    return source_scalar(
        None,
        source="unavailable",
        confidence="unresolved",
        retrieved_at=None,
    )


def original_field(candidate: dict[str, Any], name: str) -> Any:
    if name == "doi":
        return candidate_doi(candidate)
    if name == "arxiv_id":
        return ((candidate.get("identifiers") or {}).get("arxiv_id") or {}).get(
            "value"
        )
    if name == "venue":
        venue = candidate.get("venue") or {}
        return venue.get("normalized") or venue.get("raw")
    if name == "publication_date":
        return str(candidate.get("year")) if candidate.get("year") else None
    if name == "landing_page":
        return (candidate.get("links") or {}).get("primary_url")
    if name in {"openalex_id", "open_access_url", "cited_by_count"}:
        return None
    return candidate.get(name)


def choose_record(
    records: list[tuple[Attempt, dict[str, Any]]],
    field: str,
) -> tuple[Attempt, dict[str, Any]] | None:
    preference = {"openalex": 0, "crossref": 1}
    if field in {
        "title",
        "doi",
        "venue",
        "publication_date",
        "year",
        "landing_page",
    }:
        preference = {"crossref": 0, "openalex": 1}
    candidates = [
        item
        for item in records
        if item[1].get(field) is not None and item[1].get(field) != []
    ]
    candidates.sort(
        key=lambda item: (
            0 if item[0].confidence == "exact" else 1,
            preference.get(item[0].provider, 9),
        )
    )
    return candidates[0] if candidates else None


def merge_enrichment(
    candidate: dict[str, Any],
    attempts: list[Attempt],
) -> dict[str, Any]:
    records = [
        (attempt, attempt.record)
        for attempt in attempts
        if attempt.status == "found" and isinstance(attempt.record, dict)
    ]
    match_status = (
        "exact"
        if any(
            attempt.confidence == "exact" and attempt.status == "found"
            for attempt in attempts
        )
        else (
            "high"
            if any(
                attempt.confidence == "high" and attempt.status == "found"
                for attempt in attempts
            )
            else "unresolved"
        )
    )
    fields: dict[str, Any] = {}
    for field in PROVIDER_FIELDS:
        choice = choose_record(records, field)
        if choice:
            attempt, record = choice
            fields[field] = source_scalar(
                record.get(field),
                source=attempt.provider,
                confidence=attempt.confidence,
                retrieved_at=attempt.retrieved_at,
            )
        else:
            original = original_field(candidate, field)
            fields[field] = (
                source_scalar(
                    original,
                    source="scholar_email",
                    confidence="raw",
                    retrieved_at=str(candidate.get("extracted_at") or "") or None,
                )
                if original is not None
                else unavailable_scalar()
            )

    author_choice = next(
        (
            (attempt, record)
            for attempt, record in sorted(
                records,
                key=lambda item: (
                    0 if item[0].confidence == "exact" else 1,
                    0 if item[0].provider == "crossref" else 1,
                ),
            )
            if record.get("authors")
        ),
        None,
    )
    if author_choice:
        attempt, record = author_choice
        authors = [
            {
                "name": author["name"],
                "orcid": author.get("orcid"),
                "source": attempt.provider,
                "confidence": attempt.confidence,
                "retrieved_at": attempt.retrieved_at,
            }
            for author in record["authors"]
        ]
    else:
        authors = [
            {
                "name": str(author.get("name") or ""),
                "orcid": author.get("orcid"),
                "source": "scholar_email",
                "confidence": "raw",
                "retrieved_at": str(candidate.get("extracted_at") or "") or None,
            }
            for author in candidate.get("authors") or []
            if isinstance(author, dict) and author.get("name")
        ]

    timestamps = [attempt.retrieved_at for attempt in attempts if attempt.retrieved_at]
    enriched_at = max(timestamps) if timestamps else str(candidate.get("extracted_at"))
    return {
        "schema_version": 1,
        "candidate_id": str(candidate["candidate_id"]),
        "source_content_fingerprint": candidate.get("content_fingerprint"),
        "match": {
            "status": match_status,
            "providers": sorted(
                {
                    attempt.provider
                    for attempt in attempts
                    if attempt.status == "found"
                }
            ),
            "methods": sorted(
                {
                    attempt.method
                    for attempt in attempts
                    if attempt.status == "found"
                }
            ),
        },
        "fields": fields,
        "authors": authors,
        "provider_attempts": [attempt.public() for attempt in attempts],
        "enriched_at": enriched_at,
    }


def enrich_candidate(
    candidate: dict[str, Any],
    *,
    crossref: CrossrefProvider,
    openalex: OpenAlexProvider,
) -> dict[str, Any]:
    attempts: list[Attempt] = []
    doi = candidate_doi(candidate)
    if doi:
        crossref_attempt = crossref.by_doi(doi)
        attempts.append(crossref_attempt)
        attempts.append(openalex.by_doi(doi))
        if not any(attempt.status == "found" for attempt in attempts):
            attempts.append(crossref.by_title(candidate))
            attempts.append(openalex.by_title(candidate))
    else:
        crossref_attempt = crossref.by_title(candidate)
        attempts.append(crossref_attempt)
        discovered_doi = (
            normalize_doi(crossref_attempt.record.get("doi"))
            if crossref_attempt.record
            else None
        )
        attempts.append(
            openalex.by_doi(discovered_doi)
            if discovered_doi
            else openalex.by_title(candidate)
        )
    return merge_enrichment(candidate, attempts)

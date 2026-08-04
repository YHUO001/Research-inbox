from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from scripts.classify.recognition import classify_candidate, load_yaml
from scripts.enrich.metadata import (
    HttpRequestError,
    JsonHttpClient,
    OpenAlexProvider,
    normalize_doi,
    normalize_openalex,
)
from scripts.ingest.scholar_parser import normalize_title


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    project_id: str
    text: str


class RegistryIdentityIndex:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.work_ids: set[str] = set()
        self.dois: set[str] = set()
        self.title_years: set[str] = set()
        self.fingerprints: set[str] = set()
        for record in records or []:
            self.add(record)

    @staticmethod
    def _doi(record: dict[str, Any]) -> str | None:
        evidence = ((record.get("identifiers") or {}).get("doi") or {})
        return normalize_doi(evidence.get("value"))

    @staticmethod
    def _work_id(record: dict[str, Any]) -> str | None:
        source = record.get("source") or {}
        if source.get("source_type") != "openalex":
            return None
        value = source.get("work_id")
        return str(value) if value else None

    @staticmethod
    def _title_year(record: dict[str, Any]) -> str | None:
        title = str(record.get("normalized_title") or "").strip()
        year = record.get("year")
        return f"{title}|{year}" if title and year else None

    def duplicate_reason(self, record: dict[str, Any]) -> str | None:
        work_id = self._work_id(record)
        if work_id and work_id in self.work_ids:
            return "openalex_work_id"
        doi = self._doi(record)
        if doi and doi in self.dois:
            return "doi"
        title_year = self._title_year(record)
        if title_year and title_year in self.title_years:
            return "normalized_title_year"
        fingerprint = record.get("content_fingerprint")
        if fingerprint and str(fingerprint) in self.fingerprints:
            return "content_fingerprint"
        return None

    def add(self, record: dict[str, Any]) -> None:
        work_id = self._work_id(record)
        if work_id:
            self.work_ids.add(work_id)
        doi = self._doi(record)
        if doi:
            self.dois.add(doi)
        title_year = self._title_year(record)
        if title_year:
            self.title_years.add(title_year)
        fingerprint = record.get("content_fingerprint")
        if fingerprint:
            self.fingerprints.add(str(fingerprint))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain a JSON object")
        records.append(value)
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    addition = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write(path, existing + addition)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def identifier_evidence(value: str | None) -> dict[str, Any]:
    return {
        "value": value,
        "verification_status": "metadata_verified" if value else "missing",
        "source": "external_metadata" if value else None,
    }


def normalize_external_identifier(value: Any, marker: str) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if marker in text.lower():
        text = text.rstrip("/").rsplit("/", 1)[-1]
    return text or None


def metadata_line(authors: list[dict[str, Any]], venue: str | None, year: int | None) -> str | None:
    names = ", ".join(author["name"] for author in authors[:8])
    publication = ", ".join(part for part in (venue, str(year) if year else None) if part)
    if names and publication:
        return f"{names} - {publication}"
    return publication or names or None


def make_candidate(
    work: dict[str, Any],
    *,
    query: QuerySpec,
    rank: int,
    discovered_at: str,
    maximum_abstract: int,
) -> dict[str, Any]:
    normalized = normalize_openalex(work, maximum_abstract)
    work_id = str(normalized.get("openalex_id") or work.get("id") or "").strip()
    title = str(normalized.get("title") or "").strip()
    if not work_id or not title:
        raise ValueError("OpenAlex result is missing work ID or title")

    title_normalized = normalize_title(title)
    if not title_normalized:
        raise ValueError("OpenAlex title normalizes to an empty value")

    year_value = normalized.get("year")
    year = int(year_value) if year_value is not None else None
    venue_value = normalized.get("venue")
    venue = str(venue_value).strip() if venue_value else None
    authors = [
        {
            "name": str(author.get("name") or "").strip(),
            "orcid": author.get("orcid"),
            "verification_status": "metadata_verified",
        }
        for author in normalized.get("authors") or []
        if isinstance(author, dict) and str(author.get("name") or "").strip()
    ]

    ids = work.get("ids") or {}
    doi = normalize_doi(normalized.get("doi"))
    arxiv_id = normalize_external_identifier(
        normalized.get("arxiv_id") or ids.get("arxiv"), "arxiv"
    )
    pmid = normalize_external_identifier(ids.get("pmid"), "pubmed")
    primary_url = normalized.get("landing_page")
    open_access_url = normalized.get("open_access_url")
    auxiliary_urls = sorted(
        {
            str(value)
            for value in (open_access_url, f"https://doi.org/{doi}" if doi else None)
            if value and value != primary_url
        }
    )

    warnings: list[str] = []
    if not venue:
        warnings.append("missing_venue")
    if not year:
        warnings.append("missing_year")
    state = "partial" if warnings else "complete"
    fingerprint = hashlib.sha256(
        f"{title_normalized}|{year or ''}|{venue or ''}".encode("utf-8")
    ).hexdigest()
    candidate_id = hashlib.sha256(f"openalex:{work_id}".encode("utf-8")).hexdigest()[:24]

    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "source": {
            "source_type": "openalex",
            "work_id": work_id,
            "discovered_at": discovered_at,
            "query_id": query.query_id,
            "query_text": query.text,
            "publication_date": normalized.get("publication_date"),
        },
        "position_in_message": rank,
        "title": title,
        "normalized_title": title_normalized,
        "authors": authors,
        "raw_metadata_line": metadata_line(authors, venue, year),
        "venue": {
            "raw": venue,
            "normalized": venue,
            "verification_status": "metadata_verified" if venue else "missing",
        },
        "year": year,
        "snippet": normalized.get("abstract"),
        "identifiers": {
            "doi": identifier_evidence(doi),
            "arxiv_id": identifier_evidence(arxiv_id),
            "pmid": identifier_evidence(pmid),
        },
        "links": {
            "primary_url": primary_url,
            "auxiliary_urls": auxiliary_urls,
        },
        "parse_status": {
            "state": state,
            "warnings": warnings,
            "errors": [],
            "parser_strategy": "openalex_api",
        },
        "content_fingerprint": fingerprint,
        "extracted_at": discovered_at,
    }


def validate_candidate(record: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(error.message for error in errors)
        raise ValueError(f"OpenAlex candidate failed schema validation: {detail}")


def default_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "discovery_version": 1,
        "last_attempt_at": None,
        "last_successful_run_at": None,
        "consecutive_failures": 0,
        "last_window_start": None,
        "last_window_end": None,
        "last_summary": None,
    }


def is_due(
    state: dict[str, Any],
    *,
    now: datetime,
    minimum_interval_hours: int,
    force: bool,
) -> bool:
    if force:
        return True
    previous = parse_datetime(state.get("last_successful_run_at"))
    return previous is None or now - previous >= timedelta(hours=minimum_interval_hours)


def query_window(
    state: dict[str, Any],
    *,
    now: datetime,
    overlap_days: int,
    initial_lookback_days: int,
) -> tuple[str, str]:
    previous = parse_datetime(state.get("last_successful_run_at"))
    start = (
        previous - timedelta(days=overlap_days)
        if previous
        else now - timedelta(days=initial_lookback_days)
    )
    return start.date().isoformat(), now.date().isoformat()


def load_queries(config: dict[str, Any]) -> list[QuerySpec]:
    queries: list[QuerySpec] = []
    for item in config.get("queries", []):
        query_id = str(item.get("id") or "").strip()
        project_id = str(item.get("project_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not query_id or not project_id or not text:
            raise ValueError("Every OpenAlex discovery query needs id, project_id, and text")
        queries.append(QuerySpec(query_id, project_id, text))
    if not queries:
        raise ValueError("No OpenAlex discovery queries are configured")
    return queries


def discover_once(
    *,
    config_path: Path,
    registry_path: Path,
    state_path: Path,
    manifest_path: Path,
    schema_path: Path,
    recognition_path: Path,
    venues_path: Path,
    api_key: str | None = None,
    client: JsonHttpClient | None = None,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    now = now or utc_now()
    now_text = isoformat(now)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("OpenAlex discovery config must be a YAML object")
    state = load_json(state_path, default_state())
    schedule = config["schedule"]
    if not is_due(
        state,
        now=now,
        minimum_interval_hours=int(schedule["minimum_interval_hours"]),
        force=force,
    ):
        return {
            "status": "skipped_not_due",
            "candidate_count": len(load_jsonl(registry_path)),
            "last_successful_run_at": state.get("last_successful_run_at"),
        }

    provider_config = config["provider"]
    api_key = api_key or os.environ.get(str(provider_config["api_key_env"]))
    if not api_key:
        raise RuntimeError("Missing required OpenAlex API key")
    client = client or JsonHttpClient(
        user_agent=str(provider_config["user_agent"]),
        timeout_seconds=float(provider_config["timeout_seconds"]),
        max_attempts=int(provider_config["max_attempts"]),
        min_interval_seconds=float(provider_config["min_interval_seconds"]),
    )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    recognition = load_yaml(recognition_path)
    venues = load_yaml(venues_path)
    queries = load_queries(config)
    limits = config["limits"]
    maximum_per_query = int(limits["maximum_results_per_query"])
    maximum_raw = int(limits["maximum_raw_results_per_run"])
    maximum_accepted = int(limits["maximum_accepted_candidates_per_run"])
    maximum_abstract = int(limits["maximum_abstract_characters"])
    start_date, end_date = query_window(
        state,
        now=now,
        overlap_days=int(schedule["overlap_days"]),
        initial_lookback_days=int(schedule["initial_lookback_days"]),
    )

    existing = load_jsonl(registry_path)
    identities = RegistryIdentityIndex(existing)
    accepted: list[dict[str, Any]] = []
    raw_count = 0
    duplicate_count = 0
    filtered_count = 0
    invalid_count = 0
    error_count = 0
    duplicate_reasons: dict[str, int] = {}
    query_summaries: list[dict[str, Any]] = []
    stop = False

    for query in queries:
        query_raw = 0
        query_accepted = 0
        query_duplicates = 0
        query_filtered = 0
        query_invalid = 0
        query_error: str | None = None
        try:
            payload = client.get_json(
                str(provider_config["base_url"]),
                "works",
                params={
                    "search": query.text,
                    "filter": (
                        f"from_publication_date:{start_date},"
                        f"to_publication_date:{end_date}"
                    ),
                    "sort": "publication_date:desc",
                    "per_page": maximum_per_query,
                    "select": OpenAlexProvider.SELECT,
                    "api_key": api_key,
                },
            )
            results = payload.get("results") or []
            if not isinstance(results, list):
                raise ValueError("OpenAlex results must be a list")
        except HttpRequestError as error:
            results = []
            error_count += 1
            query_error = error.reason

        for rank, work in enumerate(results):
            if raw_count >= maximum_raw or len(accepted) >= maximum_accepted:
                stop = True
                break
            if not isinstance(work, dict):
                invalid_count += 1
                query_invalid += 1
                continue
            raw_count += 1
            query_raw += 1
            try:
                candidate = make_candidate(
                    work,
                    query=query,
                    rank=rank,
                    discovered_at=now_text,
                    maximum_abstract=maximum_abstract,
                )
                validate_candidate(candidate, schema)
            except (TypeError, ValueError):
                invalid_count += 1
                query_invalid += 1
                continue

            duplicate_reason = identities.duplicate_reason(candidate)
            if duplicate_reason:
                duplicate_count += 1
                query_duplicates += 1
                duplicate_reasons[duplicate_reason] = (
                    duplicate_reasons.get(duplicate_reason, 0) + 1
                )
                continue

            result = classify_candidate(
                candidate,
                recognition_config=recognition,
                venues_config=venues,
                classified_at=now_text,
            )
            matched_project_ids = {
                str(project.get("project_id"))
                for project in result.get("matched_projects") or []
            }
            if not matched_project_ids or query.project_id not in matched_project_ids:
                filtered_count += 1
                query_filtered += 1
                continue

            accepted.append(candidate)
            identities.add(candidate)
            query_accepted += 1

        query_summaries.append(
            {
                "query_id": query.query_id,
                "project_id": query.project_id,
                "raw_results": query_raw,
                "accepted": query_accepted,
                "duplicates": query_duplicates,
                "filtered": query_filtered,
                "invalid": query_invalid,
                "error": query_error,
            }
        )
        if stop:
            break

    if error_count == len(queries) and raw_count == 0:
        raise RuntimeError("All OpenAlex discovery queries failed")

    append_jsonl(registry_path, accepted)
    state.update(
        {
            "schema_version": 1,
            "discovery_version": int(config["discovery_version"]),
            "last_attempt_at": now_text,
            "last_successful_run_at": now_text,
            "consecutive_failures": 0,
            "last_window_start": start_date,
            "last_window_end": end_date,
            "last_summary": {
                "raw_results": raw_count,
                "accepted": len(accepted),
                "duplicates": duplicate_count,
                "filtered": filtered_count,
                "invalid": invalid_count,
                "query_errors": error_count,
            },
        }
    )
    manifest = {
        "schema_version": 1,
        "discovery_version": int(config["discovery_version"]),
        "status": "completed",
        "built_at": now_text,
        "window_start": start_date,
        "window_end": end_date,
        "raw_result_count": raw_count,
        "accepted_count": len(accepted),
        "duplicate_count": duplicate_count,
        "duplicate_reasons": dict(sorted(duplicate_reasons.items())),
        "filtered_count": filtered_count,
        "invalid_count": invalid_count,
        "query_error_count": error_count,
        "accepted_candidate_ids": sorted(
            str(candidate["candidate_id"]) for candidate in accepted
        ),
        "query_summaries": query_summaries,
        "registry_candidate_count": len(existing) + len(accepted),
        "registry_sha256": file_digest(registry_path),
    }
    atomic_write(
        state_path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover recent research through OpenAlex with deterministic filtering"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/openalex_discovery.yaml"),
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("runtime-state/data/paper_registry.jsonl"),
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("runtime-state/state/openalex_discovery_state.json"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("runtime-state/state/openalex_discovery_manifest.json"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/openalex_candidate.schema.json"),
    )
    parser.add_argument(
        "--recognition",
        type=Path,
        default=Path("config/recognition_rules.yaml"),
    )
    parser.add_argument(
        "--venues",
        type=Path,
        default=Path("config/venues.yaml"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = discover_once(
        config_path=args.config,
        registry_path=args.registry_path,
        state_path=args.state_path,
        manifest_path=args.manifest_path,
        schema_path=args.schema,
        recognition_path=args.recognition,
        venues_path=args.venues,
        force=args.force,
    )
    summary = {
        key: manifest.get(key)
        for key in (
            "status",
            "raw_result_count",
            "accepted_count",
            "duplicate_count",
            "filtered_count",
            "invalid_count",
            "query_error_count",
            "registry_candidate_count",
            "candidate_count",
            "last_successful_run_at",
        )
        if key in manifest
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

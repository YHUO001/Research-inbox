from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.enrich.metadata import normalize_doi
from scripts.ingest.scholar_parser import normalize_title

SCHOLAR_SOURCE = "google_scholar_email"
OPENALEX_SOURCE = "openalex"


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
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain a JSON object")
        records.append(value)
    return records


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def completed_ids(history: dict[str, Any]) -> set[str]:
    value = history.get("completed_candidate_ids") or {}
    if isinstance(value, dict):
        return {str(item) for item in value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def source_type(record: dict[str, Any]) -> str:
    return str((record.get("source") or {}).get("source_type") or "unknown")


def record_doi(record: dict[str, Any]) -> str | None:
    evidence = ((record.get("identifiers") or {}).get("doi") or {})
    return normalize_doi(evidence.get("value"))


def record_openalex_id(record: dict[str, Any]) -> str | None:
    source = record.get("source") or {}
    if source.get("source_type") == OPENALEX_SOURCE and source.get("work_id"):
        return str(source["work_id"]).rstrip("/").rsplit("/", 1)[-1].casefold()
    return None


def record_title(record: dict[str, Any]) -> str:
    return str(record.get("normalized_title") or normalize_title(str(record.get("title") or ""))).strip()


def record_year(record: dict[str, Any]) -> int | None:
    value = record.get("year")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def first_author_key(record: dict[str, Any]) -> str | None:
    authors = record.get("authors") or []
    if not authors:
        return None
    first = authors[0]
    name = str(first.get("name") if isinstance(first, dict) else first).strip().casefold()
    if not name:
        return None
    return "".join(character for character in name.split()[-1] if character.isalnum()) or None


def identity_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    doi = record_doi(record)
    if doi:
        keys.append(f"doi:{doi}")
    openalex_id = record_openalex_id(record)
    if openalex_id:
        keys.append(f"openalex:{openalex_id}")
    title = record_title(record)
    year = record_year(record)
    if title and year:
        keys.append(f"title_year:{title}|{year}")
    fingerprint = str(record.get("content_fingerprint") or "").strip()
    if fingerprint:
        keys.append(f"fingerprint:{fingerprint}")
    return keys


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def group_records(records: list[dict[str, Any]]) -> list[list[int]]:
    union = UnionFind(len(records))
    owner: dict[str, int] = {}
    title_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for key in identity_keys(record):
            previous = owner.get(key)
            if previous is None:
                owner[key] = index
            else:
                union.union(previous, index)
        title = record_title(record)
        if len(title) >= 24:
            title_groups[title].append(index)

    for indexes in title_groups.values():
        for offset, left in enumerate(indexes):
            left_year = record_year(records[left])
            left_author = first_author_key(records[left])
            for right in indexes[offset + 1 :]:
                right_year = record_year(records[right])
                right_author = first_author_key(records[right])
                years_compatible = (
                    left_year is None
                    or right_year is None
                    or abs(left_year - right_year) <= 1
                )
                authors_compatible = (
                    left_author is None
                    or right_author is None
                    or left_author == right_author
                )
                if years_compatible and authors_compatible:
                    union.union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union.find(index)].append(index)
    return sorted(grouped.values(), key=lambda values: min(values))


def canonical_index(
    indexes: list[int],
    records: list[dict[str, Any]],
    completed: set[str],
) -> int:
    def priority(index: int) -> tuple[int, int, int]:
        record = records[index]
        candidate_id = str(record.get("candidate_id") or "")
        completed_rank = 0 if candidate_id in completed else 1
        source_rank = 0 if source_type(record) == SCHOLAR_SOURCE else 1
        return completed_rank, source_rank, index

    return min(indexes, key=priority)


def evidence_score(value: Any) -> tuple[int, int]:
    if not isinstance(value, dict) or not value.get("value"):
        return 0, 0
    verification = str(value.get("verification_status") or "")
    rank = {
        "metadata_verified": 4,
        "doi_extracted": 3,
        "parser_extracted": 2,
        "unverified": 1,
    }.get(verification, 1)
    return rank, len(str(value.get("value") or ""))


def merge_identifier(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [((record.get("identifiers") or {}).get(name) or {}) for record in records]
    best = max(values, key=evidence_score, default={})
    if best and best.get("value"):
        return copy.deepcopy(best)
    return {"value": None, "verification_status": "missing", "source": None}


def venue_value(record: dict[str, Any]) -> str | None:
    venue = record.get("venue") or {}
    if isinstance(venue, dict):
        value = venue.get("normalized") or venue.get("raw")
        return str(value).strip() if value else None
    return str(venue).strip() if venue else None


def venue_score(record: dict[str, Any]) -> tuple[int, int]:
    venue = record.get("venue") or {}
    value = venue_value(record)
    if not value:
        return 0, 0
    verification = str(venue.get("verification_status") or "") if isinstance(venue, dict) else ""
    rank = {"metadata_verified": 3, "parser_extracted": 2, "unverified": 1}.get(verification, 1)
    return rank, len(value)


def provenance(record: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(record.get("source") or {})
    return {
        "candidate_id": str(record.get("candidate_id") or ""),
        "source_type": source.pop("source_type", "unknown"),
        "source": source,
        "content_fingerprint": record.get("content_fingerprint"),
    }


def merge_group(
    indexes: list[int],
    records: list[dict[str, Any]],
    completed: set[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    canonical_position = canonical_index(indexes, records, completed)
    canonical = copy.deepcopy(records[canonical_position])
    members = [records[index] for index in indexes]
    canonical_id = str(canonical.get("candidate_id") or "")
    if not canonical_id:
        raise ValueError("Every raw registry record must contain candidate_id")

    aliases = {
        str(member.get("candidate_id") or ""): canonical_id
        for member in members
        if str(member.get("candidate_id") or "") and str(member.get("candidate_id")) != canonical_id
    }

    titles = [str(member.get("title") or "").strip() for member in members]
    titles = [value for value in titles if value]
    if titles:
        canonical["title"] = max(titles, key=len)
        canonical["normalized_title"] = normalize_title(canonical["title"])

    author_lists = [member.get("authors") or [] for member in members]
    canonical["authors"] = copy.deepcopy(max(author_lists, key=len, default=[]))

    best_venue_record = max(members, key=venue_score)
    if venue_value(best_venue_record):
        canonical["venue"] = copy.deepcopy(best_venue_record.get("venue"))

    years = [record_year(member) for member in members if record_year(member) is not None]
    canonical_year = record_year(canonical)
    if canonical_year is None and years:
        canonical_year = Counter(years).most_common(1)[0][0]
    canonical["year"] = canonical_year

    snippets = [str(member.get("snippet") or "").strip() for member in members]
    snippets = [value for value in snippets if value]
    canonical["snippet"] = max(snippets, key=len) if snippets else None

    metadata_lines = [str(member.get("raw_metadata_line") or "").strip() for member in members]
    metadata_lines = [value for value in metadata_lines if value]
    canonical["raw_metadata_line"] = max(metadata_lines, key=len) if metadata_lines else None

    canonical["identifiers"] = {
        name: merge_identifier(members, name)
        for name in ("doi", "arxiv_id", "pmid")
    }

    primary_urls: list[str] = []
    auxiliary_urls: set[str] = set()
    for member in members:
        links = member.get("links") or {}
        primary = str(links.get("primary_url") or "").strip()
        if primary:
            primary_urls.append(primary)
        auxiliary_urls.update(
            str(value).strip()
            for value in links.get("auxiliary_urls") or []
            if str(value).strip()
        )
    doi = record_doi(canonical)
    doi_url = f"https://doi.org/{doi}" if doi else None
    primary = str((canonical.get("links") or {}).get("primary_url") or "").strip()
    if not primary:
        primary = doi_url or (primary_urls[0] if primary_urls else "")
    auxiliary_urls.update(primary_urls)
    if doi_url:
        auxiliary_urls.add(doi_url)
    auxiliary_urls.discard(primary)
    canonical["links"] = {
        "primary_url": primary or None,
        "auxiliary_urls": sorted(auxiliary_urls),
    }

    states = [str((member.get("parse_status") or {}).get("state") or "partial") for member in members]
    warnings = {
        str(value)
        for member in members
        for value in ((member.get("parse_status") or {}).get("warnings") or [])
    }
    errors = sorted(
        {
            str(value)
            for member in members
            for value in ((member.get("parse_status") or {}).get("errors") or [])
        }
    )
    if venue_value(canonical):
        warnings.discard("missing_venue")
        warnings.discard("unresolved_venue")
    if canonical_year:
        warnings.discard("missing_year")
    if len(set(years)) > 1:
        warnings.add("conflicting_year_across_sources")
    if len(members) > 1:
        warnings.add("merged_cross_source_records")
    manual_review = (
        str((canonical.get("parse_status") or {}).get("state") or "") == "manual_review"
    )
    state = "manual_review" if manual_review else (
        "complete" if venue_value(canonical) and canonical_year else "partial"
    )
    canonical["parse_status"] = {
        "state": state,
        "warnings": sorted(warnings),
        "errors": errors,
        "parser_strategy": "unified_registry_reconciliation",
    }

    normalized = record_title(canonical)
    canonical["content_fingerprint"] = hashlib.sha256(
        f"{normalized}|{canonical_year or ''}|{venue_value(canonical) or ''}".encode("utf-8")
    ).hexdigest()
    canonical["source_provenance"] = sorted(
        (provenance(member) for member in members),
        key=lambda item: (str(item["source_type"]), str(item["candidate_id"])),
    )
    canonical["source_types"] = sorted({source_type(member) for member in members})
    canonical["source_alias_candidate_ids"] = sorted(aliases)
    canonical["unified_registry_version"] = 1
    return canonical, aliases


def reconcile_registry(
    *,
    raw_registry_path: Path,
    unified_registry_path: Path,
    alias_path: Path,
    manifest_path: Path,
    history_path: Path | None = None,
) -> dict[str, Any]:
    records = load_jsonl(raw_registry_path)
    history = load_json(history_path, {}) if history_path else {}
    completed = completed_ids(history if isinstance(history, dict) else {})

    unified: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    source_combinations: Counter[str] = Counter()
    merged_groups = 0
    for indexes in group_records(records):
        record, group_aliases = merge_group(indexes, records, completed)
        unified.append(record)
        aliases.update(group_aliases)
        types = "+".join(record.get("source_types") or ["unknown"])
        source_combinations[types] += 1
        merged_groups += int(len(indexes) > 1)

    unified.sort(key=lambda item: str(item.get("candidate_id") or ""))
    atomic_write(
        unified_registry_path,
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in unified),
    )
    alias_document = {
        "schema_version": 1,
        "aliases": dict(sorted(aliases.items())),
    }
    atomic_write(
        alias_path,
        json.dumps(alias_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "raw_candidate_count": len(records),
        "unified_candidate_count": len(unified),
        "merged_group_count": merged_groups,
        "alias_count": len(aliases),
        "source_combination_counts": dict(sorted(source_combinations.items())),
        "raw_registry_path": str(raw_registry_path),
        "raw_registry_sha256": file_sha256(raw_registry_path),
        "unified_registry_path": str(unified_registry_path),
        "unified_registry_sha256": file_sha256(unified_registry_path),
        "alias_path": str(alias_path),
        "alias_sha256": file_sha256(alias_path),
    }
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconcile Scholar and OpenAlex into one registry")
    parser.add_argument("--raw-registry-path", type=Path, required=True)
    parser.add_argument("--unified-registry-path", type=Path, required=True)
    parser.add_argument("--alias-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--history-path", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reconcile_registry(
        raw_registry_path=args.raw_registry_path,
        unified_registry_path=args.unified_registry_path,
        alias_path=args.alias_path,
        manifest_path=args.manifest_path,
        history_path=args.history_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

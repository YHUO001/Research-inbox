from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from scripts.enrich.metadata import (
    CrossrefProvider,
    JsonHttpClient,
    NormalizedCache,
    OpenAlexProvider,
    enrich_candidate,
)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain a JSON object")
        records.append(value)
    return records


def jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return value


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(error.message for error in errors)
        raise ValueError(f"Enriched record failed schema validation: {detail}")


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def build_clients(
    config: dict[str, Any],
    *,
    cache: NormalizedCache,
) -> tuple[CrossrefProvider, OpenAlexProvider, bool]:
    providers = config["providers"]
    matching = config["matching"]
    cache_config = config["cache"]
    maximum_abstract = int(config["limits"]["maximum_abstract_characters"])

    crossref_config = providers["crossref"]
    crossref_email = os.environ.get(str(crossref_config["contact_email_env"]))
    crossref_client = JsonHttpClient(
        user_agent=str(crossref_config["user_agent"]),
        timeout_seconds=float(crossref_config["timeout_seconds"]),
        max_attempts=int(crossref_config["max_attempts"]),
        min_interval_seconds=float(crossref_config["min_interval_seconds"]),
    )
    crossref = CrossrefProvider(
        crossref_config,
        client=crossref_client,
        cache=cache,
        contact_email=crossref_email,
        maximum_abstract=maximum_abstract,
        exact_ttl_days=int(cache_config["exact_identifier_ttl_days"]),
        search_ttl_days=int(cache_config["title_search_ttl_days"]),
        matching=matching,
    )

    openalex_config = providers["openalex"]
    openalex_key = os.environ.get(str(openalex_config["api_key_env"]))
    openalex_client = JsonHttpClient(
        user_agent="ResearchInbox/0.3 (OpenAlex metadata enrichment)",
        timeout_seconds=float(openalex_config["timeout_seconds"]),
        max_attempts=int(openalex_config["max_attempts"]),
        min_interval_seconds=float(openalex_config["min_interval_seconds"]),
    )
    openalex = OpenAlexProvider(
        openalex_config,
        client=openalex_client,
        cache=cache,
        api_key=openalex_key,
        maximum_abstract=maximum_abstract,
        exact_ttl_days=int(cache_config["exact_identifier_ttl_days"]),
        search_ttl_days=int(cache_config["title_search_ttl_days"]),
        matching=matching,
    )
    return crossref, openalex, bool(openalex_key)


def enrich_registry(
    *,
    registry_path: Path,
    output_path: Path,
    manifest_path: Path,
    cache_path: Path,
    config_path: Path,
    schema_path: Path,
    crossref: CrossrefProvider | None = None,
    openalex: OpenAlexProvider | None = None,
    openalex_configured: bool | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    candidates = load_jsonl(registry_path)
    maximum = int(config["limits"]["maximum_candidates_per_run"])
    if len(candidates) > maximum:
        raise ValueError(
            f"Registry contains {len(candidates)} candidates; "
            f"configured maximum is {maximum}"
        )

    candidate_ids = [
        str(candidate.get("candidate_id") or "") for candidate in candidates
    ]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("Every source record must contain candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Source registry contains duplicate candidate_id values")

    cache = NormalizedCache.load(cache_path)
    if crossref is None or openalex is None:
        crossref, openalex, configured = build_clients(config, cache=cache)
        openalex_configured = configured
    elif openalex_configured is None:
        openalex_configured = True

    enriched = [
        enrich_candidate(candidate, crossref=crossref, openalex=openalex)
        for candidate in candidates
    ]
    for record in enriched:
        validate_record(record, schema)

    match_counts = Counter(record["match"]["status"] for record in enriched)
    attempt_counts = Counter(
        (attempt["provider"], attempt["status"])
        for record in enriched
        for attempt in record["provider_attempts"]
    )
    timestamps = [
        str(record["enriched_at"])
        for record in enriched
        if record.get("enriched_at")
    ]
    manifest = {
        "schema_version": 1,
        "enrichment_version": int(config["enrichment_version"]),
        "source_registry_sha256": source_digest(registry_path),
        "candidate_count": len(candidates),
        "match_counts": {
            status: match_counts.get(status, 0)
            for status in ("exact", "high", "unresolved")
        },
        "provider_attempt_counts": {
            provider: {
                status: attempt_counts.get((provider, status), 0)
                for status in ("found", "not_found", "error", "skipped")
            }
            for provider in ("crossref", "openalex")
        },
        "openalex_configured": bool(openalex_configured),
        "built_at": max(timestamps) if timestamps else None,
    }

    atomic_write(output_path, jsonl(enriched))
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        cache_path,
        json.dumps(
            cache.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservatively enrich the immutable paper registry"
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("runtime-state/data/paper_registry.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("runtime-state/data/enriched_paper_registry.jsonl"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("runtime-state/state/enrichment_manifest.json"),
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("runtime-state/state/enrichment_cache.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/metadata_enrichment.yaml"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/enriched_paper.schema.json"),
    )
    args = parser.parse_args()

    manifest = enrich_registry(
        registry_path=args.registry_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        cache_path=args.cache_path,
        config_path=args.config,
        schema_path=args.schema,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "candidate_count": manifest["candidate_count"],
                "match_counts": manifest["match_counts"],
                "provider_attempt_counts": manifest[
                    "provider_attempt_counts"
                ],
                "openalex_configured": manifest["openalex_configured"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

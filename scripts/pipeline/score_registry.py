from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker


SCORING_VERSION = 1
DECISION_ORDER = {
    "mandatory": 0,
    "urgent": 1,
    "summarize": 2,
    "retain_without_summary": 3,
    "archive": 4,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
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


def jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return value


def scalar_value(enriched: dict[str, Any] | None, field: str) -> Any:
    if not enriched:
        return None
    item = (enriched.get("fields") or {}).get(field) or {}
    return item.get("value") if isinstance(item, dict) else None


def candidate_doi(candidate: dict[str, Any]) -> str | None:
    item = ((candidate.get("identifiers") or {}).get("doi") or {})
    value = item.get("value") if isinstance(item, dict) else None
    return str(value) if value else None


def candidate_openalex_id(candidate: dict[str, Any]) -> str | None:
    source = candidate.get("source") or {}
    if source.get("source_type") == "openalex" and source.get("work_id"):
        return str(source["work_id"])
    return None


def candidate_venue(candidate: dict[str, Any]) -> str | None:
    venue = candidate.get("venue") or {}
    value = venue.get("normalized") or venue.get("raw")
    return str(value) if value else None


def candidate_authors(
    candidate: dict[str, Any], enriched: dict[str, Any] | None
) -> list[str]:
    enriched_authors = enriched.get("authors") if enriched else None
    source = (
        enriched_authors
        if isinstance(enriched_authors, list) and enriched_authors
        else candidate.get("authors") or []
    )
    return [
        str(item.get("name") or "").strip()
        for item in source
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]


def resolved_metadata(
    candidate: dict[str, Any],
    enriched: dict[str, Any] | None,
) -> dict[str, Any]:
    source = candidate.get("source") or {}
    publication_date = (
        scalar_value(enriched, "publication_date") or source.get("publication_date")
    )
    year = scalar_value(enriched, "year") or candidate.get("year")
    venue = scalar_value(enriched, "venue") or candidate_venue(candidate)
    doi = scalar_value(enriched, "doi") or candidate_doi(candidate)
    openalex_id = scalar_value(enriched, "openalex_id") or candidate_openalex_id(
        candidate
    )
    abstract = scalar_value(enriched, "abstract") or candidate.get("snippet")
    landing_page = scalar_value(enriched, "landing_page") or (
        candidate.get("links") or {}
    ).get("primary_url")
    open_access_url = scalar_value(enriched, "open_access_url")
    return {
        "title": str(scalar_value(enriched, "title") or candidate.get("title") or ""),
        "source_type": str(source.get("source_type") or "unknown"),
        "publication_date": str(publication_date) if publication_date else None,
        "year": int(year) if year is not None else None,
        "venue": str(venue) if venue else None,
        "doi": str(doi) if doi else None,
        "openalex_id": str(openalex_id) if openalex_id else None,
        "abstract": str(abstract) if abstract else None,
        "landing_page": str(landing_page) if landing_page else None,
        "open_access_url": str(open_access_url) if open_access_url else None,
        "authors": candidate_authors(candidate, enriched),
    }


def add_breakdown(
    breakdown: list[dict[str, Any]],
    feature: str,
    contribution: float,
    evidence: list[str] | None = None,
) -> float:
    value = round(float(contribution), 6)
    if value == 0:
        return 0.0
    breakdown.append(
        {
            "feature": feature,
            "contribution": value,
            "evidence": sorted(set(evidence or [])),
        }
    )
    return value


def signal_matches(text: str, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        hit = re.search(str(pattern), text, flags=re.IGNORECASE)
        if hit:
            matches.append(hit.group(0))
    return matches


def publication_age_days(metadata: dict[str, Any], scored_at: datetime) -> int | None:
    publication_date = parse_datetime(metadata.get("publication_date"))
    if publication_date:
        return max(0, (scored_at - publication_date).days)
    year = metadata.get("year")
    if year:
        return max(
            0,
            (
                scored_at.date()
                - datetime(int(year), 1, 1, tzinfo=timezone.utc).date()
            ).days,
        )
    return None


def score_candidate(
    candidate: dict[str, Any],
    recognition: dict[str, Any],
    enriched: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    scored_at: datetime,
) -> dict[str, Any]:
    metadata = resolved_metadata(candidate, enriched)
    text = " ".join(
        value for value in (metadata["title"], metadata["abstract"] or "") if value
    )
    weights = config["weights"]
    breakdown: list[dict[str, Any]] = []
    projects = recognition.get("matched_projects") or []
    confidence_weights = weights["project_confidence"]
    confidences = [
        str(project.get("confidence") or "uncertain")
        for project in projects
        if isinstance(project, dict)
    ]
    best_confidence = max(
        (float(confidence_weights.get(value, 0.0)) for value in confidences),
        default=0.0,
    )
    add_breakdown(
        breakdown,
        "project_confidence",
        best_confidence,
        confidences,
    )
    if len(projects) > 1:
        add_breakdown(
            breakdown,
            "multiple_project_match",
            float(weights["multiple_project_bonus"]),
            [str(project.get("project_id")) for project in projects],
        )

    title_evidence: list[str] = []
    contextual_evidence: list[str] = []
    priority_weight = 0.0
    negative_count = 0
    for project in projects:
        for evidence in project.get("positive_evidence") or []:
            field = str(evidence.get("field") or "")
            matched = str(evidence.get("matched_text") or "")
            if field == "title":
                title_evidence.append(matched)
            else:
                contextual_evidence.append(matched)
        priority_weight += sum(
            float(feature.get("weight") or 0.0)
            for feature in project.get("priority_features") or []
        )
        negative_count += len(project.get("negative_evidence") or [])

    if title_evidence:
        add_breakdown(
            breakdown,
            "title_evidence",
            float(weights["title_evidence"]),
            title_evidence,
        )
    if contextual_evidence:
        add_breakdown(
            breakdown,
            "contextual_evidence",
            float(weights["contextual_evidence"]),
            contextual_evidence,
        )
    if priority_weight > 0:
        add_breakdown(
            breakdown,
            "priority_features",
            min(
                float(weights["priority_features_cap"]),
                priority_weight * float(weights["priority_features_cap"]),
            ),
            [
                str(feature.get("feature"))
                for project in projects
                for feature in project.get("priority_features") or []
            ],
        )

    for name in ("architecture", "training", "application"):
        item = config["signals"][name]
        hits = signal_matches(text, list(item.get("patterns") or []))
        if hits:
            add_breakdown(
                breakdown,
                f"{name}_signal",
                float(item.get("weight") or 0.0),
                hits,
            )

    venue_tier = (recognition.get("venue_policy") or {}).get("matched_tier")
    venue_key = str(venue_tier) if venue_tier else "unresolved"
    add_breakdown(
        breakdown,
        "venue_policy",
        float((weights.get("venue") or {}).get(venue_key, 0.0)),
        [venue_key],
    )

    metadata_weights = weights["metadata"]
    if metadata["abstract"]:
        add_breakdown(
            breakdown,
            "metadata_abstract",
            float(metadata_weights["abstract"]),
            ["abstract"],
        )
    if metadata["doi"]:
        add_breakdown(
            breakdown,
            "metadata_doi",
            float(metadata_weights["doi"]),
            [metadata["doi"]],
        )
    if metadata["openalex_id"]:
        add_breakdown(
            breakdown,
            "metadata_openalex_id",
            float(metadata_weights["openalex_id"]),
            [metadata["openalex_id"]],
        )
    if metadata["venue"] and metadata["year"]:
        add_breakdown(
            breakdown,
            "metadata_venue_and_year",
            float(metadata_weights["venue_and_year"]),
            [metadata["venue"], str(metadata["year"])],
        )

    age_days = publication_age_days(metadata, scored_at)
    freshness = weights["freshness"]
    if age_days is not None:
        if age_days <= 30:
            freshness_value = float(freshness["within_30_days"])
        elif age_days <= 90:
            freshness_value = float(freshness["within_90_days"])
        elif age_days <= 365:
            freshness_value = float(freshness["within_365_days"])
        else:
            freshness_value = 0.0
        add_breakdown(
            breakdown,
            "freshness",
            freshness_value,
            [f"{age_days}_days"],
        )

    if metadata["source_type"] == "google_scholar_email":
        add_breakdown(
            breakdown,
            "primary_source_bonus",
            float(weights["primary_source_bonus"]),
            ["google_scholar_email"],
        )

    if negative_count:
        penalty = -min(
            float(weights["negative_evidence_penalty_cap"]),
            negative_count * float(weights["negative_evidence_penalty_each"]),
        )
        add_breakdown(
            breakdown,
            "negative_evidence_penalty",
            penalty,
            [str(negative_count)],
        )

    review_hits = signal_matches(
        text,
        list(config["signals"]["review_or_perspective"].get("patterns") or []),
    )
    if review_hits:
        add_breakdown(
            breakdown,
            "review_or_perspective_penalty",
            -float(weights["review_or_perspective_penalty"]),
            review_hits,
        )

    raw_score = round(
        max(0.0, min(1.0, sum(item["contribution"] for item in breakdown))),
        6,
    )
    routing = recognition.get("routing") or {}
    mandatory = bool(routing.get("mandatory"))
    thresholds = config["thresholds"]
    final_score = (
        max(raw_score, float(thresholds["urgent_from"])) if mandatory else raw_score
    )
    if mandatory:
        decision = "mandatory"
    elif final_score >= float(thresholds["urgent_from"]):
        decision = "urgent"
    elif final_score >= float(thresholds["summarize_from"]):
        decision = "summarize"
    elif final_score >= float(thresholds["retain_without_summary_from"]):
        decision = "retain_without_summary"
    else:
        decision = "archive"

    return {
        "schema_version": 1,
        "candidate_id": str(candidate["candidate_id"]),
        "scoring_version": int(config["scoring_version"]),
        "route": str(routing.get("route") or ""),
        "mandatory": mandatory,
        "score": round(final_score, 6),
        "score_without_mandatory_override": raw_score,
        "decision": decision,
        "breakdown": breakdown,
        "matched_projects": sorted(
            str(project.get("project_id"))
            for project in projects
            if project.get("project_id")
        ),
        "metadata": {
            "title": metadata["title"],
            "source_type": metadata["source_type"],
            "publication_date": metadata["publication_date"],
            "year": metadata["year"],
            "venue": metadata["venue"],
            "doi": metadata["doi"],
            "openalex_id": metadata["openalex_id"],
            "abstract_available": bool(metadata["abstract"]),
        },
        "selection_eligible": decision in {"mandatory", "urgent", "summarize"},
        "scored_at": isoformat(scored_at),
    }


def selection_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    metadata = result["metadata"]
    date_digits = re.sub(r"\D", "", str(metadata.get("publication_date") or ""))
    date_rank = int((date_digits + "00000000")[:8]) if date_digits else 0
    year = int(metadata.get("year") or 0)
    return (
        0 if result["mandatory"] else 1,
        DECISION_ORDER[result["decision"]],
        -float(result["score"]),
        -date_rank,
        -year,
        str(result["candidate_id"]),
    )


def queue_entry(
    result: dict[str, Any],
    candidate: dict[str, Any],
    enriched: dict[str, Any] | None,
    *,
    selection_status: str,
) -> dict[str, Any]:
    metadata = resolved_metadata(candidate, enriched)
    return {
        "schema_version": 1,
        "candidate_id": result["candidate_id"],
        "title": metadata["title"],
        "authors": metadata["authors"],
        "venue": metadata["venue"],
        "year": metadata["year"],
        "source_type": metadata["source_type"],
        "doi": metadata["doi"],
        "openalex_id": metadata["openalex_id"],
        "landing_page": metadata["landing_page"],
        "open_access_url": metadata["open_access_url"],
        "abstract": metadata["abstract"],
        "matched_projects": result["matched_projects"],
        "mandatory": result["mandatory"],
        "score": result["score"],
        "decision": result["decision"],
        "selection_status": selection_status,
        "score_breakdown": result["breakdown"],
    }


def validate_record(record: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(error.message for error in errors)
        raise ValueError(f"{label} failed schema validation: {detail}")


def completed_ids(history: dict[str, Any]) -> set[str]:
    value = history.get("completed_candidate_ids") or {}
    if isinstance(value, dict):
        return {str(key) for key in value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def score_registry(
    *,
    registry_path: Path,
    enriched_path: Path,
    recognition_path: Path,
    history_path: Path,
    scoring_path: Path,
    queue_path: Path,
    manifest_path: Path,
    config_path: Path,
    scoring_schema_path: Path,
    queue_schema_path: Path,
    scored_at: datetime | None = None,
) -> dict[str, Any]:
    scored_at = scored_at or utc_now()
    config = load_yaml(config_path)
    source_records = load_jsonl(registry_path)
    recognition_records = load_jsonl(recognition_path)
    enriched_records = load_jsonl(enriched_path)
    history = load_json(
        history_path,
        {
            "schema_version": 1,
            "completed_candidate_ids": {},
            "failed_candidate_ids": {},
        },
    )
    if not isinstance(history, dict):
        raise ValueError("Summary history must be a JSON object")

    candidates = {
        str(item.get("candidate_id") or ""): item for item in source_records
    }
    recognitions = {
        str(item.get("candidate_id") or ""): item for item in recognition_records
    }
    enrichments = {
        str(item.get("candidate_id") or ""): item for item in enriched_records
    }
    if "" in candidates or len(candidates) != len(source_records):
        raise ValueError("Source registry must contain unique non-empty candidate IDs")
    if "" in recognitions or len(recognitions) != len(recognition_records):
        raise ValueError(
            "Recognition results must contain unique non-empty candidate IDs"
        )

    routes = set(config["policy"]["score_only_routes"])
    scoring_schema = json.loads(scoring_schema_path.read_text(encoding="utf-8"))
    queue_schema = json.loads(queue_schema_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        recognition = recognitions.get(candidate_id)
        if not recognition:
            continue
        route = str((recognition.get("routing") or {}).get("route") or "")
        if route not in routes:
            continue
        result = score_candidate(
            candidate,
            recognition,
            enrichments.get(candidate_id),
            config,
            scored_at=scored_at,
        )
        validate_record(result, scoring_schema, "Scoring result")
        results.append(result)

    results.sort(key=selection_sort_key)
    done = completed_ids(history)
    eligible = [
        result
        for result in results
        if result["selection_eligible"] and result["candidate_id"] not in done
    ]
    limits = config["limits"]
    after_scoring = eligible[: int(limits["max_candidates_after_scoring"])]
    llm_candidates = after_scoring[: int(limits["max_candidates_to_llm"])]
    summary_limit = int(limits["max_daily_summaries"])
    queue: list[dict[str, Any]] = []
    for index, result in enumerate(llm_candidates):
        status = "summary_slot" if index < summary_limit else "llm_candidate_only"
        candidate_id = result["candidate_id"]
        entry = queue_entry(
            result,
            candidates[candidate_id],
            enrichments.get(candidate_id),
            selection_status=status,
        )
        validate_record(entry, queue_schema, "LLM candidate")
        queue.append(entry)

    summary_ids = {
        entry["candidate_id"]
        for entry in queue
        if entry["selection_status"] == "summary_slot"
    }
    mandatory_pending = [
        result["candidate_id"]
        for result in results
        if result["mandatory"] and result["candidate_id"] not in done
    ]
    carry_forward = [
        candidate_id
        for candidate_id in mandatory_pending
        if candidate_id not in summary_ids
    ]
    decision_counts = Counter(result["decision"] for result in results)
    manifest = {
        "schema_version": 1,
        "scoring_version": int(config["scoring_version"]),
        "built_at": isoformat(scored_at),
        "source_registry_sha256": file_digest(registry_path),
        "recognition_results_sha256": file_digest(recognition_path),
        "enriched_registry_sha256": file_digest(enriched_path),
        "scored_candidate_count": len(results),
        "decision_counts": {
            decision: decision_counts.get(decision, 0)
            for decision in DECISION_ORDER
        },
        "completed_candidate_count": len(done),
        "eligible_candidate_count": len(eligible),
        "after_scoring_candidate_count": len(after_scoring),
        "llm_candidate_count": len(queue),
        "summary_slot_count": len(summary_ids),
        "mandatory_pending_count": len(mandatory_pending),
        "mandatory_carry_forward_ids": carry_forward,
        "llm_enabled": bool(config["policy"].get("llm_enabled", False)),
        "email_enabled": bool(config["policy"].get("email_enabled", False)),
    }

    atomic_write(scoring_path, jsonl(results))
    atomic_write(queue_path, jsonl(queue))
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if not history_path.exists():
        atomic_write(
            history_path,
            json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score routed research candidates and build a budgeted LLM queue"
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("runtime-state/data/paper_registry.jsonl"),
    )
    parser.add_argument(
        "--enriched-path",
        type=Path,
        default=Path("runtime-state/data/enriched_paper_registry.jsonl"),
    )
    parser.add_argument(
        "--recognition-path",
        type=Path,
        default=Path("runtime-state/data/recognition_results.jsonl"),
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("runtime-state/state/summary_history.json"),
    )
    parser.add_argument(
        "--scoring-path",
        type=Path,
        default=Path("runtime-state/data/scoring_results.jsonl"),
    )
    parser.add_argument(
        "--queue-path",
        type=Path,
        default=Path("runtime-state/data/queues/llm_candidate_queue.jsonl"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("runtime-state/state/selection_manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/scoring.yaml"),
    )
    parser.add_argument(
        "--scoring-schema",
        type=Path,
        default=Path("schemas/scoring_result.schema.json"),
    )
    parser.add_argument(
        "--queue-schema",
        type=Path,
        default=Path("schemas/llm_candidate.schema.json"),
    )
    args = parser.parse_args()
    manifest = score_registry(
        registry_path=args.registry_path,
        enriched_path=args.enriched_path,
        recognition_path=args.recognition_path,
        history_path=args.history_path,
        scoring_path=args.scoring_path,
        queue_path=args.queue_path,
        manifest_path=args.manifest_path,
        config_path=args.config,
        scoring_schema_path=args.scoring_schema,
        queue_schema_path=args.queue_schema,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "scored_candidate_count": manifest["scored_candidate_count"],
                "decision_counts": manifest["decision_counts"],
                "llm_candidate_count": manifest["llm_candidate_count"],
                "summary_slot_count": manifest["summary_slot_count"],
                "mandatory_pending_count": manifest["mandatory_pending_count"],
                "mandatory_carry_forward_count": len(
                    manifest["mandatory_carry_forward_ids"]
                ),
                "llm_enabled": manifest["llm_enabled"],
                "email_enabled": manifest["email_enabled"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

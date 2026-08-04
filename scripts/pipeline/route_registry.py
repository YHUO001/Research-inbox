from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.classify.recognition import (
    CLASSIFIER_VERSION,
    ROUTES,
    classify_candidate,
    load_yaml,
    validate_result,
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


def registry_digest(path: Path) -> str:
    if not path.exists():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enforce_source_scoped_overrides(
    candidate: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Keep Scholar-only mandatory policies from leaking to discovery sources."""
    source_type = str((candidate.get("source") or {}).get("source_type") or "")
    reasons = set((result.get("routing") or {}).get("reasons") or [])
    if source_type == "google_scholar_email" or "optical_zo_project_override" not in reasons:
        return result

    parse_state = str((candidate.get("parse_status") or {}).get("state") or "partial")
    venue_missing = not (candidate.get("venue") or {}).get("normalized")
    if parse_state == "partial" or venue_missing:
        replacement_reasons = ["missing_metadata", "optical_zo_discovery"]
        if parse_state == "partial":
            replacement_reasons.append("parser_partial")
        if venue_missing:
            replacement_reasons.append("unresolved_venue")
        result["routing"] = {
            "route": "metadata_enrichment_queue",
            "priority": "high",
            "mandatory": False,
            "reasons": sorted(set(replacement_reasons)),
            "requires_semantic_scoring": False,
            "requires_manual_review": False,
            "overflow_action": None,
        }
        return result

    confirmed = any(
        project.get("confidence") == "confirmed"
        for project in result.get("matched_projects") or []
    )
    result["routing"] = {
        "route": "standard_scoring_queue",
        "priority": "high" if confirmed else "normal",
        "mandatory": False,
        "reasons": ["optical_zo_discovery", "project_match"],
        "requires_semantic_scoring": True,
        "requires_manual_review": False,
        "overflow_action": None,
    }
    return result


def queue_entry(
    candidate: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "route": result["routing"]["route"],
        "priority": result["routing"]["priority"],
        "mandatory": result["routing"]["mandatory"],
        "reasons": result["routing"]["reasons"],
        "matched_projects": [
            {
                "project_id": project["project_id"],
                "confidence": project["confidence"],
            }
            for project in result["matched_projects"]
        ],
    }


def rebuild_routes(
    *,
    registry_path: Path,
    recognition_path: Path,
    queue_dir: Path,
    manifest_path: Path,
    rules_path: Path,
    venues_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    candidates = load_jsonl(registry_path)
    rules = load_yaml(rules_path)
    venues = load_yaml(venues_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    queues: dict[str, list[dict[str, Any]]] = {route: [] for route in ROUTES}

    seen_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("Every registry record must have candidate_id")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id in registry: {candidate_id}")
        seen_ids.add(candidate_id)

        result = classify_candidate(
            candidate,
            recognition_config=rules,
            venues_config=venues,
        )
        result = enforce_source_scoped_overrides(candidate, result)
        validate_result(result, schema)
        results.append(result)
        queues[result["routing"]["route"]].append(queue_entry(candidate, result))

    atomic_write(recognition_path, jsonl(results))
    for route, entries in queues.items():
        atomic_write(queue_dir / f"{route}.jsonl", jsonl(entries))

    counts = Counter(result["routing"]["route"] for result in results)
    project_counts = Counter(
        project["project_id"]
        for result in results
        for project in result["matched_projects"]
    )
    source_counts = Counter(
        str((candidate.get("source") or {}).get("source_type") or "unknown")
        for candidate in candidates
    )
    timestamps = [
        str(result["classified_at"])
        for result in results
        if result.get("classified_at")
    ]
    manifest = {
        "schema_version": 1,
        "classifier_version": CLASSIFIER_VERSION,
        "source_registry_sha256": registry_digest(registry_path),
        "classified_at": max(timestamps) if timestamps else None,
        "candidate_count": len(candidates),
        "route_counts": {route: counts.get(route, 0) for route in ROUTES},
        "project_counts": dict(sorted(project_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild deterministic recognition results and routing queues"
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("runtime-state/data/paper_registry.jsonl"),
    )
    parser.add_argument(
        "--recognition-path",
        type=Path,
        default=Path("runtime-state/data/recognition_results.jsonl"),
    )
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=Path("runtime-state/data/queues"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("runtime-state/state/routing_manifest.json"),
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("config/recognition_rules.yaml"),
    )
    parser.add_argument(
        "--venues",
        type=Path,
        default=Path("config/venues.yaml"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/recognition_result.schema.json"),
    )
    args = parser.parse_args()

    manifest = rebuild_routes(
        registry_path=args.registry_path,
        recognition_path=args.recognition_path,
        queue_dir=args.queue_dir,
        manifest_path=args.manifest_path,
        rules_path=args.rules,
        venues_path=args.venues,
        schema_path=args.schema,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "candidate_count": manifest["candidate_count"],
                "route_counts": manifest["route_counts"],
                "project_counts": manifest["project_counts"],
                "source_counts": manifest["source_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

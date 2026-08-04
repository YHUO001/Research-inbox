from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.classify.recognition import classify_candidate, load_yaml
from scripts.discovery.openalex_discovery import (
    QuerySpec,
    isoformat,
    load_json,
    load_queries,
    make_candidate,
    parse_datetime,
)
from scripts.enrich.metadata import HttpRequestError, JsonHttpClient, OpenAlexProvider


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def audit_window(
    state: dict[str, Any], now: datetime, config: dict[str, Any]
) -> tuple[str, str]:
    start = state.get("last_window_start")
    end = state.get("last_window_end")
    if start and end:
        return str(start), str(end)
    schedule = config["schedule"]
    previous = parse_datetime(state.get("last_successful_run_at"))
    start_date = (
        previous - timedelta(days=int(schedule["overlap_days"]))
        if previous
        else now - timedelta(days=int(schedule["initial_lookback_days"]))
    )
    return start_date.date().isoformat(), now.date().isoformat()


def filtered_sample(
    candidate: dict[str, Any],
    *,
    query: QuerySpec,
    matched_projects: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "query_id": query.query_id,
        "expected_project_id": query.project_id,
        "title": str(candidate.get("title") or ""),
        "year": candidate.get("year"),
        "venue": (candidate.get("venue") or {}).get("normalized"),
        "matched_projects": sorted(
            {
                str(project.get("project_id"))
                for project in matched_projects
                if project.get("project_id")
            }
        ),
        "reason": (
            "no_project_match"
            if not matched_projects
            else "expected_project_not_matched"
        ),
    }


def audit_filters(
    *,
    config_path: Path,
    state_path: Path,
    output_path: Path,
    recognition_path: Path,
    venues_path: Path,
    api_key: str | None = None,
    client: JsonHttpClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("OpenAlex discovery config must be a YAML object")
    provider = config["provider"]
    api_key = api_key or os.environ.get(str(provider["api_key_env"]))
    if not api_key:
        raise RuntimeError("Missing required OpenAlex API key")
    client = client or JsonHttpClient(
        user_agent=str(provider["user_agent"]),
        timeout_seconds=float(provider["timeout_seconds"]),
        max_attempts=int(provider["max_attempts"]),
        min_interval_seconds=float(provider["min_interval_seconds"]),
    )
    state = load_json(state_path, {})
    start_date, end_date = audit_window(state, now, config)
    recognition = load_yaml(recognition_path)
    venues = load_yaml(venues_path)
    queries = load_queries(config)
    limits = config["limits"]
    sample_limit = int(
        (config.get("observability") or {}).get("maximum_filtered_samples", 20)
    )
    maximum_per_query = int(limits["maximum_results_per_query"])
    maximum_abstract = int(limits["maximum_abstract_characters"])

    samples: list[dict[str, Any]] = []
    raw_count = 0
    filtered_count = 0
    matched_count = 0
    invalid_count = 0
    query_error_count = 0
    query_summaries: list[dict[str, Any]] = []

    for query in queries:
        query_raw = 0
        query_filtered = 0
        query_matched = 0
        query_invalid = 0
        query_error: str | None = None
        try:
            payload = client.get_json(
                str(provider["base_url"]),
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
            query_error_count += 1
            query_error = error.reason

        for rank, work in enumerate(results):
            raw_count += 1
            query_raw += 1
            if not isinstance(work, dict):
                invalid_count += 1
                query_invalid += 1
                continue
            try:
                candidate = make_candidate(
                    work,
                    query=query,
                    rank=rank,
                    discovered_at=isoformat(now),
                    maximum_abstract=maximum_abstract,
                )
            except (TypeError, ValueError):
                invalid_count += 1
                query_invalid += 1
                continue
            result = classify_candidate(
                candidate,
                recognition_config=recognition,
                venues_config=venues,
                classified_at=isoformat(now),
            )
            projects = result.get("matched_projects") or []
            project_ids = {
                str(project.get("project_id"))
                for project in projects
                if project.get("project_id")
            }
            if query.project_id in project_ids:
                matched_count += 1
                query_matched += 1
                continue
            filtered_count += 1
            query_filtered += 1
            if len(samples) < sample_limit:
                samples.append(
                    filtered_sample(
                        candidate,
                        query=query,
                        matched_projects=projects,
                    )
                )

        query_summaries.append(
            {
                "query_id": query.query_id,
                "project_id": query.project_id,
                "raw_results": query_raw,
                "matched": query_matched,
                "filtered": query_filtered,
                "invalid": query_invalid,
                "error": query_error,
            }
        )

    manifest = {
        "schema_version": 1,
        "audit_version": 1,
        "built_at": isoformat(now),
        "window_start": start_date,
        "window_end": end_date,
        "raw_result_count": raw_count,
        "matched_count": matched_count,
        "filtered_count": filtered_count,
        "invalid_count": invalid_count,
        "query_error_count": query_error_count,
        "sample_limit": sample_limit,
        "sample_count": len(samples),
        "filtered_samples": samples,
        "query_summaries": query_summaries,
        "raw_provider_responses_persisted": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit OpenAlex deterministic filtering without mutating the registry"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/openalex_discovery.yaml"),
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("runtime-state/state/openalex_discovery_state.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("runtime-state/state/openalex_filter_audit.json"),
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
    args = parser.parse_args()
    manifest = audit_filters(
        config_path=args.config,
        state_path=args.state_path,
        output_path=args.output_path,
        recognition_path=args.recognition,
        venues_path=args.venues,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "raw_result_count": manifest["raw_result_count"],
                "matched_count": manifest["matched_count"],
                "filtered_count": manifest["filtered_count"],
                "sample_count": manifest["sample_count"],
                "query_error_count": manifest["query_error_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

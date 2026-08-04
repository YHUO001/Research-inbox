from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.pipeline.score_registry import score_registry


ROOT = Path(__file__).resolve().parents[1]


def candidate(
    candidate_id: str,
    *,
    title: str,
    source_type: str = "google_scholar_email",
    year: int = 2026,
    snippet: str = "A photonic neural network with forward-only training.",
) -> dict:
    source = (
        {
            "source_type": "google_scholar_email",
            "message_id": f"message-{candidate_id}",
            "received_at": "2026-08-03T00:00:00Z",
        }
        if source_type == "google_scholar_email"
        else {
            "source_type": "openalex",
            "work_id": f"https://openalex.org/{candidate_id}",
            "discovered_at": "2026-08-03T00:00:00Z",
            "publication_date": "2026-08-01",
        }
    )
    return {
        "candidate_id": candidate_id,
        "source": source,
        "title": title,
        "year": year,
        "snippet": snippet,
        "authors": [{"name": "Alice Researcher"}],
        "venue": {"raw": "Optica", "normalized": "Optica"},
        "identifiers": {
            "doi": {"value": f"10.1000/{candidate_id}"},
            "arxiv_id": {"value": None},
            "pmid": {"value": None},
        },
        "links": {"primary_url": f"https://doi.org/10.1000/{candidate_id}"},
    }


def recognition(
    candidate_id: str,
    *,
    route: str,
    mandatory: bool,
    confidence: str = "confirmed",
    priority_feature: bool = False,
) -> dict:
    features = (
        [{"feature": "query_efficiency", "weight": 1.0, "evidence": []}]
        if priority_feature
        else []
    )
    return {
        "candidate_id": candidate_id,
        "matched_projects": [
            {
                "project_id": "optical-neural-networks",
                "confidence": confidence,
                "positive_evidence": [
                    {
                        "field": "title",
                        "pattern": "optical",
                        "matched_text": "Optical Neural Network",
                    }
                ],
                "negative_evidence": [],
                "priority_features": features,
            }
        ],
        "venue_policy": {
            "matched_tier": (
                "tier_1_must_summarize"
                if mandatory
                else "tier_2_relevance_gated"
            )
        },
        "routing": {
            "route": route,
            "mandatory": mandatory,
        },
    }


def enriched(candidate_id: str, *, abstract: bool = True) -> dict:
    return {
        "candidate_id": candidate_id,
        "fields": {
            "title": {"value": f"Optical Neural Network {candidate_id}"},
            "publication_date": {"value": "2026-08-01"},
            "year": {"value": 2026},
            "venue": {"value": "Optica"},
            "doi": {"value": f"10.1000/{candidate_id}"},
            "openalex_id": {"value": f"https://openalex.org/{candidate_id}"},
            "abstract": {
                "value": (
                    "A diffractive photonic architecture with forward-only training."
                    if abstract
                    else None
                )
            },
            "landing_page": {
                "value": f"https://doi.org/10.1000/{candidate_id}"
            },
            "open_access_url": {"value": None},
        },
        "authors": [{"name": "Alice Researcher"}],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def run_score(
    tmp_path: Path, history: dict | None = None
) -> tuple[dict, list[dict], list[dict]]:
    registry = tmp_path / "registry.jsonl"
    recognition_path = tmp_path / "recognition.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    history_path = tmp_path / "history.json"
    scoring_path = tmp_path / "scoring.jsonl"
    queue_path = tmp_path / "queue.jsonl"
    manifest_path = tmp_path / "manifest.json"

    records = [
        candidate("mandatory-a", title="Optical Neural Network Training"),
        candidate("mandatory-b", title="Optical Neural Network Architecture"),
        candidate("mandatory-c", title="Optical Neural Network Calibration"),
        candidate(
            "standard-high",
            title="Forward-Only Diffractive Optical Neural Network",
        ),
        candidate(
            "standard-low",
            title="Optical Neural Network Overview",
            snippet="A review and perspective on optical neural networks.",
        ),
    ]
    recognitions = [
        recognition("mandatory-a", route="mandatory_summary_queue", mandatory=True),
        recognition("mandatory-b", route="mandatory_summary_queue", mandatory=True),
        recognition("mandatory-c", route="mandatory_summary_queue", mandatory=True),
        recognition(
            "standard-high",
            route="standard_scoring_queue",
            mandatory=False,
            priority_feature=True,
        ),
        recognition(
            "standard-low",
            route="standard_scoring_queue",
            mandatory=False,
            confidence="probable",
        ),
    ]
    enrichments = [
        enriched(
            record["candidate_id"],
            abstract=record["candidate_id"] != "standard-low",
        )
        for record in records
    ]

    write_jsonl(registry, records)
    write_jsonl(recognition_path, recognitions)
    write_jsonl(enriched_path, enrichments)
    if history is not None:
        history_path.write_text(json.dumps(history), encoding="utf-8")

    manifest = score_registry(
        registry_path=registry,
        enriched_path=enriched_path,
        recognition_path=recognition_path,
        history_path=history_path,
        scoring_path=scoring_path,
        queue_path=queue_path,
        manifest_path=manifest_path,
        config_path=ROOT / "config" / "scoring.yaml",
        scoring_schema_path=ROOT / "schemas" / "scoring_result.schema.json",
        queue_schema_path=ROOT / "schemas" / "llm_candidate.schema.json",
        scored_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
    )
    scores = [
        json.loads(line)
        for line in scoring_path.read_text(encoding="utf-8").splitlines()
    ]
    queue = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
    ]
    return manifest, scores, queue


def test_mandatory_candidates_sort_first_and_budget_is_enforced(
    tmp_path: Path,
) -> None:
    manifest, scores, queue = run_score(tmp_path)
    assert manifest["scored_candidate_count"] == 5
    assert manifest["summary_slot_count"] == 3
    assert manifest["llm_candidate_count"] <= 5
    assert [item["candidate_id"] for item in queue[:3]] == [
        "mandatory-a",
        "mandatory-b",
        "mandatory-c",
    ]
    assert all(item["selection_status"] == "summary_slot" for item in queue[:3])
    assert manifest["mandatory_carry_forward_ids"] == []
    assert scores[0]["decision"] == "mandatory"


def test_score_is_explainable_and_review_penalty_is_applied(
    tmp_path: Path,
) -> None:
    _, scores, _ = run_score(tmp_path)
    by_id = {item["candidate_id"]: item for item in scores}
    high = by_id["standard-high"]
    low = by_id["standard-low"]
    assert high["score"] > low["score"]
    assert high["selection_eligible"] is True
    assert any(
        item["feature"] == "priority_features"
        for item in high["breakdown"]
    )
    assert any(
        item["feature"] == "review_or_perspective_penalty"
        and item["contribution"] < 0
        for item in low["breakdown"]
    )


def test_completed_summary_is_excluded_from_selection(tmp_path: Path) -> None:
    history = {
        "schema_version": 1,
        "completed_candidate_ids": {
            "mandatory-a": {"completed_at": "2026-08-03T00:00:00Z"}
        },
        "failed_candidate_ids": {},
    }
    manifest, _, queue = run_score(tmp_path, history)
    selected = {item["candidate_id"] for item in queue}
    assert "mandatory-a" not in selected
    assert manifest["completed_candidate_count"] == 1
    assert manifest["mandatory_pending_count"] == 2


def test_fixed_time_rebuild_is_byte_stable(tmp_path: Path) -> None:
    run_score(tmp_path)
    first_scoring = (tmp_path / "scoring.jsonl").read_bytes()
    first_queue = (tmp_path / "queue.jsonl").read_bytes()
    first_manifest = (tmp_path / "manifest.json").read_bytes()
    run_score(tmp_path)
    assert (tmp_path / "scoring.jsonl").read_bytes() == first_scoring
    assert (tmp_path / "queue.jsonl").read_bytes() == first_queue
    assert (tmp_path / "manifest.json").read_bytes() == first_manifest

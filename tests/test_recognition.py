from __future__ import annotations

import json
from pathlib import Path

from scripts.classify.recognition import (
    classify_candidate,
    load_yaml,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
RULES = load_yaml(ROOT / "config" / "recognition_rules.yaml")
VENUES = load_yaml(ROOT / "config" / "venues.yaml")
SCHEMA = json.loads(
    (ROOT / "schemas" / "recognition_result.schema.json").read_text(
        encoding="utf-8"
    )
)


def candidate(
    *,
    title: str,
    snippet: str = "",
    venue: str = "",
    parse_state: str = "complete",
    candidate_id: str = "candidate-001",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "title": title,
        "snippet": snippet or None,
        "venue": {
            "raw": venue or None,
            "normalized": venue or None,
        },
        "raw_metadata_line": None,
        "authors": [],
        "parse_status": {
            "state": parse_state,
            "warnings": [],
        },
        "source": {
            "source_type": "google_scholar_email",
        },
        "extracted_at": "2026-08-04T00:00:00Z",
    }


def classify(record: dict) -> dict:
    result = classify_candidate(
        record,
        recognition_config=RULES,
        venues_config=VENUES,
    )
    validate_result(result, SCHEMA)
    return result


def test_tier_one_alert_is_mandatory_even_without_project_match() -> None:
    result = classify(
        candidate(
            title="65 TOPS optoelectronic multi-core computing",
            venue="Nature Communications",
        )
    )
    assert result["matched_projects"] == []
    assert result["routing"]["route"] == "mandatory_summary_queue"
    assert result["routing"]["reasons"] == ["tier_1_alert_venue"]


def test_truncated_light_venue_alias_is_resolved() -> None:
    result = classify(
        candidate(
            title="Bio-inspired training for optical neural networks",
            venue="Light: Science & …",
        )
    )
    assert result["venue_policy"]["matched_venue"] == (
        "Light: Science & Applications"
    )
    assert result["routing"]["route"] == "mandatory_summary_queue"


def test_optical_zo_is_mandatory() -> None:
    result = classify(
        candidate(
            title=(
                "Layered-Parameter Perturbation for Zeroth-Order "
                "Optimization of Optical Neural Networks"
            ),
            venue="AAAI",
        )
    )
    projects = {item["project_id"] for item in result["matched_projects"]}
    assert projects == {
        "zeroth-order-optimization",
        "optical-neural-networks",
    }
    assert result["routing"]["route"] == "mandatory_summary_queue"
    assert result["routing"]["reasons"] == [
        "optical_zo_project_override"
    ]


def test_llm_zo_query_and_low_rank_features_are_prioritized() -> None:
    result = classify(
        candidate(
            title="Low-rank MeZO fine-tuning for language models",
            snippet=(
                "A query-efficient method with fewer queries and "
                "a Bayesian subspace gradient."
            ),
            venue="NeurIPS",
        )
    )
    project = result["matched_projects"][0]
    features = {item["feature"] for item in project["priority_features"]}
    assert project["project_id"] == "zeroth-order-optimization"
    assert {"query_efficiency", "low_rank_or_subspace"} <= features
    assert result["routing"]["route"] == "standard_scoring_queue"


def test_per_step_two_query_claim_is_not_rewarded_as_total_reduction() -> None:
    result = classify(
        candidate(
            title="Two-query zeroth-order optimization",
            snippet="A two-query estimator is used at every optimization step.",
            venue="Workshop",
        )
    )
    project = result["matched_projects"][0]
    assert project["priority_features"] == []
    assert (
        "per_step_query_count_not_total_query_reduction"
        in result["classifier_warnings"]
    )


def test_optical_flow_and_zero_order_hold_are_archived() -> None:
    optical_flow = classify(
        candidate(
            title="Optical flow neural network for video",
            venue="CVPR",
        )
    )
    zero_hold = classify(
        candidate(
            title="Zero-order hold circuits",
            venue="IEEE Transactions",
            candidate_id="candidate-002",
        )
    )
    assert optical_flow["routing"]["route"] == "archive"
    assert zero_hold["routing"]["route"] == "archive"


def test_manual_review_parser_state_precedes_tier_one_policy() -> None:
    result = classify(
        candidate(
            title="MASTER OF SCIENCE all rights reserved",
            venue="Nature",
            parse_state="manual_review",
        )
    )
    assert result["routing"]["route"] == "manual_review_queue"
    assert result["routing"]["mandatory"] is False


def test_relevant_candidate_without_venue_goes_to_enrichment() -> None:
    result = classify(
        candidate(
            title="Robust optical neural network training",
            parse_state="partial",
        )
    )
    assert result["routing"]["route"] == "metadata_enrichment_queue"
    assert "unresolved_venue" in result["routing"]["reasons"]

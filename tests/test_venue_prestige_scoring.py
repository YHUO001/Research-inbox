from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts.pipeline.score_registry import load_yaml, score_candidate, venue_prestige


ROOT = Path(__file__).resolve().parents[1]


def _candidate(*, venue: str, source_type: str = "google_scholar_email") -> dict:
    source = (
        {
            "source_type": "google_scholar_email",
            "message_id": "science-adv-alert",
            "received_at": "2026-08-25T02:40:05Z",
        }
        if source_type == "google_scholar_email"
        else {
            "source_type": "openalex",
            "work_id": "https://openalex.org/W7203903045",
            "discovered_at": "2026-08-25T02:40:05Z",
            "publication_date": "2026-08-21",
        }
    )
    return {
        "candidate_id": "science-adv-in-sensor",
        "source": source,
        "title": "End-to-end all-optical in-sensor computing system using photonic integrated circuits",
        "year": 2026,
        "snippet": (
            "The architecture performs photonic neural network inference for "
            "classification and sensing using photonic integrated circuits."
        ),
        "authors": [{"name": "Z Xiao"}],
        "venue": {"raw": venue, "normalized": venue},
        "identifiers": {
            "doi": {"value": "10.1126/sciadv.aef8657"},
            "arxiv_id": {"value": None},
            "pmid": {"value": None},
        },
        "links": {
            "primary_url": "https://doi.org/10.1126/sciadv.aef8657"
        },
    }


def _recognition() -> dict:
    return {
        "candidate_id": "science-adv-in-sensor",
        "matched_projects": [
            {
                "project_id": "optical-neural-networks",
                "confidence": "probable",
                "positive_evidence": [
                    {
                        "field": "snippet",
                        "pattern": "photonic neural network",
                        "matched_text": "photonic neural network",
                    },
                    {
                        "field": "snippet",
                        "pattern": "neural network",
                        "matched_text": "neural network",
                    },
                    {
                        "field": "snippet",
                        "pattern": "inference",
                        "matched_text": "inference",
                    },
                ],
                "negative_evidence": [],
                "priority_features": [],
            }
        ],
        # Deliberately leave the old policy as tier_3. Prestige scoring must not
        # depend on this routing tier.
        "venue_policy": {
            "matched_tier": "tier_3_and_unlisted",
            "matched_venue": "Science Advances",
            "policy_action": "summarize_only_when_exceptionally_relevant",
        },
        "routing": {
            "route": "standard_scoring_queue",
            "mandatory": False,
        },
    }


def _enriched(*, venue: str) -> dict:
    return {
        "candidate_id": "science-adv-in-sensor",
        "fields": {
            "title": {
                "value": "End-to-end all-optical in-sensor computing system using photonic integrated circuits"
            },
            "publication_date": {"value": "2026-08-21"},
            "year": {"value": 2026},
            "venue": {"value": venue},
            "doi": {"value": "10.1126/sciadv.aef8657"},
            "openalex_id": {"value": "https://openalex.org/W7203903045"},
            "abstract": {
                "value": (
                    "The architecture performs photonic neural network inference "
                    "for classification and sensing using photonic integrated circuits."
                )
            },
            "landing_page": {
                "value": "https://doi.org/10.1126/sciadv.aef8657"
            },
            "open_access_url": {"value": None},
        },
        "authors": [{"name": "Z Xiao"}],
    }


def _config() -> dict:
    return load_yaml(ROOT / "config" / "scoring.yaml")


def _prestige_contribution(result: dict) -> dict:
    return next(
        item for item in result["breakdown"] if item["feature"] == "venue_prestige"
    )


def test_science_advances_is_high_impact_without_becoming_mandatory() -> None:
    config = _config()
    result = score_candidate(
        _candidate(venue="Science Advances"),
        _recognition(),
        _enriched(venue="Science Advances"),
        config,
        scored_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    prestige = _prestige_contribution(result)
    assert prestige["contribution"] == 0.13
    assert "high_impact" in prestige["evidence"]
    assert result["mandatory"] is False
    assert result["score"] == 0.74
    assert result["decision"] == "summarize"
    assert result["selection_eligible"] is True


def test_same_relevance_in_unlisted_venue_stays_below_summary_threshold() -> None:
    config = _config()
    result = score_candidate(
        _candidate(venue="Example Unlisted Journal"),
        _recognition(),
        _enriched(venue="Example Unlisted Journal"),
        config,
        scored_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    prestige = _prestige_contribution(result)
    assert prestige["contribution"] == 0.02
    assert "general_unlisted" in prestige["evidence"]
    assert result["score"] == 0.63
    assert result["decision"] == "retain_without_summary"
    assert result["selection_eligible"] is False


def test_prestige_matching_is_independent_of_discovery_source_and_html_entities() -> None:
    config = _config()

    tier, venue, weight = venue_prestige("Science Advances", config)
    assert (tier, venue, weight) == ("high_impact", "Science Advances", 0.13)

    tier, venue, weight = venue_prestige("Light: Science &amp; Applications", config)
    assert (tier, venue, weight) == (
        "flagship",
        "Light: Science & Applications",
        0.14,
    )

    openalex_result = score_candidate(
        _candidate(venue="Science Advances", source_type="openalex"),
        _recognition(),
        _enriched(venue="Science Advances"),
        config,
        scored_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    assert _prestige_contribution(openalex_result)["contribution"] == 0.13

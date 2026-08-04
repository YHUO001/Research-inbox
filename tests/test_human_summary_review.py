from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize.build_review_packet import build_review_packet
from scripts.summarize.finalize_review import finalize_review


ROOT = Path(__file__).resolve().parents[1]


def source_request(candidate_id: str) -> dict:
    abstract = (
        "An all-optical recurrent neural network realizes linear operations, nonlinear "
        "functions, and memory in the optical domain up to 80 GHz. It analyzes soliton "
        "states from integrated optical microresonators."
    )
    return {
        "schema_version": 1,
        "request_version": 1,
        "request_id": "request-12345678",
        "candidate_id": candidate_id,
        "prepared_at": "2026-08-04T13:23:52Z",
        "provider_status": "not_configured",
        "selection_status": "summary_slot",
        "summary_schema": "paper_summary.schema.json",
        "source": {
            "title": "All-optical computing towards 100-GHz clock rates",
            "authors": ["A. Researcher"],
            "venue": "Light Science & Applications",
            "year": 2026,
            "source_type": "google_scholar_email",
            "doi": "10.1000/example",
            "openalex_id": None,
            "landing_page": "https://doi.org/10.1000/example",
            "open_access_url": None,
            "abstract": abstract,
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.85,
            "decision": "mandatory",
            "score_breakdown": [],
        },
        "instructions": ["Classify architecture only from explicit evidence."],
        "prompt": "Return one grounded JSON summary.",
    }


def generated_summary(candidate_id: str) -> dict:
    return {
        "schema_version": 1,
        "summary_version": 1,
        "candidate_id": candidate_id,
        "core_problem": "The work addresses electronic clock-rate bottlenecks.",
        "method_and_architecture": "An all-optical recurrent neural network is demonstrated.",
        "main_contributions": ["The work reports all-optical recurrent processing up to 80 GHz."],
        "reported_results": [
            {
                "claim": "The system operates up to 80 GHz.",
                "reported_by_authors": True,
                "basis": "abstract",
            }
        ],
        "distinction_from_prior_work": "It avoids an electronic processing bottleneck.",
        "research_value": "It is relevant to ultrafast optical computing.",
        "limitations_and_open_questions": ["Energy efficiency is not_available."],
        "optical_neural_network_analysis": {
            "architecture_type": "free_space",
            "training_method": "not_available",
            "optical_nonlinearity": "Nonlinear optical functions are reported without implementation detail.",
            "calibration_requirements": "not_available",
            "application_tasks": ["soliton-state analysis"],
            "hardware_validation": "physical_experiment",
        },
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_and_abstract_only",
            "unsupported_numbers_detected": False,
            "missing_information": ["full_text"],
        },
    }


def prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate_id = "candidate-review-123"
    data = tmp_path / "data"
    request_path = data / "summary_requests/2026-08-04.jsonl"
    summary_path = data / "summaries/2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    summary_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(source_request(candidate_id)) + "\n")
    summary_path.write_text(json.dumps(generated_summary(candidate_id)) + "\n")
    manifest_path = tmp_path / "state/summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "digest_date": "2026-08-04",
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "email_enabled": False,
                "summary_history_updated": False,
            }
        )
    )
    review_manifest = tmp_path / "state/summary_review_manifest.json"
    return manifest_path, review_manifest, tmp_path / "state/summary_history.json"


def build(tmp_path: Path) -> tuple[dict, Path, Path]:
    manifest, review_manifest, history = prepare(tmp_path)
    state = build_review_packet(
        generation_manifest_path=manifest,
        summary_schema_path=ROOT / "schemas/paper_summary.schema.json",
        output_root=tmp_path / "data",
        review_manifest_path=review_manifest,
    )
    return state, review_manifest, history


def test_review_packet_repairs_unsupported_architecture_and_exposes_rubric(tmp_path: Path) -> None:
    state, review_manifest, _ = build(tmp_path)
    assert state["status"] == "pending_human_review"
    assert state["architecture_repairs"] == 1
    packet = json.loads((tmp_path / "data/reviews/2026-08-04.review.json").read_text())
    paper = packet["papers"][0]
    assert paper["summary"]["optical_neural_network_analysis"]["architecture_type"] == "unclear"
    assert paper["automated_checks"]["architecture_consistent"] is True
    assert paper["automated_checks"]["architecture_repaired"] is True
    markdown = (tmp_path / "data/reviews/2026-08-04.review.md").read_text()
    assert "Source abstract" in markdown
    assert "Factual accuracy" in markdown
    assert "No explicit architecture evidence" in markdown
    assert json.loads(review_manifest.read_text())["summary_history_updated"] is False


def test_hold_for_revision_does_not_update_history(tmp_path: Path) -> None:
    _, review_manifest, history = build(tmp_path)
    result = finalize_review(
        state_root=tmp_path,
        review_manifest_path=review_manifest,
        history_path=history,
        decision="hold_for_revision",
        confirmation="REVIEWED",
        reviewer="reviewer",
        reviewed_at="2026-08-04T15:30:00Z",
        notes="Architecture wording needs revision.",
    )
    assert result["status"] == "revision_requested"
    assert result["summary_history_updated"] is False
    assert not history.exists()


def test_approve_all_updates_history_and_keeps_email_disabled(tmp_path: Path) -> None:
    _, review_manifest, history = build(tmp_path)
    result = finalize_review(
        state_root=tmp_path,
        review_manifest_path=review_manifest,
        history_path=history,
        decision="approve_all",
        confirmation="REVIEWED",
        reviewer="reviewer",
        reviewed_at="2026-08-04T15:30:00Z",
        notes="Reviewed against abstract.",
    )
    assert result["status"] == "approved"
    assert result["summary_history_updated"] is True
    assert result["email_enabled"] is False
    completed = json.loads(history.read_text())["completed_candidate_ids"]
    assert "candidate-review-123" in completed
    digest = json.loads((tmp_path / "data/digests/2026-08-04.generated.json").read_text())
    assert digest["status"] == "approved_human_review"
    assert digest["safety"]["summary_history_updated"] is True
    assert digest["safety"]["email_enabled"] is False


def test_finalization_rejects_artifact_tampering(tmp_path: Path) -> None:
    _, review_manifest, history = build(tmp_path)
    summary_path = tmp_path / "data/summaries/2026-08-04.jsonl"
    summary_path.write_text(summary_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed after packet creation"):
        finalize_review(
            state_root=tmp_path,
            review_manifest_path=review_manifest,
            history_path=history,
            decision="approve_all",
            confirmation="REVIEWED",
            reviewer="reviewer",
            reviewed_at="2026-08-04T15:30:00Z",
        )

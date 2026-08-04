from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize.deepseek_provider import DeepSeekResponse
from scripts.summarize.generate_summaries_production import generate


ROOT = Path(__file__).resolve().parents[1]


def request(candidate_id: str, abstract: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt": "Return JSON using only the supplied abstract.",
        "source": {
            "title": "Optical neural computing hardware",
            "authors": ["Alice Researcher"],
            "venue": "Nature Communications",
            "year": 2026,
            "source_type": "google_scholar_email",
            "doi": "10.1000/example",
            "openalex_id": None,
            "landing_page": "https://doi.org/10.1000/example",
            "open_access_url": None,
            "abstract": abstract,
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.9,
            "decision": "mandatory",
            "score_breakdown": [],
        },
    }


def summary(candidate_id: str, claim: str) -> dict:
    return {
        "schema_version": 1,
        "summary_version": 1,
        "candidate_id": candidate_id,
        "core_problem": "The work addresses optical neural computation.",
        "method_and_architecture": "The abstract describes optical hardware.",
        "main_contributions": ["The work reports optical hardware validation."],
        "reported_results": [
            {
                "claim": claim,
                "reported_by_authors": True,
                "basis": "abstract",
            }
        ],
        "distinction_from_prior_work": "not_available",
        "research_value": "The work is relevant to optical neural hardware.",
        "limitations_and_open_questions": ["Long-term stability is not_available."],
        "optical_neural_network_analysis": {
            "architecture_type": "unclear",
            "training_method": "not_available",
            "optical_nonlinearity": "not_available",
            "calibration_requirements": "not_available",
            "application_tasks": [],
            "hardware_validation": "physical_experiment",
        },
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_and_abstract_only",
            "unsupported_numbers_detected": False,
            "missing_information": ["full_text"],
        },
    }


class FakeClient:
    def __init__(self, values: list[dict]) -> None:
        self.values = list(values)
        self.calls = 0

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        self.calls += 1
        value = self.values[min(self.calls - 1, len(self.values) - 1)]
        return DeepSeekResponse(
            content=json.dumps(value),
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
            model="deepseek-v4-pro",
        )


def prepare(tmp_path: Path, item: dict) -> tuple[Path, Path]:
    request_path = tmp_path / "data" / "summary_requests" / "2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "state" / "summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"digest_date": "2026-08-04", "request_file": str(request_path)}),
        encoding="utf-8",
    )
    return request_path, manifest_path


def test_production_injects_candidate_id_and_normalizes_equivalent_unit_glyphs(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate-expected"
    _, manifest_path = prepare(
        tmp_path,
        request(candidate_id, "The authors report a compute density of 5.16 TOPS/mm2."),
    )
    client = FakeClient(
        [summary("wrong-model-id", "The authors report 5.16 TOPS/mm² compute density.")]
    )

    state = generate(
        dry_run_manifest_path=manifest_path,
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        config_path=ROOT / "config" / "summary_generation.yaml",
        output_root=tmp_path / "data",
        manifest_path=manifest_path,
        api_key="test-key",
        client=client,
    )

    assert state["status"] == "completed"
    assert state["summary_count"] == 1
    assert state["transport_repairs"] == {
        "candidate_id_repair_responses": 1,
        "unit_format_normalization_responses": 1,
    }
    generated = json.loads(
        (tmp_path / "data" / "summaries" / "2026-08-04.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert generated["candidate_id"] == candidate_id
    assert "5.16 TOPS/mm2" in generated["reported_results"][0]["claim"]


def test_production_preserves_strict_approximation_semantics_and_diagnostics(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate-approximation"
    _, manifest_path = prepare(
        tmp_path,
        request(candidate_id, "Conventional processors remain at ~5 GHz."),
    )
    client = FakeClient(
        [
            summary(candidate_id, "Conventional processors remain at 5 GHz."),
            summary(candidate_id, "Conventional processors remain at 5 GHz."),
        ]
    )

    with pytest.raises(RuntimeError, match="failed local validation"):
        generate(
            dry_run_manifest_path=manifest_path,
            summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
            config_path=ROOT / "config" / "summary_generation.yaml",
            output_root=tmp_path / "data",
            manifest_path=manifest_path,
            api_key="test-key",
            client=client,
        )

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed_validation"
    assert persisted["failure_count"] == 1
    assert "unsupported numeric claims: 5ghz" in persisted["failures"][0]["reason"]
    assert persisted["transport_repairs"] == {
        "candidate_id_repair_responses": 0,
        "unit_format_normalization_responses": 0,
    }
    assert not (tmp_path / "data" / "summaries" / "2026-08-04.jsonl").exists()

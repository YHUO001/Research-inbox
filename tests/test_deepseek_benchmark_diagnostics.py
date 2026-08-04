from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.summarize.benchmark_deepseek_models_v2 import benchmark
from scripts.summarize.deepseek_provider import DeepSeekResponse


ROOT = Path(__file__).resolve().parents[1]


def request(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt": "Return JSON using only the supplied abstract.",
        "source": {
            "title": "Optical hardware for neural computation",
            "authors": ["Alice Researcher"],
            "venue": "Optica",
            "year": 2026,
            "source_type": "google_scholar_email",
            "doi": "10.1000/example",
            "openalex_id": None,
            "landing_page": "https://doi.org/10.1000/example",
            "open_access_url": None,
            "abstract": "The authors report experimental optical hardware validation.",
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.9,
            "decision": "mandatory",
            "score_breakdown": [],
        },
    }


def valid_summary(candidate_id: str) -> dict:
    return {
        "schema_version": 1,
        "summary_version": 1,
        "candidate_id": candidate_id,
        "core_problem": "The work addresses optical neural computation.",
        "method_and_architecture": "The abstract describes optical hardware.",
        "main_contributions": ["The work reports optical hardware validation."],
        "reported_results": [
            {
                "claim": "The authors report experimental optical hardware validation.",
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
    def __init__(self, model: str) -> None:
        self.model = model

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        match = re.search(r'"candidate_id":\s*"([^"]+)"', kwargs["system_prompt"])
        assert match
        candidate_id = match.group(1)
        content = (
            json.dumps(valid_summary(candidate_id))
            if self.model == "deepseek-v4-flash"
            else json.dumps({"candidate_id": "wrong-id"})
        )
        return DeepSeekResponse(
            content=content,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
            model=self.model,
        )


def test_partial_model_failure_preserves_successful_model_output(tmp_path: Path) -> None:
    request_path = tmp_path / "state" / "request.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request("candidate-isolated")) + "\n", encoding="utf-8")
    dry_manifest = tmp_path / "state" / "dry.json"
    dry_manifest.write_text(
        json.dumps({"digest_date": "2026-08-04", "request_file": str(request_path)}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "state" / "benchmark.json"

    manifest = benchmark(
        dry_run_manifest_path=dry_manifest,
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        config_path=ROOT / "config" / "summary_generation.yaml",
        output_root=tmp_path / "data",
        manifest_path=manifest_path,
        api_key="test-key",
        client_factory=lambda model: FakeClient(model),
    )

    assert manifest["status"] == "partial_failure"
    assert manifest["candidate_level_isolation"] is True
    assert manifest["models"]["flash"]["summary_count"] == 1
    assert manifest["models"]["flash"]["failure_count"] == 0
    assert manifest["models"]["pro"]["summary_count"] == 0
    assert manifest["models"]["pro"]["failure_count"] == 1
    assert manifest["models"]["pro"]["failures"][0]["candidate_id"] == "candidate-isolated"
    assert Path(manifest["models"]["flash"]["summary_file"]).exists()
    assert manifest_path.exists()


def test_workflow_uses_candidate_isolation_and_restores_dry_run_files() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "benchmark-deepseek-models.yml"
    ).read_text(encoding="utf-8")
    assert "benchmark_deepseek_models_v2" in workflow
    assert "git restore --" in workflow
    assert "state/deepseek_benchmark_manifest.json" in workflow

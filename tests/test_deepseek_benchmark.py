from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize.benchmark_deepseek_models import benchmark
from scripts.summarize.deepseek_provider import DeepSeekResponse


ROOT = Path(__file__).resolve().parents[1]


def valid_summary(candidate_id: str, model: str) -> dict:
    return {
        "schema_version": 1,
        "summary_version": 1,
        "candidate_id": candidate_id,
        "core_problem": "The paper addresses efficient optical neural computation.",
        "method_and_architecture": "The abstract reports a hybrid optical-electronic architecture.",
        "main_contributions": [f"The {model} output identifies the reported hardware method."],
        "reported_results": [
            {
                "claim": "The authors report experimental optical hardware validation.",
                "reported_by_authors": True,
                "basis": "abstract",
            }
        ],
        "distinction_from_prior_work": "The abstract emphasizes a hardware-oriented implementation.",
        "research_value": "The work is relevant to optical neural-network architecture research.",
        "limitations_and_open_questions": [
            "Long-term stability is not available in the supplied abstract."
        ],
        "optical_neural_network_analysis": {
            "architecture_type": "hybrid",
            "training_method": "not_available",
            "optical_nonlinearity": "not_available",
            "calibration_requirements": "not_available",
            "application_tasks": ["neural computation"],
            "hardware_validation": "physical_experiment",
        },
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_and_abstract_only",
            "unsupported_numbers_detected": False,
            "missing_information": ["full_text"],
        },
    }


def summary_request(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt": "Return JSON using only the supplied title, metadata, and abstract.",
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
            "abstract": "The authors report experimental optical hardware validation for neural computation.",
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.9,
            "decision": "mandatory",
            "score_breakdown": [],
        },
    }


class FakeModelClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict] = []

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        self.calls.append(kwargs)
        candidate_id = "candidate-benchmark"
        return DeepSeekResponse(
            content=json.dumps(valid_summary(candidate_id, self.model)),
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000,
            },
            model=self.model,
        )


def test_flash_and_pro_use_identical_requests_and_isolated_outputs(tmp_path: Path) -> None:
    request_path = tmp_path / "runtime-state" / "data" / "summary_requests" / "2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(summary_request("candidate-benchmark")) + "\n",
        encoding="utf-8",
    )
    dry_manifest = tmp_path / "runtime-state" / "state" / "summary_generation_manifest.json"
    dry_manifest.parent.mkdir(parents=True)
    dry_manifest.write_text(
        json.dumps({"digest_date": "2026-08-04", "request_file": str(request_path)}),
        encoding="utf-8",
    )
    clients = {
        "deepseek-v4-flash": FakeModelClient("deepseek-v4-flash"),
        "deepseek-v4-pro": FakeModelClient("deepseek-v4-pro"),
    }
    clock_values = iter([0.0, 1.0, 2.0, 4.0])
    manifest_path = tmp_path / "runtime-state" / "state" / "deepseek_benchmark_manifest.json"
    manifest = benchmark(
        dry_run_manifest_path=dry_manifest,
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        config_path=ROOT / "config" / "summary_generation.yaml",
        output_root=tmp_path / "runtime-state" / "data",
        manifest_path=manifest_path,
        api_key="test-key",
        client_factory=lambda model: clients[model],
        clock=lambda: next(clock_values),
    )

    assert manifest["status"] == "completed"
    assert manifest["request_count"] == 1
    assert manifest["same_prompt_for_both_models"] is True
    assert manifest["thinking_enabled"] is False
    assert manifest["email_enabled"] is False
    assert manifest["summary_history_updated"] is False
    assert manifest["models"]["flash"]["estimated_cost_cny"] == 0.002
    assert manifest["models"]["pro"]["estimated_cost_cny"] == 0.006
    assert manifest["models"]["pro"]["estimated_cost_cny"] == 3 * manifest["models"]["flash"]["estimated_cost_cny"]
    assert manifest["models"]["flash"]["elapsed_seconds"] == 1.0
    assert manifest["models"]["pro"]["elapsed_seconds"] == 2.0
    assert manifest["candidate_comparisons"][0]["architecture_agreement"] is True
    assert manifest["candidate_comparisons"][0]["hardware_validation_agreement"] is True

    flash_call = clients["deepseek-v4-flash"].calls[0]
    pro_call = clients["deepseek-v4-pro"].calls[0]
    assert flash_call["user_prompt"] == pro_call["user_prompt"]
    assert flash_call["system_prompt"] == pro_call["system_prompt"]
    assert flash_call["thinking_enabled"] is False
    assert pro_call["thinking_enabled"] is False
    assert flash_call["model"] == "deepseek-v4-flash"
    assert pro_call["model"] == "deepseek-v4-pro"

    benchmark_root = (
        tmp_path / "runtime-state" / "data" / "benchmarks" / "deepseek" / "2026-08-04"
    )
    assert (benchmark_root / "flash" / "summaries" / "2026-08-04.jsonl").exists()
    assert (benchmark_root / "pro" / "summaries" / "2026-08-04.jsonl").exists()
    assert (benchmark_root / "comparison.md").exists()
    assert manifest_path.exists()


def test_benchmark_workflow_is_manual_and_does_not_touch_delivery_or_history() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "benchmark-deepseek-models.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "DEEPSEEK_API_KEY" in workflow
    assert "GMAIL_CLIENT_SECRET" not in workflow
    assert "gmail_sender" not in workflow
    assert "summary_history.json" not in workflow
    assert "data/summaries" not in workflow
    assert "data/benchmarks/deepseek" in workflow


def test_benchmark_config_uses_current_official_models_and_nonthinking_mode() -> None:
    import yaml

    config = yaml.safe_load(
        (ROOT / "config" / "summary_generation.yaml").read_text(encoding="utf-8")
    )
    benchmark_config = config["benchmark"]
    assert benchmark_config["thinking_enabled"] is False
    assert benchmark_config["same_prompt_for_all_models"] is True
    assert [item["model"] for item in benchmark_config["models"]] == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    assert benchmark_config["models"][0]["pricing"]["input_cache_miss_cny_per_million"] == 1.0
    assert benchmark_config["models"][1]["pricing"]["input_cache_miss_cny_per_million"] == 3.0

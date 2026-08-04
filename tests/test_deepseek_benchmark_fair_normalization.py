from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize.benchmark_deepseek_models_v3 import (
    BenchmarkNormalizingClient,
    TransportRepairDiagnostics,
)
from scripts.summarize.deepseek_provider import DeepSeekResponse


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, content: dict) -> None:
        self.content = content

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        return DeepSeekResponse(
            content=json.dumps(self.content, ensure_ascii=False),
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
            model="deepseek-v4-flash",
        )


def test_transport_id_and_equivalent_unit_format_are_canonicalized() -> None:
    diagnostics = TransportRepairDiagnostics(expected_candidate_id="candidate-correct")
    client = BenchmarkNormalizingClient(
        base_client=FakeClient(
            {
                "candidate_id": "candidate-wrong",
                "reported_results": [
                    {"claim": "Compute density is 5.16 TOPS/mm²."}
                ],
            }
        ),
        diagnostics=diagnostics,
    )
    response = client.complete_json()
    value = json.loads(response.content)
    assert value["candidate_id"] == "candidate-correct"
    assert value["reported_results"][0]["claim"] == "Compute density is 5.16 TOPS/mm2."
    assert diagnostics.candidate_id_repair_responses == 1
    assert diagnostics.unit_format_normalization_responses == 1


def test_approximation_semantics_are_not_relaxed() -> None:
    diagnostics = TransportRepairDiagnostics(expected_candidate_id="candidate-correct")
    client = BenchmarkNormalizingClient(
        base_client=FakeClient(
            {
                "candidate_id": "candidate-correct",
                "core_problem": "Clock rates remain at 5 GHz.",
            }
        ),
        diagnostics=diagnostics,
    )
    response = client.complete_json()
    value = json.loads(response.content)
    assert value["core_problem"] == "Clock rates remain at 5 GHz."
    assert diagnostics.candidate_id_repair_responses == 0
    assert diagnostics.unit_format_normalization_responses == 0


def test_workflow_uses_v3_and_remains_manual_only() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "benchmark-deepseek-models.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "benchmark_deepseek_models_v3" in workflow
    assert "DEEPSEEK_API_KEY" in workflow
    assert "GMAIL_CLIENT_SECRET" not in workflow
    assert "summary_history.json" not in workflow

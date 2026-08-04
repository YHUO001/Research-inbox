from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize.deepseek_provider import DeepSeekClient, DeepSeekResponse
from scripts.summarize.generate_summaries import generate


ROOT = Path(__file__).resolve().parents[1]


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def summary(candidate_id: str, *, research_value: str = "Relevant to optical hardware research.") -> dict:
    return {
        "schema_version": 1,
        "summary_version": 1,
        "candidate_id": candidate_id,
        "core_problem": "The work addresses efficient optical computation.",
        "method_and_architecture": "The abstract describes an optical computing architecture.",
        "main_contributions": ["It presents an optical hardware method."],
        "reported_results": [],
        "distinction_from_prior_work": "not_available",
        "research_value": research_value,
        "limitations_and_open_questions": ["Long-term stability is not available."],
        "optical_neural_network_analysis": None,
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_and_abstract_only",
            "unsupported_numbers_detected": False,
            "missing_information": ["full_text"],
        },
    }


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
            "abstract": "The paper presents optical hardware for neural computation.",
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.9,
            "decision": "mandatory",
            "score_breakdown": [],
        },
    }


def test_deepseek_client_uses_json_mode_and_disables_thinking() -> None:
    captured: dict = {}

    def opener(http_request, timeout: float):
        captured["request"] = http_request
        captured["timeout"] = timeout
        return FakeHttpResponse(
            {
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": "{\"ok\": true}"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    client = DeepSeekClient(
        api_key="secret-test-key",
        opener=opener,
        sleeper=lambda _: None,
    )
    response = client.complete_json(
        model="deepseek-v4-pro",
        system_prompt="Return JSON.",
        user_prompt="Summarize.",
        max_tokens=1200,
        thinking_enabled=False,
    )
    body = json.loads(captured["request"].data.decode("utf-8"))
    assert body["model"] == "deepseek-v4-pro"
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert captured["request"].get_header("Authorization") == "Bearer secret-test-key"
    assert response.usage["total_tokens"] == 15


class FakeSummaryClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        self.calls += 1
        value = self.responses.pop(0)
        return DeepSeekResponse(
            content=json.dumps(value),
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000,
            },
            model="deepseek-v4-pro",
        )


def test_generation_retries_unsupported_number_and_writes_valid_preview(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate-abcdef"
    request_path = tmp_path / "runtime-state" / "data" / "summary_requests" / "2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request(candidate_id)) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "runtime-state" / "state" / "summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "digest_date": "2026-08-04",
                "request_file": str(request_path),
            }
        ),
        encoding="utf-8",
    )
    client = FakeSummaryClient(
        [
            summary(candidate_id, research_value="The method reaches 99% accuracy."),
            summary(candidate_id),
        ]
    )
    state = generate(
        dry_run_manifest_path=manifest_path,
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        config_path=ROOT / "config" / "summary_generation.yaml",
        output_root=tmp_path / "runtime-state" / "data",
        manifest_path=manifest_path,
        api_key="test-key",
        client=client,
    )
    assert client.calls == 2
    assert state["status"] == "completed"
    assert state["model"] == "deepseek-v4-pro"
    assert state["summary_count"] == 1
    assert state["estimated_cost_cny"] == 0.012
    assert state["email_enabled"] is False
    assert state["summary_history_updated"] is False
    assert (tmp_path / "runtime-state" / "data" / "summaries" / "2026-08-04.jsonl").exists()
    markdown = (
        tmp_path / "runtime-state" / "data" / "digests" / "2026-08-04.generated.md"
    ).read_text(encoding="utf-8")
    assert "Generated from title, metadata, and abstract only" in markdown
    assert "99%" not in markdown


def test_schema_versions_and_candidate_id_digits_are_not_treated_as_claims() -> None:
    from scripts.summarize.generate_summaries import validate_summary_numeric_grounding

    value = summary("83a830ace13b5158f346c392")
    assert validate_summary_numeric_grounding(
        value,
        title="Optical hardware",
        abstract="No numerical result is reported.",
    ) == []

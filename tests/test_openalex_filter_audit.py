from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.discovery.audit_openalex_filters import (
    audit_filters,
    balanced_filtered_samples,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def get_json(self, base_url: str, path: str, *, params: dict) -> dict:
        assert path == "works"
        assert params["api_key"] == "test-key"
        return {"results": self.results}


def work(work_id: str, title: str, abstract: str) -> dict:
    words = abstract.split()
    inverted: dict[str, list[int]] = {}
    for index, word in enumerate(words):
        inverted.setdefault(word, []).append(index)
    return {
        "id": f"https://openalex.org/{work_id}",
        "doi": f"https://doi.org/10.1000/{work_id.lower()}",
        "title": title,
        "publication_year": 2026,
        "publication_date": "2026-08-01",
        "authorships": [
            {"author": {"display_name": "Alice Researcher", "orcid": None}}
        ],
        "primary_location": {
            "landing_page_url": f"https://doi.org/10.1000/{work_id.lower()}",
            "source": {"display_name": "Optica"},
        },
        "best_oa_location": None,
        "open_access": {"oa_url": None},
        "abstract_inverted_index": inverted,
        "cited_by_count": 0,
        "ids": {
            "openalex": f"https://openalex.org/{work_id}",
            "doi": f"https://doi.org/10.1000/{work_id.lower()}",
        },
    }


def test_audit_records_bounded_filtered_samples_without_raw_payloads(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
discovery_version: 1
schedule:
  overlap_days: 3
  initial_lookback_days: 14
provider:
  base_url: https://api.openalex.org
  api_key_env: OPENALEX_API_KEY
  user_agent: test
  timeout_seconds: 1
  max_attempts: 1
  min_interval_seconds: 0
limits:
  maximum_results_per_query: 10
  maximum_abstract_characters: 2500
observability:
  maximum_filtered_samples: 1
queries:
  - id: optical
    project_id: optical-neural-networks
    text: optical neural network
""".strip()
        + "\n",
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "last_window_start": "2026-07-21",
                "last_window_end": "2026-08-04",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    manifest = audit_filters(
        config_path=config,
        state_path=state,
        output_path=output,
        recognition_path=ROOT / "config" / "recognition_rules.yaml",
        venues_path=ROOT / "config" / "venues.yaml",
        api_key="test-key",
        client=FakeClient(
            [
                work(
                    "W-RELEVANT",
                    "Forward-Only Training for Photonic Neural Networks",
                    "A photonic neural network using forward-only physical training.",
                ),
                work(
                    "W-NOISE",
                    "Marine Wireless Channel Survey",
                    "A survey of underwater radio communication.",
                ),
                work(
                    "W-NOISE-2",
                    "Thermal Control Review",
                    "A review of industrial control systems.",
                ),
            ]
        ),
        now=datetime(2026, 8, 4, tzinfo=timezone.utc),
    )
    assert manifest["matched_count"] == 1
    assert manifest["filtered_count"] == 2
    assert manifest["sample_count"] == 1
    assert manifest["sampling_strategy"] == "round_robin_unique_by_candidate"
    sample = manifest["filtered_samples"][0]
    assert sample["reason"] == "no_project_match"
    assert sample["title"] == "Marine Wireless Channel Survey"
    assert "abstract_inverted_index" not in output.read_text(encoding="utf-8")
    assert manifest["raw_provider_responses_persisted"] is False


def test_balanced_samples_cover_queries_and_remove_duplicate_candidates() -> None:
    duplicate = {
        "candidate_id": "same-candidate",
        "query_id": "query-a",
        "title": "Repeated result",
    }
    samples = balanced_filtered_samples(
        {
            "query-a": [
                duplicate,
                {
                    "candidate_id": "a-second",
                    "query_id": "query-a",
                    "title": "A second result",
                },
            ],
            "query-b": [
                {
                    **duplicate,
                    "query_id": "query-b",
                },
                {
                    "candidate_id": "b-second",
                    "query_id": "query-b",
                    "title": "B second result",
                },
            ],
            "query-c": [
                {
                    "candidate_id": "c-first",
                    "query_id": "query-c",
                    "title": "C first result",
                }
            ],
        },
        query_order=["query-a", "query-b", "query-c"],
        sample_limit=3,
    )
    assert [sample["candidate_id"] for sample in samples] == [
        "same-candidate",
        "b-second",
        "c-first",
    ]
    assert {sample["query_id"] for sample in samples} == {
        "query-a",
        "query-b",
        "query-c",
    }

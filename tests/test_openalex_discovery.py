from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from scripts.discovery.openalex_discovery import (
    QuerySpec,
    RegistryIdentityIndex,
    discover_once,
    make_candidate,
)
from scripts.pipeline.route_registry import rebuild_routes


ROOT = Path(__file__).resolve().parents[1]


def openalex_work(
    *,
    work_id: str,
    title: str,
    doi: str | None,
    abstract: str,
    venue: str = "Optica",
    year: int = 2026,
) -> dict:
    words = abstract.split()
    inverted: dict[str, list[int]] = {}
    for position, word in enumerate(words):
        inverted.setdefault(word, []).append(position)
    return {
        "id": f"https://openalex.org/{work_id}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": title,
        "publication_year": year,
        "publication_date": f"{year}-08-01",
        "authorships": [
            {
                "author": {
                    "display_name": "Alice Researcher",
                    "orcid": "https://orcid.org/0000-0001-0000-0001",
                }
            }
        ],
        "primary_location": {
            "landing_page_url": f"https://doi.org/{doi}" if doi else f"https://openalex.org/{work_id}",
            "source": {"display_name": venue},
        },
        "best_oa_location": None,
        "open_access": {"oa_url": None},
        "abstract_inverted_index": inverted,
        "cited_by_count": 0,
        "ids": {
            "openalex": f"https://openalex.org/{work_id}",
            "doi": f"https://doi.org/{doi}" if doi else None,
        },
    }


def scholar_record(*, candidate_id: str, title: str, year: int, doi: str | None) -> dict:
    return {
        "candidate_id": candidate_id,
        "source": {"source_type": "google_scholar_email", "message_id": "message-1"},
        "title": title,
        "normalized_title": title.lower(),
        "year": year,
        "identifiers": {
            "doi": {"value": doi},
            "arxiv_id": {"value": None},
            "pmid": {"value": None},
        },
        "content_fingerprint": f"fingerprint-{candidate_id}",
    }


def test_openalex_candidate_is_schema_valid() -> None:
    record = make_candidate(
        openalex_work(
            work_id="W1",
            title="Robust Optical Neural Networks",
            doi="10.1000/onn",
            abstract="An optical neural network with robust physical training.",
        ),
        query=QuerySpec("optical", "optical-neural-networks", "optical neural network"),
        rank=0,
        discovered_at="2026-08-04T00:00:00Z",
        maximum_abstract=2500,
    )
    schema = json.loads(
        (ROOT / "schemas" / "openalex_candidate.schema.json").read_text(encoding="utf-8")
    )
    errors = list(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(record)
    )
    assert errors == []
    assert record["source"]["source_type"] == "openalex"
    assert record["identifiers"]["doi"]["value"] == "10.1000/onn"
    assert record["snippet"].startswith("An optical neural network")


def test_registry_identity_deduplicates_doi_and_title_year() -> None:
    existing = scholar_record(
        candidate_id="scholar-1",
        title="Robust Optical Neural Networks",
        year=2026,
        doi="10.1000/onn",
    )
    index = RegistryIdentityIndex([existing])
    doi_duplicate = make_candidate(
        openalex_work(
            work_id="W2",
            title="A Different Title",
            doi="10.1000/onn",
            abstract="Optical neural network training.",
        ),
        query=QuerySpec("optical", "optical-neural-networks", "optical neural network"),
        rank=0,
        discovered_at="2026-08-04T00:00:00Z",
        maximum_abstract=2500,
    )
    assert index.duplicate_reason(doi_duplicate) == "doi"

    title_duplicate = make_candidate(
        openalex_work(
            work_id="W3",
            title="Robust Optical Neural Networks",
            doi="10.1000/other",
            abstract="Optical neural network training.",
        ),
        query=QuerySpec("optical", "optical-neural-networks", "optical neural network"),
        rank=0,
        discovered_at="2026-08-04T00:00:00Z",
        maximum_abstract=2500,
    )
    assert index.duplicate_reason(title_duplicate) == "normalized_title_year"


class FakeClient:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls = 0

    def get_json(self, base_url: str, path: str, *, params: dict) -> dict:
        self.calls += 1
        assert path == "works"
        assert "from_publication_date:" in params["filter"]
        assert params["api_key"] == "test-key"
        return {"results": self.results}


def write_discovery_config(path: Path) -> None:
    path.write_text(
        """
discovery_version: 1
schedule:
  minimum_interval_hours: 44
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
  maximum_raw_results_per_run: 10
  maximum_accepted_candidates_per_run: 10
  maximum_abstract_characters: 2500
queries:
  - id: optical
    project_id: optical-neural-networks
    text: optical neural network
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_discovery_filters_and_deduplicates_then_honors_state_gate(tmp_path: Path) -> None:
    config = tmp_path / "openalex.yaml"
    registry = tmp_path / "paper_registry.jsonl"
    state = tmp_path / "state.json"
    manifest = tmp_path / "manifest.json"
    write_discovery_config(config)
    existing = scholar_record(
        candidate_id="scholar-1",
        title="Existing Optical Neural Network",
        year=2026,
        doi="10.1000/existing",
    )
    registry.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    results = [
        openalex_work(
            work_id="W-existing",
            title="Existing Optical Neural Network",
            doi="10.1000/existing",
            abstract="An optical neural network.",
        ),
        openalex_work(
            work_id="W-new",
            title="Forward-Only Training for Photonic Neural Networks",
            doi="10.1000/new",
            abstract="A photonic neural network using forward-only physical training.",
        ),
        openalex_work(
            work_id="W-noise",
            title="Marine Wireless Channel Survey",
            doi="10.1000/noise",
            abstract="A survey of underwater radio communication.",
        ),
    ]
    client = FakeClient(results)
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    first = discover_once(
        config_path=config,
        registry_path=registry,
        state_path=state,
        manifest_path=manifest,
        schema_path=ROOT / "schemas" / "openalex_candidate.schema.json",
        recognition_path=ROOT / "config" / "recognition_rules.yaml",
        venues_path=ROOT / "config" / "venues.yaml",
        api_key="test-key",
        client=client,
        now=now,
        force=True,
    )
    assert first["accepted_count"] == 1
    assert first["duplicate_count"] == 1
    assert first["filtered_count"] == 1
    records = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[-1]["source"]["work_id"] == "https://openalex.org/W-new"

    before = registry.read_bytes()
    second = discover_once(
        config_path=config,
        registry_path=registry,
        state_path=state,
        manifest_path=manifest,
        schema_path=ROOT / "schemas" / "openalex_candidate.schema.json",
        recognition_path=ROOT / "config" / "recognition_rules.yaml",
        venues_path=ROOT / "config" / "venues.yaml",
        api_key="test-key",
        client=client,
        now=datetime(2026, 8, 4, 1, tzinfo=timezone.utc),
        force=False,
    )
    assert second["status"] == "skipped_not_due"
    assert registry.read_bytes() == before
    assert client.calls == 1


def test_openalex_optical_zo_is_not_mandatory(tmp_path: Path) -> None:
    candidate = make_candidate(
        openalex_work(
            work_id="W-ZO",
            title="Layered-Parameter Perturbation for Zeroth-Order Optimization of Optical Neural Networks",
            doi="10.1000/zo",
            abstract=(
                "A zeroth-order optimization method trains an optical neural network "
                "with structured layer-wise perturbations."
            ),
        ),
        query=QuerySpec("zo", "zeroth-order-optimization", "zeroth-order optical neural network"),
        rank=0,
        discovered_at="2026-08-04T00:00:00Z",
        maximum_abstract=2500,
    )
    registry = tmp_path / "registry.jsonl"
    recognition = tmp_path / "recognition.jsonl"
    queues = tmp_path / "queues"
    manifest = tmp_path / "routing.json"
    registry.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    rebuilt = rebuild_routes(
        registry_path=registry,
        recognition_path=recognition,
        queue_dir=queues,
        manifest_path=manifest,
        rules_path=ROOT / "config" / "recognition_rules.yaml",
        venues_path=ROOT / "config" / "venues.yaml",
        schema_path=ROOT / "schemas" / "recognition_result.schema.json",
    )
    assert rebuilt["route_counts"]["mandatory_summary_queue"] == 0
    assert rebuilt["route_counts"]["standard_scoring_queue"] == 1
    result = json.loads(recognition.read_text(encoding="utf-8"))
    assert result["routing"]["mandatory"] is False
    assert result["routing"]["reasons"] == ["project_match"]

from __future__ import annotations

import json
from pathlib import Path

from scripts.enrich.enrich_registry import enrich_registry
from scripts.enrich.metadata import Attempt


ROOT = Path(__file__).resolve().parents[1]


def candidate(index: int) -> dict:
    doi = f"10.1000/test-{index:04d}"
    return {
        "schema_version": 1,
        "candidate_id": f"candidate-{index:04d}",
        "source": {
            "source_type": "google_scholar_email",
            "message_id": f"message-{index:04d}",
            "received_at": "2026-08-18T00:00:00Z",
            "sender": "scholaralerts-noreply@google.com",
            "subject": '"optical neural network" - new results',
        },
        "position_in_message": index,
        "title": f"Robust Optical Neural Networks {index}",
        "normalized_title": f"robust optical neural networks {index}",
        "authors": [
            {
                "name": "A Researcher",
                "orcid": None,
                "verification_status": "raw_email",
            }
        ],
        "raw_metadata_line": "A Researcher - Optica, 2026",
        "venue": {
            "raw": "Optica",
            "normalized": "Optica",
            "verification_status": "raw_email",
        },
        "year": 2026,
        "snippet": "An optical neural network calibration method.",
        "identifiers": {
            "doi": {
                "value": doi,
                "verification_status": "regex_extracted",
                "source": "url",
            },
            "arxiv_id": {
                "value": None,
                "verification_status": "missing",
                "source": None,
            },
            "pmid": {
                "value": None,
                "verification_status": "missing",
                "source": None,
            },
        },
        "links": {
            "primary_url": f"https://doi.org/{doi}",
            "auxiliary_urls": [],
        },
        "parse_status": {
            "state": "complete",
            "warnings": [],
            "errors": [],
            "parser_strategy": "plain_text_blocks",
        },
        "content_fingerprint": f"fingerprint-{index:04d}",
        "extracted_at": "2026-08-18T00:00:00Z",
    }


def provider_record(provider: str, doi: str) -> dict:
    return {
        "provider": provider,
        "provider_id": (
            doi if provider == "crossref" else f"https://openalex.org/W{doi[-4:]}"
        ),
        "title": "Robust Optical Neural Networks",
        "authors": [{"name": "A Researcher", "orcid": None}],
        "doi": doi,
        "openalex_id": (
            f"https://openalex.org/W{doi[-4:]}" if provider == "openalex" else None
        ),
        "arxiv_id": None,
        "venue": "Optica",
        "publication_date": "2026-08-18",
        "year": 2026,
        "abstract": "Verified abstract.",
        "landing_page": f"https://doi.org/{doi}",
        "open_access_url": (
            f"https://example.org/oa/{doi[-4:]}" if provider == "openalex" else None
        ),
        "cited_by_count": 0,
    }


class CountingProvider:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.calls = 0

    def by_doi(self, doi: str) -> Attempt:
        self.calls += 1
        return Attempt(
            self.provider,
            "doi",
            "found",
            "exact",
            False,
            "2026-08-18T00:00:00Z",
            provider_record(self.provider, doi),
        )

    def by_title(self, source: dict) -> Attempt:
        self.calls += 1
        return Attempt(
            self.provider,
            "title_year_author",
            "found",
            "high",
            False,
            "2026-08-18T00:00:00Z",
            provider_record(self.provider, "10.1000/fallback"),
        )


def write_registry(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def config_with_limit(tmp_path: Path, limit: int) -> Path:
    source = (ROOT / "config" / "metadata_enrichment.yaml").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        "maximum_candidates_per_run: 100",
        f"maximum_candidates_per_run: {limit}",
    )
    path = tmp_path / f"metadata_enrichment_{limit}.yaml"
    path.write_text(source, encoding="utf-8")
    return path


def run_enrichment(
    tmp_path: Path,
    records: list[dict],
    *,
    limit: int,
    crossref: CountingProvider,
    openalex: CountingProvider,
) -> dict:
    registry = tmp_path / "paper_registry.jsonl"
    write_registry(registry, records)
    return enrich_registry(
        registry_path=registry,
        output_path=tmp_path / "enriched.jsonl",
        manifest_path=tmp_path / "manifest.json",
        cache_path=tmp_path / "cache.json",
        config_path=config_with_limit(tmp_path, limit),
        schema_path=ROOT / "schemas" / "enriched_paper.schema.json",
        crossref=crossref,
        openalex=openalex,
        openalex_configured=True,
    )


def test_registry_can_grow_past_lifetime_limit_by_reusing_prior_enrichments(
    tmp_path: Path,
) -> None:
    crossref = CountingProvider("crossref")
    openalex = CountingProvider("openalex")

    first_records = [candidate(index) for index in range(59)]
    first = run_enrichment(
        tmp_path,
        first_records,
        limit=100,
        crossref=crossref,
        openalex=openalex,
    )
    assert first["candidate_count"] == 59
    assert first["enriched_this_run_count"] == 59
    assert first["deferred_candidate_count"] == 0
    assert crossref.calls == 59
    assert openalex.calls == 59

    expanded_records = [candidate(index) for index in range(109)]
    second = run_enrichment(
        tmp_path,
        expanded_records,
        limit=100,
        crossref=crossref,
        openalex=openalex,
    )

    assert second["candidate_count"] == 109
    assert second["output_candidate_count"] == 109
    assert second["reused_candidate_count"] == 59
    assert second["enrichment_due_count"] == 50
    assert second["enriched_this_run_count"] == 50
    assert second["deferred_candidate_count"] == 0
    assert crossref.calls == 109
    assert openalex.calls == 109


def test_excess_new_candidates_are_deferred_without_failing_the_pipeline(
    tmp_path: Path,
) -> None:
    crossref = CountingProvider("crossref")
    openalex = CountingProvider("openalex")
    records = [candidate(index) for index in range(3)]

    first = run_enrichment(
        tmp_path,
        records,
        limit=2,
        crossref=crossref,
        openalex=openalex,
    )
    assert first["candidate_count"] == 3
    assert first["output_candidate_count"] == 2
    assert first["reused_candidate_count"] == 0
    assert first["enrichment_due_count"] == 3
    assert first["enriched_this_run_count"] == 2
    assert first["deferred_candidate_count"] == 1
    assert first["deferred_candidate_ids"] == ["candidate-0002"]

    second = run_enrichment(
        tmp_path,
        records,
        limit=2,
        crossref=crossref,
        openalex=openalex,
    )
    assert second["candidate_count"] == 3
    assert second["output_candidate_count"] == 3
    assert second["reused_candidate_count"] == 2
    assert second["enrichment_due_count"] == 1
    assert second["enriched_this_run_count"] == 1
    assert second["deferred_candidate_count"] == 0
    assert crossref.calls == 3
    assert openalex.calls == 3


def test_changed_source_fingerprint_forces_reenrichment(tmp_path: Path) -> None:
    crossref = CountingProvider("crossref")
    openalex = CountingProvider("openalex")
    record = candidate(1)

    run_enrichment(
        tmp_path,
        [record],
        limit=100,
        crossref=crossref,
        openalex=openalex,
    )
    assert crossref.calls == 1
    assert openalex.calls == 1

    changed = json.loads(json.dumps(record))
    changed["content_fingerprint"] = "fingerprint-0001-reconciled"
    second = run_enrichment(
        tmp_path,
        [changed],
        limit=100,
        crossref=crossref,
        openalex=openalex,
    )
    assert second["reused_candidate_count"] == 0
    assert second["enriched_this_run_count"] == 1
    assert crossref.calls == 2
    assert openalex.calls == 2

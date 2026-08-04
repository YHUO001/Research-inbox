from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import HTTPError

from scripts.enrich.enrich_registry import enrich_registry
from scripts.enrich.metadata import (
    Attempt,
    JsonHttpClient,
    NormalizedCache,
    OpenAlexProvider,
    conservative_match,
    merge_enrichment,
)


ROOT = Path(__file__).resolve().parents[1]


def candidate(*, doi: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "candidate_id": "candidate-0001",
        "source": {
            "source_type": "google_scholar_email",
            "message_id": "message-1",
            "received_at": "2026-08-04T00:00:00Z",
            "sender": "scholaralerts-noreply@google.com",
            "subject": '"optical neural network" - new results',
        },
        "position_in_message": 0,
        "title": "Robust Optical Neural Networks",
        "normalized_title": "robust optical neural networks",
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
                "verification_status": "regex_extracted" if doi else "missing",
                "source": "url" if doi else None,
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
            "primary_url": "https://example.org/paper",
            "auxiliary_urls": [],
        },
        "parse_status": {
            "state": "complete",
            "warnings": [],
            "errors": [],
            "parser_strategy": "plain_text_blocks",
        },
        "content_fingerprint": "fingerprint-001",
        "extracted_at": "2026-08-04T00:00:00Z",
    }


def provider_record(provider: str, *, doi: str = "10.1000/test") -> dict:
    return {
        "provider": provider,
        "provider_id": (
            doi if provider == "crossref" else "https://openalex.org/W1"
        ),
        "title": "Robust Optical Neural Networks",
        "authors": [{"name": "Alice Researcher", "orcid": None}],
        "doi": doi,
        "openalex_id": (
            "https://openalex.org/W1" if provider == "openalex" else None
        ),
        "arxiv_id": None,
        "venue": "Optica",
        "publication_date": "2026-08-01",
        "year": 2026,
        "abstract": "Verified abstract.",
        "landing_page": f"https://doi.org/{doi}",
        "open_access_url": (
            "https://example.org/oa" if provider == "openalex" else None
        ),
        "cited_by_count": 4,
    }


def test_conservative_match_requires_corroborating_metadata() -> None:
    source = candidate()
    record = provider_record("crossref")
    assert (
        conservative_match(
            source,
            [record],
            minimum=0.96,
            ambiguity_margin=0.01,
            maximum_year_difference=1,
            require_confirmation=True,
        )
        == record
    )

    source["year"] = None
    source["authors"] = []
    assert (
        conservative_match(
            source,
            [record],
            minimum=0.96,
            ambiguity_margin=0.01,
            maximum_year_difference=1,
            require_confirmation=True,
        )
        is None
    )


def test_conservative_match_rejects_ambiguous_results() -> None:
    source = candidate()
    first = provider_record("crossref", doi="10.1000/one")
    second = provider_record("crossref", doi="10.1000/two")
    assert (
        conservative_match(
            source,
            [first, second],
            minimum=0.96,
            ambiguity_margin=0.01,
            maximum_year_difference=1,
            require_confirmation=True,
        )
        is None
    )


def test_openalex_without_key_is_safely_skipped() -> None:
    config = {"base_url": "https://api.openalex.org"}
    client = JsonHttpClient(
        user_agent="test",
        timeout_seconds=1,
        max_attempts=1,
        min_interval_seconds=0,
    )
    provider = OpenAlexProvider(
        config,
        client=client,
        cache=NormalizedCache(),
        api_key=None,
        maximum_abstract=1000,
        exact_ttl_days=30,
        search_ttl_days=7,
        matching={
            "maximum_search_results_per_provider": 3,
            "title_similarity_minimum": 0.96,
            "ambiguity_margin": 0.01,
            "maximum_year_difference": 1,
            "require_year_or_first_author_confirmation": True,
        },
    )
    attempt = provider.by_doi("10.1000/test")
    assert attempt.status == "skipped"
    assert attempt.reason == "missing_api_key"
    assert attempt.record is None


def test_merge_prefers_verified_provider_fields_without_mutating_source() -> None:
    source = candidate(doi="10.1000/test")
    original = json.loads(json.dumps(source))
    crossref = Attempt(
        provider="crossref",
        method="doi",
        status="found",
        confidence="exact",
        cache_hit=False,
        retrieved_at="2026-08-04T01:00:00Z",
        record=provider_record("crossref"),
    )
    openalex = Attempt(
        provider="openalex",
        method="doi",
        status="found",
        confidence="exact",
        cache_hit=False,
        retrieved_at="2026-08-04T01:00:01Z",
        record=provider_record("openalex"),
    )

    enriched = merge_enrichment(source, [crossref, openalex])
    assert source == original
    assert enriched["match"]["status"] == "exact"
    assert enriched["fields"]["doi"]["source"] == "crossref"
    assert enriched["fields"]["abstract"]["source"] == "openalex"
    assert (
        enriched["fields"]["open_access_url"]["value"]
        == "https://example.org/oa"
    )
    assert "record" not in enriched["provider_attempts"][0]


class StubProvider:
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def by_doi(self, doi: str) -> Attempt:
        return Attempt(
            self.provider,
            "doi",
            "found",
            "exact",
            False,
            "2026-08-04T01:00:00Z",
            provider_record(self.provider, doi=doi),
        )

    def by_title(self, source: dict) -> Attempt:
        return Attempt(
            self.provider,
            "title_year_author",
            "found",
            "high",
            False,
            "2026-08-04T01:00:00Z",
            provider_record(self.provider),
        )


def test_registry_output_is_schema_valid_and_idempotent(tmp_path: Path) -> None:
    registry = tmp_path / "paper_registry.jsonl"
    output = tmp_path / "enriched.jsonl"
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.json"
    registry.write_text(
        json.dumps(candidate(doi="10.1000/test"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    kwargs = {
        "registry_path": registry,
        "output_path": output,
        "manifest_path": manifest,
        "cache_path": cache,
        "config_path": ROOT / "config" / "metadata_enrichment.yaml",
        "schema_path": ROOT / "schemas" / "enriched_paper.schema.json",
        "crossref": StubProvider("crossref"),
        "openalex": StubProvider("openalex"),
        "openalex_configured": True,
    }
    first = enrich_registry(**kwargs)
    first_bytes = (output.read_bytes(), manifest.read_bytes(), cache.read_bytes())
    second = enrich_registry(**kwargs)
    second_bytes = (output.read_bytes(), manifest.read_bytes(), cache.read_bytes())

    assert first == second
    assert first_bytes == second_bytes
    assert first["candidate_count"] == 1
    assert first["match_counts"]["exact"] == 1
    assert first["openalex_configured"] is True


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_http_client_retries_rate_limit_without_logging_url() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def opener(request: object, timeout: float) -> FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                url="https://api.example.test/secret",
                code=429,
                msg="rate limited",
                hdrs=None,
                fp=io.BytesIO(),
            )
        return FakeResponse({"ok": True})

    client = JsonHttpClient(
        user_agent="test",
        timeout_seconds=1,
        max_attempts=2,
        min_interval_seconds=0,
        opener=opener,
        sleeper=sleeps.append,
        monotonic=lambda: 1.0,
    )
    assert client.get_json("https://api.example.test", "works") == {"ok": True}
    assert calls["count"] == 2
    assert sleeps == [1.0]

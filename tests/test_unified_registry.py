from __future__ import annotations

import json
from pathlib import Path

from scripts.pipeline.reconcile_registry import reconcile_registry


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def scholar_record(candidate_id: str = "scholar-001") -> dict:
    return {
        "candidate_id": candidate_id,
        "source": {
            "source_type": "google_scholar_email",
            "message_id": "message-1",
        },
        "title": "Unified Optical Neural Network Training",
        "normalized_title": "unified optical neural network training",
        "authors": [{"name": "Alice Smith"}],
        "raw_metadata_line": "Alice Smith - Nature Photonics, 2026",
        "venue": {
            "raw": "Nature Photonics",
            "normalized": "Nature Photonics",
            "verification_status": "parser_extracted",
        },
        "year": 2026,
        "snippet": "Short Scholar abstract.",
        "identifiers": {
            "doi": {
                "value": "10.1000/unified",
                "verification_status": "doi_extracted",
                "source": "scholar_email",
            },
            "arxiv_id": {"value": None, "verification_status": "missing", "source": None},
            "pmid": {"value": None, "verification_status": "missing", "source": None},
        },
        "links": {"primary_url": "https://example.org/scholar", "auxiliary_urls": []},
        "parse_status": {
            "state": "complete",
            "warnings": [],
            "errors": [],
            "parser_strategy": "test",
        },
        "content_fingerprint": "scholar-fingerprint",
    }


def openalex_record(candidate_id: str = "openalex-001") -> dict:
    return {
        "candidate_id": candidate_id,
        "source": {
            "source_type": "openalex",
            "work_id": "https://openalex.org/W123",
        },
        "title": "Unified Optical Neural Network Training",
        "normalized_title": "unified optical neural network training",
        "authors": [{"name": "Alice Smith"}, {"name": "Bob Lee"}],
        "raw_metadata_line": "Alice Smith, Bob Lee - Nature Photonics, 2026",
        "venue": {
            "raw": "Nature Photonics",
            "normalized": "Nature Photonics",
            "verification_status": "metadata_verified",
        },
        "year": 2026,
        "snippet": "A substantially longer OpenAlex abstract with method and result context.",
        "identifiers": {
            "doi": {
                "value": "https://doi.org/10.1000/unified",
                "verification_status": "metadata_verified",
                "source": "external_metadata",
            },
            "arxiv_id": {"value": None, "verification_status": "missing", "source": None},
            "pmid": {"value": None, "verification_status": "missing", "source": None},
        },
        "links": {
            "primary_url": "https://openalex.org/W123",
            "auxiliary_urls": ["https://example.org/open.pdf"],
        },
        "parse_status": {
            "state": "complete",
            "warnings": [],
            "errors": [],
            "parser_strategy": "openalex_api",
        },
        "content_fingerprint": "openalex-fingerprint",
    }


def run_reconciliation(tmp_path: Path, records: list[dict], history: dict | None = None):
    raw = tmp_path / "paper_registry.jsonl"
    unified = tmp_path / "unified_paper_registry.jsonl"
    aliases = tmp_path / "aliases.json"
    manifest = tmp_path / "manifest.json"
    history_path = tmp_path / "history.json"
    write_jsonl(raw, records)
    history_path.write_text(json.dumps(history or {}), encoding="utf-8")
    result = reconcile_registry(
        raw_registry_path=raw,
        unified_registry_path=unified,
        alias_path=aliases,
        manifest_path=manifest,
        history_path=history_path,
    )
    return result, read_jsonl(unified), json.loads(aliases.read_text(encoding="utf-8"))


def test_merges_scholar_and_openalex_by_doi_and_preserves_provenance(tmp_path: Path) -> None:
    result, records, aliases = run_reconciliation(
        tmp_path, [scholar_record(), openalex_record()]
    )

    assert result["raw_candidate_count"] == 2
    assert result["unified_candidate_count"] == 1
    assert result["merged_group_count"] == 1
    merged = records[0]
    assert merged["candidate_id"] == "scholar-001"
    assert merged["source_types"] == ["google_scholar_email", "openalex"]
    assert merged["identifiers"]["doi"]["value"] == "https://doi.org/10.1000/unified"
    assert len(merged["authors"]) == 2
    assert "longer OpenAlex abstract" in merged["snippet"]
    assert len(merged["source_provenance"]) == 2
    assert aliases["aliases"] == {"openalex-001": "scholar-001"}


def test_preserves_an_already_completed_candidate_id_as_canonical(tmp_path: Path) -> None:
    history = {"completed_candidate_ids": {"openalex-001": {"completed_at": "now"}}}
    _, records, aliases = run_reconciliation(
        tmp_path, [scholar_record(), openalex_record()], history
    )

    assert records[0]["candidate_id"] == "openalex-001"
    assert aliases["aliases"] == {"scholar-001": "openalex-001"}

from __future__ import annotations

import json
from pathlib import Path

from scripts.pipeline.route_registry import rebuild_routes

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def candidate(
    candidate_id: str,
    title: str,
    *,
    snippet: str = "",
    venue: str = "",
    parse_state: str = "complete",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "title": title,
        "snippet": snippet or None,
        "venue": {
            "raw": venue or None,
            "normalized": venue or None,
        },
        "raw_metadata_line": None,
        "authors": [],
        "parse_status": {
            "state": parse_state,
            "warnings": [],
        },
        "source": {
            "source_type": "google_scholar_email",
        },
        "extracted_at": "2026-08-04T00:00:00Z",
    }


def test_rebuild_routes_is_stable_and_writes_all_queues(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "data" / "paper_registry.jsonl"
    recognition = tmp_path / "data" / "recognition_results.jsonl"
    queues = tmp_path / "data" / "queues"
    manifest = tmp_path / "state" / "routing_manifest.json"

    records = [
        candidate(
            "candidate-001",
            "Zeroth-order optimization of optical neural networks",
            venue="AAAI",
        ),
        candidate(
            "candidate-002",
            "Optical flow neural network for video",
            venue="CVPR",
        ),
        candidate(
            "candidate-003",
            "A broad computing result",
            venue="Nature Communications",
        ),
    ]
    write_jsonl(registry, records)

    first = rebuild_routes(
        registry_path=registry,
        recognition_path=recognition,
        queue_dir=queues,
        manifest_path=manifest,
        rules_path=ROOT / "config" / "recognition_rules.yaml",
        venues_path=ROOT / "config" / "venues.yaml",
        schema_path=ROOT / "schemas" / "recognition_result.schema.json",
    )
    first_snapshot = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    second = rebuild_routes(
        registry_path=registry,
        recognition_path=recognition,
        queue_dir=queues,
        manifest_path=manifest,
        rules_path=ROOT / "config" / "recognition_rules.yaml",
        venues_path=ROOT / "config" / "venues.yaml",
        schema_path=ROOT / "schemas" / "recognition_result.schema.json",
    )
    second_snapshot = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    assert first == second
    assert first_snapshot == second_snapshot
    assert first["candidate_count"] == 3
    assert first["route_counts"]["mandatory_summary_queue"] == 2
    assert first["route_counts"]["archive"] == 1

    mandatory = (
        queues / "mandatory_summary_queue.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    archive = (
        queues / "archive.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(mandatory) == 2
    assert len(archive) == 1

    for route in (
        "mandatory_summary_queue",
        "standard_scoring_queue",
        "metadata_enrichment_queue",
        "manual_review_queue",
        "archive",
    ):
        assert (queues / f"{route}.jsonl").exists()

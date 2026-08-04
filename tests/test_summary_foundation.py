from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize.prepare_digest import (
    build_summary_request,
    prepare_dry_run,
    validate_numeric_grounding,
)


ROOT = Path(__file__).resolve().parents[1]


def candidate(
    candidate_id: str,
    *,
    status: str,
    projects: list[str],
    score: float,
    abstract: str = "The system achieved 95% accuracy using physical optical hardware.",
) -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "title": f"Optical neural network paper {candidate_id}",
        "authors": ["Alice Researcher"],
        "venue": "Optica",
        "year": 2026,
        "source_type": "google_scholar_email",
        "doi": f"10.1000/{candidate_id}",
        "openalex_id": f"https://openalex.org/{candidate_id}",
        "landing_page": f"https://doi.org/10.1000/{candidate_id}",
        "open_access_url": None,
        "abstract": abstract,
        "matched_projects": projects,
        "mandatory": status == "summary_slot",
        "score": score,
        "decision": "mandatory" if status == "summary_slot" else "summarize",
        "selection_status": status,
        "score_breakdown": [],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_prepare_dry_run_is_bounded_and_byte_stable(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    records = [
        candidate(
            f"candidate-{index}",
            status="summary_slot",
            projects=["optical-neural-networks"],
            score=0.9 - index / 100,
        )
        for index in range(4)
    ]
    records.append(
        candidate(
            "candidate-next",
            status="llm_candidate_only",
            projects=["optical-neural-networks"],
            score=0.8,
        )
    )
    write_jsonl(queue, records)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"built_at": "2026-08-04T11:27:51Z"}),
        encoding="utf-8",
    )
    output = tmp_path / "data"
    manifest_path = tmp_path / "state" / "manifest.json"
    first = prepare_dry_run(
        queue_path=queue,
        selection_manifest_path=selection,
        config_path=ROOT / "config" / "summary_generation.yaml",
        request_schema_path=ROOT / "schemas" / "summary_request.schema.json",
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        output_root=output,
        state_manifest_path=manifest_path,
    )
    first_bytes = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    second = prepare_dry_run(
        queue_path=queue,
        selection_manifest_path=selection,
        config_path=ROOT / "config" / "summary_generation.yaml",
        request_schema_path=ROOT / "schemas" / "summary_request.schema.json",
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        output_root=output,
        state_manifest_path=manifest_path,
    )
    second_bytes = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert first == second
    assert first_bytes == second_bytes
    assert first["request_count"] == 3
    assert first["actual_summary_count"] == 0
    assert first["llm_enabled"] is False
    assert first["email_enabled"] is False
    assert first["summary_history_updated"] is False


def test_project_specific_prompt_instructions() -> None:
    request = build_summary_request(
        candidate(
            "candidate-both",
            status="summary_slot",
            projects=["optical-neural-networks", "zeroth-order-optimization"],
            score=0.9,
        ),
        prepared_at="2026-08-04T11:27:51Z",
        prompt_version=1,
        summary_schema_name="paper_summary.schema.json",
    )
    prompt = request["prompt"]
    assert "free-space, integrated, or hybrid" in prompt
    assert "total query complexity from per-step query count" in prompt
    assert "Do not infer" in prompt


def test_numeric_grounding_rejects_invented_result() -> None:
    summary = {
        "candidate_id": "candidate-1",
        "reported_results": [{"claim": "Accuracy reached 99.9%."}],
    }
    unsupported = validate_numeric_grounding(
        summary,
        title="Optical neural network",
        abstract="The system achieved 95% accuracy.",
    )
    assert "99.9%" in unsupported
    supported = validate_numeric_grounding(
        {"reported_results": [{"claim": "Accuracy reached 95%."}]},
        title="Optical neural network",
        abstract="The system achieved 95% accuracy.",
    )
    assert supported == []


def test_preview_contains_no_generated_claims(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    write_jsonl(
        queue,
        [
            candidate(
                "candidate-1",
                status="summary_slot",
                projects=["optical-neural-networks"],
                score=0.9,
            )
        ],
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"built_at": "2026-08-04T11:27:51Z"}),
        encoding="utf-8",
    )
    prepare_dry_run(
        queue_path=queue,
        selection_manifest_path=selection,
        config_path=ROOT / "config" / "summary_generation.yaml",
        request_schema_path=ROOT / "schemas" / "summary_request.schema.json",
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        output_root=tmp_path / "data",
        state_manifest_path=tmp_path / "state" / "manifest.json",
    )
    markdown = (tmp_path / "data" / "digests" / "2026-08-04.preview.md").read_text(
        encoding="utf-8"
    )
    digest = json.loads(
        (tmp_path / "data" / "digests" / "2026-08-04.preview.json").read_text(
            encoding="utf-8"
        )
    )
    assert "No model was called" in markdown
    assert digest["summary_count"] == 0
    assert digest["safety"]["summary_history_updated"] is False
    assert digest["sections"]["must_read"][0]["status"] == "pending_model_summary"

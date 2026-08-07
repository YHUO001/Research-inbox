from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize.finalize_automatic_safe import validate_required_artifacts
from scripts.summarize.generate_automatic_summaries import preserve_preparation_artifacts


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_generation_preserves_prepared_artifact_references() -> None:
    prepared = {
        "digest_date": "2026-08-07",
        "request_file": "runtime-state/data/summary_requests/2026-08-07.jsonl",
        "request_sha256": "request-hash",
        "selection_manifest_sha256": "selection-hash",
        "queue_sha256": "queue-hash",
        "history_path": "runtime-state/state/summary_history.json",
        "summary_slot_count": 3,
    }
    generated = {
        "digest_date": "2026-08-07",
        "status": "completed",
        "summary_file": "runtime-state/data/summaries/2026-08-07.jsonl",
    }

    result = preserve_preparation_artifacts(generated, prepared)

    assert result["request_file"] == prepared["request_file"]
    assert result["request_sha256"] == "request-hash"
    assert result["selection_manifest_sha256"] == "selection-hash"
    assert result["queue_sha256"] == "queue-hash"
    assert result["history_path"] == "runtime-state/state/summary_history.json"
    assert result["summary_slot_count"] == 3


def test_generation_rejects_mismatched_digest_dates() -> None:
    with pytest.raises(RuntimeError, match="does not match prepared date"):
        preserve_preparation_artifacts(
            {"digest_date": "2026-08-08"},
            {"digest_date": "2026-08-07", "request_file": "requests.jsonl"},
        )


def test_finalization_rejects_missing_request_file_explicitly(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {
            "summary_file": str(tmp_path / "summary.jsonl"),
            "digest_json_file": str(tmp_path / "digest.json"),
            "digest_markdown_file": str(tmp_path / "digest.md"),
        },
    )

    with pytest.raises(RuntimeError, match="missing request_file"):
        validate_required_artifacts(manifest_path)


def test_finalization_rejects_directory_as_artifact(tmp_path: Path) -> None:
    summary = tmp_path / "summary.jsonl"
    digest_json = tmp_path / "digest.json"
    digest_markdown = tmp_path / "digest.md"
    summary.write_text("{}\n", encoding="utf-8")
    digest_json.write_text("{}\n", encoding="utf-8")
    digest_markdown.write_text("# digest\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {
            "request_file": str(tmp_path),
            "summary_file": str(summary),
            "digest_json_file": str(digest_json),
            "digest_markdown_file": str(digest_markdown),
        },
    )

    with pytest.raises(RuntimeError, match="Missing request artifact file"):
        validate_required_artifacts(manifest_path)

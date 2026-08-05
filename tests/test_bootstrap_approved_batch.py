from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize import bootstrap_approved_batch as bootstrap


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def test_bootstraps_approved_legacy_batch_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root = tmp_path / "runtime-state"
    manifest_path = state_root / "state/summary_generation_manifest.json"
    history_path = state_root / "state/summary_history.json"
    request_path = state_root / "data/summary_requests/2026-08-05.jsonl"
    write_jsonl(request_path, [{"candidate_id": "candidate-approved"}])
    write_json(
        manifest_path,
        {
            "status": "completed",
            "digest_date": "2026-08-05",
            "request_count": 1,
            "summary_count": 1,
            "failure_count": 0,
        },
    )
    write_json(
        history_path,
        {
            "schema_version": 1,
            "completed_candidate_ids": {
                "candidate-approved": {
                    "completion_mode": "migrated_from_user_reviewed_batch",
                    "review_required": False,
                }
            },
            "failed_candidate_ids": {},
        },
    )

    calls: list[dict] = []

    def fake_finalize(**kwargs):
        calls.append(kwargs)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["request_file"] == str(request_path)
        manifest.update(
            {
                "status": "completed_automatic",
                "summary_history_updated": True,
                "completed_at": "2026-08-05T10:00:00Z",
            }
        )
        write_json(manifest_path, manifest)
        return manifest

    monkeypatch.setattr(bootstrap, "finalize_automatic", fake_finalize)
    result = bootstrap.bootstrap_approved_batch(
        generation_manifest_path=manifest_path,
        state_root=state_root,
        history_path=history_path,
        summary_schema_path=tmp_path / "schema.json",
        config_path=tmp_path / "config.yaml",
    )

    assert result == {
        "status": "completed",
        "bootstrapped": True,
        "digest_date": "2026-08-05",
        "candidate_count": 1,
        "new_provider_calls": 0,
    }
    assert len(calls) == 1
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed_automatic"
    assert persisted["bootstrapped"] is True
    assert persisted["new_provider_calls"] == 0


def test_does_not_bootstrap_unapproved_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root = tmp_path / "runtime-state"
    manifest_path = state_root / "state/summary_generation_manifest.json"
    history_path = state_root / "state/summary_history.json"
    write_jsonl(
        state_root / "data/summary_requests/2026-08-05.jsonl",
        [{"candidate_id": "candidate-unapproved"}],
    )
    write_json(
        manifest_path,
        {
            "status": "completed",
            "digest_date": "2026-08-05",
            "request_count": 1,
            "summary_count": 1,
            "failure_count": 0,
        },
    )
    write_json(
        history_path,
        {
            "completed_candidate_ids": {},
            "failed_candidate_ids": {},
        },
    )
    monkeypatch.setattr(
        bootstrap,
        "finalize_automatic",
        lambda **kwargs: pytest.fail("finalizer must not run"),
    )

    with pytest.raises(RuntimeError, match="not recorded as user-approved"):
        bootstrap.bootstrap_approved_batch(
            generation_manifest_path=manifest_path,
            state_root=state_root,
            history_path=history_path,
            summary_schema_path=tmp_path / "schema.json",
            config_path=tmp_path / "config.yaml",
        )


def test_skips_batch_already_in_automatic_state(tmp_path: Path) -> None:
    manifest_path = tmp_path / "state/summary_generation_manifest.json"
    write_json(
        manifest_path,
        {"status": "completed_automatic", "summary_history_updated": True},
    )
    result = bootstrap.bootstrap_approved_batch(
        generation_manifest_path=manifest_path,
        state_root=tmp_path,
        history_path=tmp_path / "history.json",
        summary_schema_path=tmp_path / "schema.json",
        config_path=tmp_path / "config.yaml",
    )
    assert result == {
        "status": "skipped_already_automatic",
        "bootstrapped": False,
    }

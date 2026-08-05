from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.pipeline.daily_source_gate import PRIMARY_CRON, RETRY_CRON, plan, verify


def write_state(path: Path, timestamp: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_successful_run_at": timestamp}),
        encoding="utf-8",
    )


def write_delivery(path: Path, date: str = "2026-08-05") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"sent_digests": {date: {"status": "sent"}}}),
        encoding="utf-8",
    )


def plan_for(tmp_path: Path, now: datetime, schedule: str):
    return plan(
        gmail_state_path=tmp_path / "gmail.json",
        openalex_state_path=tmp_path / "openalex.json",
        delivery_state_path=tmp_path / "delivery.json",
        event_name="schedule",
        event_schedule=schedule,
        now=now,
    )


def test_primary_run_collects_both_sources(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 0, 17, tzinfo=timezone.utc)
    result = plan_for(tmp_path, now, PRIMARY_CRON)
    assert result["mode"] == "primary"
    assert result["run_scholar"] is True
    assert result["run_openalex"] is True
    assert result["should_run"] is True


def test_retry_runs_only_the_source_missing_today(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc)
    write_state(tmp_path / "gmail.json", "2026-08-05T00:20:00Z")
    write_state(tmp_path / "openalex.json", "2026-08-04T02:00:00Z")

    result = plan_for(tmp_path, now, RETRY_CRON)
    assert result["run_scholar"] is False
    assert result["run_openalex"] is True
    assert result["should_run"] is True


def test_retry_continues_downstream_when_sources_succeeded_but_digest_missing(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc)
    write_state(tmp_path / "gmail.json", "2026-08-05T00:20:00Z")
    write_state(tmp_path / "openalex.json", "2026-08-05T00:30:00Z")

    result = plan_for(tmp_path, now, RETRY_CRON)
    assert result["run_scholar"] is False
    assert result["run_openalex"] is False
    assert result["should_run"] is True
    assert result["daily_digest_delivered_before"] is False


def test_retry_is_noop_after_sources_and_digest_succeeded(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc)
    write_state(tmp_path / "gmail.json", "2026-08-05T00:20:00Z")
    write_state(tmp_path / "openalex.json", "2026-08-05T00:30:00Z")
    write_delivery(tmp_path / "delivery.json")

    result = plan_for(tmp_path, now, RETRY_CRON)
    assert result["should_run"] is False
    assert result["run_scholar"] is False
    assert result["run_openalex"] is False
    assert result["daily_digest_delivered_before"] is True


def test_missing_or_corrupt_state_is_retried(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc)
    (tmp_path / "gmail.json").write_text("not-json", encoding="utf-8")

    result = plan_for(tmp_path, now, RETRY_CRON)
    assert result["run_scholar"] is True
    assert result["run_openalex"] is True


def test_verify_requires_both_sources_on_same_local_day(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
    write_state(tmp_path / "gmail.json", "2026-08-05T14:00:00Z")
    write_state(tmp_path / "openalex.json", "2026-08-04T14:00:00Z")

    result = verify(
        gmail_state_path=tmp_path / "gmail.json",
        openalex_state_path=tmp_path / "openalex.json",
        now=now,
    )
    assert result["ready"] is False
    assert result["missing_sources"] == ["openalex"]

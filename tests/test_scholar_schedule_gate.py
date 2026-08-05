from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ingest.scholar_schedule_gate import (
    PRIMARY_SCHEDULE,
    RETRY_SCHEDULE,
    decide_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_state(path: Path, last_successful_run_at: str | None) -> None:
    path.write_text(
        json.dumps({"last_successful_run_at": last_successful_run_at}),
        encoding="utf-8",
    )


def test_primary_schedule_always_runs(tmp_path: Path) -> None:
    state = tmp_path / "gmail_ingestion_state.json"
    write_state(state, "2026-08-05T00:00:00Z")
    decision = decide_run(
        event_name="schedule",
        event_schedule=PRIMARY_SCHEDULE,
        state_path=state,
        now=datetime(2026, 8, 5, 0, 17, tzinfo=timezone.utc),
    )
    assert decision.should_run is True
    assert decision.reason == "primary_daily_schedule"


def test_evening_retry_skips_after_success_on_same_singapore_date(
    tmp_path: Path,
) -> None:
    state = tmp_path / "gmail_ingestion_state.json"
    write_state(state, "2026-08-05T00:20:00Z")
    decision = decide_run(
        event_name="schedule",
        event_schedule=RETRY_SCHEDULE,
        state_path=state,
        now=datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc),
    )
    assert decision.should_run is False
    assert decision.reason == "already_succeeded_today"


def test_evening_retry_runs_after_no_success_today(tmp_path: Path) -> None:
    state = tmp_path / "gmail_ingestion_state.json"
    write_state(state, "2026-08-04T12:48:00Z")
    decision = decide_run(
        event_name="schedule",
        event_schedule=RETRY_SCHEDULE,
        state_path=state,
        now=datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc),
    )
    assert decision.should_run is True
    assert decision.reason == "retry_after_no_success_today"


def test_evening_retry_runs_when_state_is_missing_or_malformed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert decide_run(
        event_name="schedule",
        event_schedule=RETRY_SCHEDULE,
        state_path=missing,
        now=datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc),
    ).should_run

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert decide_run(
        event_name="schedule",
        event_schedule=RETRY_SCHEDULE,
        state_path=malformed,
        now=datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc),
    ).should_run


def test_manual_dispatch_always_runs(tmp_path: Path) -> None:
    decision = decide_run(
        event_name="workflow_dispatch",
        event_schedule="",
        state_path=tmp_path / "state.json",
        now=datetime(2026, 8, 5, 12, 47, tzinfo=timezone.utc),
    )
    assert decision.should_run is True
    assert decision.reason == "manual_dispatch"


def test_workflow_wires_primary_and_retry_gate() -> None:
    workflow = (
        REPO_ROOT / ".github/workflows/daily-research-inbox.yml"
    ).read_text(encoding="utf-8")
    assert 'cron: "17 0 * * *"' in workflow
    assert 'cron: "47 12 * * *"' in workflow
    assert "scripts.ingest.scholar_schedule_gate" in workflow
    assert "steps.schedule_gate.outputs.should_run == 'true'" in workflow
    assert "primary run at 08:17; 20:47 is retry-only" in workflow

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PRIMARY_SCHEDULE = "17 0 * * *"
RETRY_SCHEDULE = "47 12 * * *"
LOCAL_TIMEZONE = ZoneInfo("Asia/Singapore")


@dataclass(frozen=True)
class GateDecision:
    should_run: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"should_run": self.should_run, "reason": self.reason}


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_last_successful_run(state_path: Path) -> datetime | None:
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    return parse_timestamp(state.get("last_successful_run_at"))


def decide_run(
    *,
    event_name: str,
    event_schedule: str,
    state_path: Path,
    now: datetime | None = None,
) -> GateDecision:
    if event_name == "workflow_dispatch":
        return GateDecision(True, "manual_dispatch")

    if event_name != "schedule":
        return GateDecision(False, "unsupported_event")

    if event_schedule == PRIMARY_SCHEDULE:
        return GateDecision(True, "primary_daily_schedule")

    if event_schedule != RETRY_SCHEDULE:
        return GateDecision(False, "unknown_schedule")

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_today = now.astimezone(LOCAL_TIMEZONE).date()
    previous = load_last_successful_run(state_path)
    if previous and previous.astimezone(LOCAL_TIMEZONE).date() == local_today:
        return GateDecision(False, "already_succeeded_today")
    return GateDecision(True, "retry_after_no_success_today")


def write_github_output(path: Path, decision: GateDecision) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"should_run={'true' if decision.should_run else 'false'}\n")
        handle.write(f"reason={decision.reason}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Scholar once daily, with an evening retry only after no success"
    )
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--event-schedule", default="")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    decision = decide_run(
        event_name=args.event_name,
        event_schedule=args.event_schedule,
        state_path=args.state_path,
    )
    if args.github_output:
        write_github_output(args.github_output, decision)
    print(json.dumps(decision.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

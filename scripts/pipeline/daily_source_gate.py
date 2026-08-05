from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PRIMARY_CRON = "17 0 * * *"
RETRY_CRON = "47 12 * * *"
LOCAL_TIMEZONE = ZoneInfo("Asia/Singapore")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def succeeded_today(state: dict[str, Any], now: datetime) -> bool:
    previous = parse_datetime(state.get("last_successful_run_at"))
    if previous is None:
        return False
    return previous.astimezone(LOCAL_TIMEZONE).date() == now.astimezone(LOCAL_TIMEZONE).date()


def write_outputs(path: Path | None, values: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                rendered = ",".join(str(item) for item in value)
            else:
                rendered = str(value)
            handle.write(f"{key}={rendered}\n")


def plan(
    *,
    gmail_state_path: Path,
    openalex_state_path: Path,
    event_name: str,
    event_schedule: str,
    now: datetime,
) -> dict[str, Any]:
    gmail_ok = succeeded_today(load_json(gmail_state_path), now)
    openalex_ok = succeeded_today(load_json(openalex_state_path), now)

    if event_name == "workflow_dispatch":
        run_scholar = True
        run_openalex = True
        mode = "manual"
    elif event_schedule == PRIMARY_CRON:
        run_scholar = True
        run_openalex = True
        mode = "primary"
    elif event_schedule == RETRY_CRON:
        run_scholar = not gmail_ok
        run_openalex = not openalex_ok
        mode = "retry"
    else:
        run_scholar = True
        run_openalex = True
        mode = "safe_default"

    return {
        "mode": mode,
        "run_scholar": run_scholar,
        "run_openalex": run_openalex,
        "should_run": run_scholar or run_openalex,
        "scholar_ready_before": gmail_ok,
        "openalex_ready_before": openalex_ok,
    }


def verify(
    *,
    gmail_state_path: Path,
    openalex_state_path: Path,
    now: datetime,
) -> dict[str, Any]:
    scholar_ready = succeeded_today(load_json(gmail_state_path), now)
    openalex_ready = succeeded_today(load_json(openalex_state_path), now)
    missing = [
        name
        for name, ready in (
            ("google_scholar", scholar_ready),
            ("openalex", openalex_ready),
        )
        if not ready
    ]
    return {
        "ready": not missing,
        "scholar_ready": scholar_ready,
        "openalex_ready": openalex_ready,
        "missing_sources": missing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and verify the unified daily discovery run")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--gmail-state-path", type=Path, required=True)
    plan_parser.add_argument("--openalex-state-path", type=Path, required=True)
    plan_parser.add_argument("--event-name", required=True)
    plan_parser.add_argument("--event-schedule", default="")
    plan_parser.add_argument("--github-output", type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--gmail-state-path", type=Path, required=True)
    verify_parser.add_argument("--openalex-state-path", type=Path, required=True)
    verify_parser.add_argument("--github-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    now = datetime.now(timezone.utc)
    if args.command == "plan":
        result = plan(
            gmail_state_path=args.gmail_state_path,
            openalex_state_path=args.openalex_state_path,
            event_name=args.event_name,
            event_schedule=args.event_schedule,
            now=now,
        )
    else:
        result = verify(
            gmail_state_path=args.gmail_state_path,
            openalex_state_path=args.openalex_state_path,
            now=now,
        )
    write_outputs(args.github_output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

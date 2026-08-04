from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.ingest.gmail_collector import (
    atomic_write_text,
    build_gmail_service,
    credentials_from_environment,
    fetch_raw_message,
    load_collector_config,
    load_json,
    parse_raw_gmail_message,
    validate_candidates,
    validate_source,
)
from scripts.ingest.scholar_parser import PARSER_VERSION, SourceContext, parse_alert_body


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def record_message_id(record: dict[str, Any]) -> str | None:
    source = record.get("source")
    if not isinstance(source, dict):
        return None
    value = source.get("message_id")
    return str(value) if value else None


def replace_scholar_records(
    existing: list[dict[str, Any]],
    *,
    target_message_ids: set[str],
    repaired: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    retained = [
        record
        for record in existing
        if record_message_id(record) not in target_message_ids
    ]
    output = list(retained)
    seen_fingerprints = {
        str(record.get("content_fingerprint"))
        for record in retained
        if record.get("content_fingerprint")
    }
    duplicate_count = 0
    for record in repaired:
        fingerprint = record.get("content_fingerprint")
        if fingerprint and str(fingerprint) in seen_fingerprints:
            duplicate_count += 1
            continue
        output.append(record)
        if fingerprint:
            seen_fingerprints.add(str(fingerprint))
    return output, duplicate_count


def write_registry(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(path, content)


def repair_once(
    *,
    service,
    config_path: Path,
    repository_root: Path,
    state_path: Path,
    registry_path: Path,
    repaired_at: str,
) -> dict[str, int]:
    config = load_collector_config(config_path, repository_root)
    state = load_json(state_path, {})
    processed = state.get("processed_message_ids", {})
    if not isinstance(processed, dict):
        raise ValueError("processed_message_ids must be an object")
    message_ids = list(processed)

    repaired: list[dict[str, Any]] = []
    for message_id in message_ids:
        raw_payload = fetch_raw_message(service, message_id)
        parsed = parse_raw_gmail_message(raw_payload)
        validate_source(parsed, config)
        context = SourceContext(
            message_id=parsed.gmail_message_id,
            thread_id=parsed.thread_id,
            received_at=parsed.received_at,
            sender=parsed.sender_header,
            subject=parsed.subject,
            spf=parsed.spf,
            dkim=parsed.dkim,
        )
        records = parse_alert_body(
            parsed.body,
            context,
            content_type=parsed.body_type,
            extracted_at=repaired_at,
        )
        if not records:
            raise ValueError("No paper candidates were extracted during registry repair")
        validate_candidates(records, config.schema_path)
        repaired.extend(records)

    existing = load_registry(registry_path)
    rebuilt, duplicate_count = replace_scholar_records(
        existing,
        target_message_ids=set(message_ids),
        repaired=repaired,
    )
    write_registry(registry_path, rebuilt)

    state["parser_version"] = PARSER_VERSION
    state["last_registry_repair_at"] = repaired_at
    state["last_registry_repair_summary"] = {
        "messages": len(message_ids),
        "repaired_candidates": len(repaired),
        "duplicates_removed": duplicate_count,
        "registry_records": len(rebuilt),
    }
    atomic_write_text(
        state_path,
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "messages": len(message_ids),
        "repaired_candidates": len(repaired),
        "duplicates_removed": duplicate_count,
        "registry_records": len(rebuilt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reparse previously processed Scholar alerts with the current parser"
    )
    parser.add_argument("--config", type=Path, default=Path("config/google_scholar_alerts.yaml"))
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("runtime-state/state/gmail_ingestion_state.json"),
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("runtime-state/data/paper_registry.jsonl"),
    )
    args = parser.parse_args()
    repository_root = Path.cwd()
    summary = repair_once(
        service=build_gmail_service(credentials_from_environment()),
        config_path=args.config,
        repository_root=repository_root,
        state_path=args.state_path,
        registry_path=args.registry_path,
        repaired_at=utc_now_text(),
    )
    print(json.dumps({"status": "completed", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

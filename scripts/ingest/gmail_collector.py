from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from jsonschema import Draft202012Validator, FormatChecker

from scripts.ingest.scholar_parser import SourceContext, parse_alert_body

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
TOKEN_URI = "https://oauth2.googleapis.com/token"
AUTH_RESULT_RE = re.compile(
    r"\b(?P<kind>spf|dkim)=(?P<state>pass|fail|neutral|softfail|none|temperror|permerror)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CollectorConfig:
    accepted_sender_addresses: frozenset[str]
    require_authentication_pass: frozenset[str]
    overlap_days: int
    initial_lookback_days: int
    max_messages_per_run: int
    processed_id_retention_days: int
    schema_path: Path


@dataclass(frozen=True)
class ParsedGmailMessage:
    gmail_message_id: str
    thread_id: str | None
    received_at: str
    sender_header: str
    sender_address: str
    subject: str
    spf: str | None
    dkim: str | None
    body: str
    body_type: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_collector_config(config_path: Path, repository_root: Path) -> CollectorConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    message_filter = raw.get("message_filter", {})
    authentication = message_filter.get("authentication", {})
    ingestion = raw.get("ingestion", {})
    output = raw.get("output", {})

    senders = {
        str(address).strip().lower()
        for address in message_filter.get("accepted_sender_addresses", [])
        if str(address).strip()
    }
    if not senders:
        raise ValueError("No accepted Scholar sender address is configured")

    schema_path = repository_root / output.get(
        "schema_path", "schemas/alert_candidate.schema.json"
    )
    return CollectorConfig(
        accepted_sender_addresses=frozenset(senders),
        require_authentication_pass=frozenset(
            str(value).lower()
            for value in authentication.get("require_at_least_one_pass", ["spf", "dkim"])
        ),
        overlap_days=int(ingestion.get("overlap_days", 7)),
        initial_lookback_days=int(ingestion.get("initial_lookback_days", 30)),
        max_messages_per_run=int(
            ingestion.get(
                "max_messages_per_run",
                raw.get("message_handling", {}).get("max_messages_per_run", 20),
            )
        ),
        processed_id_retention_days=int(
            ingestion.get("processed_id_retention_days", 120)
        ),
        schema_path=schema_path,
    )


def credentials_from_environment() -> Credentials:
    required = {
        "GMAIL_CLIENT_ID": os.environ.get("GMAIL_CLIENT_ID"),
        "GMAIL_CLIENT_SECRET": os.environ.get("GMAIL_CLIENT_SECRET"),
        "GMAIL_REFRESH_TOKEN": os.environ.get("GMAIL_REFRESH_TOKEN"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing required Gmail OAuth environment variables")

    return Credentials(
        token=None,
        refresh_token=required["GMAIL_REFRESH_TOKEN"],
        token_uri=TOKEN_URI,
        client_id=required["GMAIL_CLIENT_ID"],
        client_secret=required["GMAIL_CLIENT_SECRET"],
        scopes=[GMAIL_READONLY_SCOPE],
    )


def build_gmail_service(credentials: Credentials):
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def authentication_states(message: EmailMessage) -> dict[str, str | None]:
    header_values = list(message.get_all("Authentication-Results", []))
    header_values.extend(message.get_all("ARC-Authentication-Results", []))
    matches: dict[str, list[str]] = {"spf": [], "dkim": []}
    for header in header_values:
        for match in AUTH_RESULT_RE.finditer(str(header)):
            matches[match.group("kind").lower()].append(match.group("state").lower())

    result: dict[str, str | None] = {}
    for kind, states in matches.items():
        if "pass" in states:
            result[kind] = "pass"
        elif states:
            result[kind] = states[0]
        else:
            result[kind] = None
    return result


def decode_text_part(part: EmailMessage) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def extract_preferred_body(message: EmailMessage) -> tuple[str, str]:
    html_parts: list[str] = []
    text_parts: list[str] = []

    parts: Iterable[EmailMessage] = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.is_multipart():
            continue
        disposition = (part.get_content_disposition() or "").lower()
        if disposition == "attachment":
            continue
        content_type = part.get_content_type().lower()
        if content_type == "text/html":
            html_parts.append(decode_text_part(part))
        elif content_type == "text/plain":
            text_parts.append(decode_text_part(part))

    if html_parts:
        return "\n".join(html_parts), "html"
    if text_parts:
        return "\n".join(text_parts), "text"
    raise ValueError("Scholar email has no readable text or HTML body")


def message_received_at(message: EmailMessage, internal_date_ms: str | None) -> str:
    date_header = message.get("Date")
    if date_header:
        try:
            parsed = parsedate_to_datetime(str(date_header))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return isoformat_utc(parsed)
        except (TypeError, ValueError, OverflowError):
            pass
    if internal_date_ms:
        parsed = datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)
        return isoformat_utc(parsed)
    raise ValueError("Scholar email has no usable received timestamp")


def parse_raw_gmail_message(payload: dict[str, Any]) -> ParsedGmailMessage:
    raw = payload.get("raw")
    message_id = payload.get("id")
    if not raw or not message_id:
        raise ValueError("Gmail raw response is missing id or MIME payload")

    message = BytesParser(policy=policy.default).parsebytes(base64url_decode(raw))
    sender_header = str(message.get("From", "")).strip()
    sender_address = parseaddr(sender_header)[1].lower()
    subject = str(message.get("Subject", "")).strip()
    if not sender_address or not subject:
        raise ValueError("Scholar email is missing sender or subject")

    auth = authentication_states(message)
    body, body_type = extract_preferred_body(message)
    return ParsedGmailMessage(
        gmail_message_id=str(message_id),
        thread_id=str(payload.get("threadId")) if payload.get("threadId") else None,
        received_at=message_received_at(message, payload.get("internalDate")),
        sender_header=sender_header,
        sender_address=sender_address,
        subject=subject,
        spf=auth["spf"],
        dkim=auth["dkim"],
        body=body,
        body_type=body_type,
    )


def validate_source(message: ParsedGmailMessage, config: CollectorConfig) -> None:
    if message.sender_address not in config.accepted_sender_addresses:
        raise ValueError("Message sender is not an accepted Google Scholar sender")

    auth = {"spf": message.spf, "dkim": message.dkim}
    required = config.require_authentication_pass
    if required and not any(auth.get(kind) == "pass" for kind in required):
        raise ValueError("Message failed the configured SPF/DKIM requirement")


def build_search_query(state: dict[str, Any], config: CollectorConfig, now: datetime) -> str:
    checkpoint = parse_iso_datetime(state.get("last_successful_run_at"))
    if checkpoint:
        start = checkpoint - timedelta(days=config.overlap_days)
    else:
        start = now - timedelta(days=config.initial_lookback_days)

    sender_terms = " OR ".join(
        f"from:{address}" for address in sorted(config.accepted_sender_addresses)
    )
    return f"({sender_terms}) after:{start:%Y/%m/%d} -in:spam -in:trash"


def list_message_ids(service, query: str, max_messages: int) -> list[str]:
    message_ids: list[str] = []
    page_token: str | None = None
    while len(message_ids) < max_messages:
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=query,
                maxResults=min(100, max_messages - len(message_ids)),
                pageToken=page_token,
            )
            .execute()
        )
        for item in response.get("messages", []):
            if item.get("id"):
                message_ids.append(str(item["id"]))
                if len(message_ids) >= max_messages:
                    break
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return message_ids


def fetch_raw_message(service, message_id: str) -> dict[str, Any]:
    return (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="raw")
        .execute()
    )


def load_registry_fingerprints(registry_path: Path) -> set[str]:
    fingerprints: set[str] = set()
    if not registry_path.exists():
        return fingerprints
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        fingerprint = record.get("content_fingerprint")
        if fingerprint:
            fingerprints.add(str(fingerprint))
    return fingerprints


def append_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    addition = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(path, existing + addition)


def default_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "last_attempt_at": None,
        "last_successful_run_at": None,
        "consecutive_failures": 0,
        "processed_message_ids": {},
        "failed_message_ids": {},
    }


def prune_processed_ids(state: dict[str, Any], cutoff: datetime) -> None:
    processed = state.setdefault("processed_message_ids", {})
    state["processed_message_ids"] = {
        message_id: timestamp
        for message_id, timestamp in processed.items()
        if (parse_iso_datetime(timestamp) or cutoff) >= cutoff
    }


def validate_candidates(records: list[dict[str, Any]], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for record in records:
        if any(validator.iter_errors(record)):
            raise ValueError("Parsed candidate failed alert_candidate schema validation")


def collect_once(
    *,
    service,
    config: CollectorConfig,
    state_path: Path,
    registry_path: Path,
    now: datetime,
) -> dict[str, int]:
    state = load_json(state_path, default_state())
    processed: dict[str, str] = state.setdefault("processed_message_ids", {})
    failures: dict[str, dict[str, Any]] = state.setdefault("failed_message_ids", {})
    known_fingerprints = load_registry_fingerprints(registry_path)

    query = build_search_query(state, config, now)
    message_ids = list_message_ids(service, query, config.max_messages_per_run)
    pending_ids = [message_id for message_id in message_ids if message_id not in processed]

    new_records: list[dict[str, Any]] = []
    successful_messages = 0
    failed_messages = 0
    duplicate_candidates = 0
    now_text = isoformat_utc(now)

    for message_id in pending_ids:
        try:
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
                extracted_at=now_text,
            )
            if not records:
                raise ValueError("No paper candidates were extracted from Scholar email")
            validate_candidates(records, config.schema_path)

            for record in records:
                fingerprint = record.get("content_fingerprint")
                if fingerprint and fingerprint in known_fingerprints:
                    duplicate_candidates += 1
                    continue
                new_records.append(record)
                if fingerprint:
                    known_fingerprints.add(str(fingerprint))

            processed[message_id] = now_text
            failures.pop(message_id, None)
            successful_messages += 1
        except Exception as exc:  # noqa: BLE001 - isolate individual Gmail items
            previous = failures.get(message_id, {})
            failures[message_id] = {
                "attempt_count": int(previous.get("attempt_count", 0)) + 1,
                "last_attempt_at": now_text,
                "last_error_type": type(exc).__name__,
            }
            failed_messages += 1

    append_jsonl_records(registry_path, new_records)
    state["last_attempt_at"] = now_text
    if failed_messages == 0:
        state["last_successful_run_at"] = now_text
        state["consecutive_failures"] = 0
    else:
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1

    prune_processed_ids(
        state,
        now - timedelta(days=config.processed_id_retention_days),
    )
    atomic_write_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    return {
        "matched_messages": len(message_ids),
        "pending_messages": len(pending_ids),
        "successful_messages": successful_messages,
        "failed_messages": failed_messages,
        "new_candidates": len(new_records),
        "duplicate_candidates": duplicate_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally collect Google Scholar alert emails from Gmail"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/google_scholar_alerts.yaml"),
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    try:
        config = load_collector_config(args.config, repository_root)
        credentials = credentials_from_environment()
        service = build_gmail_service(credentials)
        summary = collect_once(
            service=service,
            config=config,
            state_path=args.state_dir / "gmail_ingestion_state.json",
            registry_path=args.output_dir / "paper_registry.jsonl",
            now=utc_now(),
        )
    except Exception as exc:  # noqa: BLE001 - sanitize top-level failure output
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__},
                sort_keys=True,
            )
        )
        return 3

    print(json.dumps({"status": "completed", **summary}, sort_keys=True))
    return 2 if summary["failed_messages"] else 0


if __name__ == "__main__":
    sys.exit(main())

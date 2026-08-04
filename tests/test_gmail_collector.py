from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

from scripts.ingest.gmail_collector import (
    CollectorConfig,
    build_search_query,
    collect_once,
    default_state,
    parse_raw_gmail_message,
    validate_source,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "alert_candidate.schema.json"


class FakeRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class FakeMessages:
    def __init__(self, payloads):
        self.payloads = payloads

    def list(self, **kwargs):
        del kwargs
        return FakeRequest(
            {"messages": [{"id": message_id} for message_id in self.payloads]}
        )

    def get(self, *, id, **kwargs):
        del kwargs
        return FakeRequest(self.payloads[id])


class FakeUsers:
    def __init__(self, payloads):
        self.payloads = payloads

    def messages(self):
        return FakeMessages(self.payloads)


class FakeService:
    def __init__(self, payloads):
        self.payloads = payloads

    def users(self):
        return FakeUsers(self.payloads)


def collector_config() -> CollectorConfig:
    return CollectorConfig(
        accepted_sender_addresses=frozenset(
            {"scholaralerts-noreply@google.com"}
        ),
        require_authentication_pass=frozenset({"spf", "dkim"}),
        overlap_days=7,
        initial_lookback_days=30,
        max_messages_per_run=20,
        processed_id_retention_days=120,
        schema_path=SCHEMA,
    )


def raw_scholar_payload(
    *,
    message_id: str = "gmail-message-001",
    sender: str = '"Google 学术搜索快讯" <scholaralerts-noreply@google.com>',
    authentication_results: str = "mx.google.com; dkim=pass; spf=pass",
) -> dict:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "researcher@example.com"
    message["Subject"] = '"optical neural network" - 新的结果'
    message["Date"] = format_datetime(
        datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)
    )
    message["Authentication-Results"] = authentication_results
    message.set_content("HTML fallback")
    message.add_alternative(
        """
        <html><body>
          <div class="result">
            <a href="https://scholar.google.com/scholar_url?url=https%3A%2F%2Fexample.org%2Fpaper">
              Zeroth-order optimization of photonic neural networks
            </a>
            <div>A Author, B Author - Optica, 2026</div>
            <div>A query-efficient black-box method for physical optical hardware.</div>
            <a href="https://scholar.google.com/citations?update_op=email_library_add">保存</a>
          </div>
        </body></html>
        """,
        subtype="html",
    )
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode().rstrip("=")
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "internalDate": "1785747600000",
        "raw": raw,
    }


def test_parse_and_validate_raw_gmail_message() -> None:
    parsed = parse_raw_gmail_message(raw_scholar_payload())
    assert parsed.gmail_message_id == "gmail-message-001"
    assert parsed.sender_address == "scholaralerts-noreply@google.com"
    assert parsed.spf == "pass"
    assert parsed.dkim == "pass"
    assert parsed.body_type == "html"
    assert "photonic neural networks" in parsed.body
    validate_source(parsed, collector_config())


def test_search_query_overlaps_checkpoint() -> None:
    state = default_state()
    state["last_successful_run_at"] = "2026-08-03T00:00:00Z"
    query = build_search_query(
        state,
        collector_config(),
        datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
    )
    assert "from:scholaralerts-noreply@google.com" in query
    assert "after:2026/07/27" in query
    assert "-in:spam" in query


def test_collect_once_is_incremental_and_idempotent(tmp_path: Path) -> None:
    payload = raw_scholar_payload()
    service = FakeService({payload["id"]: payload})
    state_path = tmp_path / "state" / "gmail_ingestion_state.json"
    registry_path = tmp_path / "data" / "paper_registry.jsonl"
    now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)

    first = collect_once(
        service=service,
        config=collector_config(),
        state_path=state_path,
        registry_path=registry_path,
        now=now,
    )
    assert first["successful_messages"] == 1
    assert first["failed_messages"] == 0
    assert first["new_candidates"] == 1

    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["title"] == (
        "Zeroth-order optimization of photonic neural networks"
    )

    second = collect_once(
        service=service,
        config=collector_config(),
        state_path=state_path,
        registry_path=registry_path,
        now=now,
    )
    assert second["pending_messages"] == 0
    assert second["new_candidates"] == 0
    assert len(registry_path.read_text(encoding="utf-8").splitlines()) == 1


def test_failed_message_does_not_advance_success_checkpoint(tmp_path: Path) -> None:
    payload = raw_scholar_payload(authentication_results="mx.google.com; spf=fail")
    service = FakeService({payload["id"]: payload})
    state_path = tmp_path / "state" / "gmail_ingestion_state.json"
    registry_path = tmp_path / "data" / "paper_registry.jsonl"

    summary = collect_once(
        service=service,
        config=collector_config(),
        state_path=state_path,
        registry_path=registry_path,
        now=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
    )
    assert summary["failed_messages"] == 1
    assert summary["successful_messages"] == 0
    assert not registry_path.exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_successful_run_at"] is None
    assert state["consecutive_failures"] == 1
    assert payload["id"] not in state["processed_message_ids"]
    assert state["failed_message_ids"][payload["id"]]["attempt_count"] == 1

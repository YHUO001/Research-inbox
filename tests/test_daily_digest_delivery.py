from __future__ import annotations

import base64
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path

from scripts.delivery.send_daily_digest import send_daily_digest


ROOT = Path(__file__).resolve().parents[1]
RECIPIENT = "a209072780@126.com"


class FakeRequest:
    def __init__(self, response: dict) -> None:
        self.response = response

    def execute(self) -> dict:
        return self.response


class FakeMessages:
    def __init__(self, *, existing: bool = False) -> None:
        self.existing = existing
        self.list_calls: list[dict] = []
        self.send_calls: list[dict] = []

    def list(self, *, userId: str, q: str, maxResults: int) -> FakeRequest:
        self.list_calls.append({"userId": userId, "q": q, "maxResults": maxResults})
        if self.existing:
            return FakeRequest({"messages": [{"id": "existing-id", "threadId": "thread-id"}]})
        return FakeRequest({"messages": []})

    def send(self, *, userId: str, body: dict[str, str]) -> FakeRequest:
        self.send_calls.append({"userId": userId, "body": body})
        return FakeRequest({"id": "sent-id", "threadId": "thread-id"})


class FakeUsers:
    def __init__(self, messages: FakeMessages) -> None:
        self._messages = messages

    def messages(self) -> FakeMessages:
        return self._messages


class FakeService:
    def __init__(self, *, existing: bool = False) -> None:
        self.message_api = FakeMessages(existing=existing)

    def users(self) -> FakeUsers:
        return FakeUsers(self.message_api)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_digest(state_root: Path, date: str = "2026-08-05") -> None:
    write_json(
        state_root / f"data/digests/{date}.generated.json",
        {
            "status": "completed_automatic",
            "summary_count": 2,
            "summaries": [{"candidate_id": "one"}, {"candidate_id": "two"}],
        },
    )
    markdown = state_root / f"data/digests/{date}.generated.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        "# Research Inbox — 2026-08-05\n\n- DOI：[10.1000/a](https://doi.org/10.1000/a)\n",
        encoding="utf-8",
    )


def test_sends_one_daily_message_and_records_only_hashes(tmp_path: Path) -> None:
    prepare_digest(tmp_path)
    service = FakeService()

    result = send_daily_digest(
        config_path=ROOT / "config/email_delivery.yaml",
        state_root=tmp_path,
        digest_date="2026-08-05",
        service=service,
    )

    assert result["status"] == "sent"
    assert result["summary_count"] == 2
    assert len(service.message_api.list_calls) == 1
    assert len(service.message_api.send_calls) == 1
    assert "rfc822msgid:<research-inbox-20260805-" in service.message_api.list_calls[0]["q"]

    raw = base64.urlsafe_b64decode(service.message_api.send_calls[0]["body"]["raw"])
    parsed = BytesParser(policy=policy.default).parsebytes(raw)
    assert parsed["To"] == RECIPIENT
    assert parsed["Subject"] == "[Research Inbox] 每日研究汇总 2026-08-05（2 篇）"
    assert parsed["Message-ID"].startswith("<research-inbox-20260805-")
    assert "长期知识库索引" in parsed.get_content()

    state_text = (tmp_path / "state/email_delivery_state.json").read_text(encoding="utf-8")
    assert RECIPIENT not in state_text
    state = json.loads(state_text)
    record = state["sent_digests"]["2026-08-05"]
    assert record["status"] == "sent"
    assert record["summary_count"] == 2
    assert record["recipient_sha256"]
    assert record["gmail_message_id_sha256"]

    repeated = send_daily_digest(
        config_path=ROOT / "config/email_delivery.yaml",
        state_root=tmp_path,
        digest_date="2026-08-05",
        service=service,
    )
    assert repeated["status"] == "skipped_already_recorded"
    assert len(service.message_api.list_calls) == 1
    assert len(service.message_api.send_calls) == 1


def test_recovers_idempotency_from_sent_mail_when_state_is_missing(tmp_path: Path) -> None:
    prepare_digest(tmp_path)
    service = FakeService(existing=True)

    result = send_daily_digest(
        config_path=ROOT / "config/email_delivery.yaml",
        state_root=tmp_path,
        digest_date="2026-08-05",
        service=service,
    )

    assert result["status"] == "already_sent"
    assert result["idempotent_recovery"] is True
    assert len(service.message_api.list_calls) == 1
    assert service.message_api.send_calls == []
    state = json.loads(
        (tmp_path / "state/email_delivery_state.json").read_text(encoding="utf-8")
    )
    assert state["sent_digests"]["2026-08-05"]["status"] == "already_sent"


def test_sends_an_empty_daily_digest_when_no_summary_slot_exists(tmp_path: Path) -> None:
    write_json(
        tmp_path / "state/selection_manifest.json",
        {"summary_slot_count": 0, "eligible_candidate_count": 0},
    )
    write_json(
        tmp_path / "state/unified_registry_manifest.json",
        {"unified_candidate_count": 12, "merged_group_count": 2},
    )
    write_json(
        tmp_path / "state/openalex_discovery_manifest.json",
        {"accepted_count": 0},
    )
    write_json(
        tmp_path / "state/routing_manifest.json",
        {"route_counts": {"metadata_enrichment_queue": 1, "manual_review_queue": 0}},
    )
    service = FakeService()

    result = send_daily_digest(
        config_path=ROOT / "config/email_delivery.yaml",
        state_root=tmp_path,
        digest_date="2026-08-05",
        service=service,
    )

    assert result["status"] == "sent"
    assert result["summary_count"] == 0
    assert result["content_status"] == "empty_daily_digest"
    assert len(service.message_api.send_calls) == 1
    raw = base64.urlsafe_b64decode(service.message_api.send_calls[0]["body"]["raw"])
    parsed = BytesParser(policy=policy.default).parsebytes(raw)
    assert "今天没有新的论文进入自动摘要名额" in parsed.get_content()
    assert "没有产生模型 token 费用" in parsed.get_content()


def test_missing_digest_is_an_error_when_summaries_were_selected(tmp_path: Path) -> None:
    write_json(tmp_path / "state/selection_manifest.json", {"summary_slot_count": 1})
    service = FakeService()

    try:
        send_daily_digest(
            config_path=ROOT / "config/email_delivery.yaml",
            state_root=tmp_path,
            digest_date="2026-08-05",
            service=service,
        )
    except RuntimeError as error:
        assert "completed digest artifact is missing" in str(error)
    else:
        raise AssertionError("Expected missing selected digest to fail")

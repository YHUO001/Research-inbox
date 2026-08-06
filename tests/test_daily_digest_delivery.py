from __future__ import annotations

import base64
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path

from scripts.delivery.send_daily_digest import send_daily_digest, sha256_text


ROOT = Path(__file__).resolve().parents[1]
RECIPIENT = "a209072780@126.com"


class FakeRequest:
    def __init__(self, response: dict) -> None:
        self.response = response

    def execute(self) -> dict:
        return self.response


class FakeMessages:
    def __init__(self, *, existing: bool = False, fail_send: bool = False) -> None:
        self.existing = existing
        self.fail_send = fail_send
        self.list_calls: list[dict] = []
        self.send_calls: list[dict] = []

    def list(self, *, userId: str, q: str, maxResults: int) -> FakeRequest:
        self.list_calls.append({"userId": userId, "q": q, "maxResults": maxResults})
        if self.existing:
            return FakeRequest({"messages": [{"id": "existing-id", "threadId": "thread-id"}]})
        return FakeRequest({"messages": []})

    def send(self, *, userId: str, body: dict[str, str]) -> FakeRequest:
        self.send_calls.append({"userId": userId, "body": body})
        if self.fail_send:
            raise RuntimeError("simulated Gmail failure")
        return FakeRequest({"id": "sent-id", "threadId": "thread-id"})


class FakeUsers:
    def __init__(self, messages: FakeMessages) -> None:
        self._messages = messages

    def messages(self) -> FakeMessages:
        return self._messages


class FakeService:
    def __init__(self, *, existing: bool = False, fail_send: bool = False) -> None:
        self.message_api = FakeMessages(existing=existing, fail_send=fail_send)

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


def archive_paths(state_root: Path, date: str = "2026-08-05") -> tuple[Path, Path]:
    root = state_root / "data/digest_archive" / date[:4]
    return root / f"{date}.email.md", root / f"{date}.delivery.json"


def test_sends_one_daily_message_and_archives_exact_body(tmp_path: Path) -> None:
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
    assert result["archive_status"] == "archived"
    assert len(service.message_api.list_calls) == 1
    assert len(service.message_api.send_calls) == 1
    assert "rfc822msgid:<research-inbox-20260805-" in service.message_api.list_calls[0]["q"]

    raw = base64.urlsafe_b64decode(service.message_api.send_calls[0]["body"]["raw"])
    parsed = BytesParser(policy=policy.default).parsebytes(raw)
    assert parsed["To"] == RECIPIENT
    assert parsed["Subject"] == "[Research Inbox] 每日研究汇总 2026-08-05（2 篇）"
    assert parsed["Message-ID"].startswith("<research-inbox-20260805-")
    assert "长期知识库索引" in parsed.get_content()

    archive_body_path, archive_metadata_path = archive_paths(tmp_path)
    archive_body = archive_body_path.read_text(encoding="utf-8")
    assert archive_body.rstrip("\n") == parsed.get_content().rstrip("\n")
    archive_metadata = json.loads(archive_metadata_path.read_text(encoding="utf-8"))
    assert archive_metadata["delivery_status"] == "sent"
    assert archive_metadata["candidate_ids"] == ["one", "two"]
    assert archive_metadata["summary_count"] == 2
    assert archive_metadata["content_status"] == "completed_digest"
    assert archive_metadata["body_sha256"] == sha256_text(archive_body)
    assert archive_metadata["source_digest_file"] == "data/digests/2026-08-05.generated.md"
    assert archive_metadata["full_text_persisted"] is False
    assert RECIPIENT not in archive_metadata_path.read_text(encoding="utf-8")

    state_text = (tmp_path / "state/email_delivery_state.json").read_text(encoding="utf-8")
    assert RECIPIENT not in state_text
    state = json.loads(state_text)
    record = state["sent_digests"]["2026-08-05"]
    assert record["status"] == "sent"
    assert record["summary_count"] == 2
    assert record["recipient_sha256"]
    assert record["gmail_message_id_sha256"]
    assert record["archive_body_file"] == "data/digest_archive/2026/2026-08-05.email.md"
    assert record["archive_metadata_file"] == "data/digest_archive/2026/2026-08-05.delivery.json"

    repeated = send_daily_digest(
        config_path=ROOT / "config/email_delivery.yaml",
        state_root=tmp_path,
        digest_date="2026-08-05",
        service=service,
    )
    assert repeated["status"] == "skipped_already_recorded"
    assert repeated["archive_status"] == "already_archived"
    assert len(service.message_api.list_calls) == 1
    assert len(service.message_api.send_calls) == 1


def test_recovers_idempotency_from_sent_mail_and_creates_archive(tmp_path: Path) -> None:
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
    assert result["archive_status"] == "archived"
    assert len(service.message_api.list_calls) == 1
    assert service.message_api.send_calls == []
    state = json.loads(
        (tmp_path / "state/email_delivery_state.json").read_text(encoding="utf-8")
    )
    assert state["sent_digests"]["2026-08-05"]["status"] == "already_sent"
    archive_body_path, archive_metadata_path = archive_paths(tmp_path)
    assert archive_body_path.exists()
    metadata = json.loads(archive_metadata_path.read_text(encoding="utf-8"))
    assert metadata["delivery_status"] == "already_sent"
    assert metadata["candidate_ids"] == ["one", "two"]


def test_sends_and_archives_an_empty_daily_digest(tmp_path: Path) -> None:
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
    assert result["archive_status"] == "archived"
    assert len(service.message_api.send_calls) == 1
    raw = base64.urlsafe_b64decode(service.message_api.send_calls[0]["body"]["raw"])
    parsed = BytesParser(policy=policy.default).parsebytes(raw)
    assert "今天没有新的论文进入自动摘要名额" in parsed.get_content()
    assert "没有产生模型 token 费用" in parsed.get_content()
    archive_body_path, archive_metadata_path = archive_paths(tmp_path)
    assert "今天没有新的论文进入自动摘要名额" in archive_body_path.read_text(
        encoding="utf-8"
    )
    metadata = json.loads(archive_metadata_path.read_text(encoding="utf-8"))
    assert metadata["candidate_ids"] == []
    assert metadata["source_digest_file"] is None


def test_failed_delivery_does_not_create_sent_archive(tmp_path: Path) -> None:
    prepare_digest(tmp_path)
    service = FakeService(fail_send=True)

    try:
        send_daily_digest(
            config_path=ROOT / "config/email_delivery.yaml",
            state_root=tmp_path,
            digest_date="2026-08-05",
            service=service,
        )
    except RuntimeError as error:
        assert "simulated Gmail failure" in str(error)
    else:
        raise AssertionError("Expected Gmail delivery to fail")

    archive_body_path, archive_metadata_path = archive_paths(tmp_path)
    assert not archive_body_path.exists()
    assert not archive_metadata_path.exists()
    state = json.loads(
        (tmp_path / "state/email_delivery_state.json").read_text(encoding="utf-8")
    )
    assert state["last_failure"]["digest_date"] == "2026-08-05"


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

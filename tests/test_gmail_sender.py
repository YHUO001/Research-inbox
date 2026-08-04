from __future__ import annotations

import base64
from email import policy
from email.parser import BytesParser

import pytest

from scripts.delivery.gmail_sender import (
    EmailDeliveryPolicy,
    compose_message,
    encode_message,
    send_digest,
    validate_delivery,
)


class FakeSendRequest:
    def __init__(self, response: dict[str, str]) -> None:
        self.response = response

    def execute(self) -> dict[str, str]:
        return self.response


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send(self, *, userId: str, body: dict[str, str]) -> FakeSendRequest:
        self.calls.append({"userId": userId, "body": body})
        return FakeSendRequest({"id": "gmail-message-id", "threadId": "thread-id"})


class FakeUsers:
    def __init__(self, messages: FakeMessages) -> None:
        self._messages = messages

    def messages(self) -> FakeMessages:
        return self._messages


class FakeService:
    def __init__(self) -> None:
        self.message_api = FakeMessages()

    def users(self) -> FakeUsers:
        return FakeUsers(self.message_api)


def enabled_policy() -> EmailDeliveryPolicy:
    return EmailDeliveryPolicy(
        enabled=True,
        allowed_recipients=frozenset({"destination@example.com"}),
        subject_prefix="[Research Inbox]",
    )


def test_compose_message_contains_expected_headers_and_bodies() -> None:
    message = compose_message(
        recipient="Destination@Example.com",
        subject="Daily digest",
        text_body="Plain summary",
        html_body="<p>HTML summary</p>",
    )

    assert message["To"] == "destination@example.com"
    assert message["Subject"] == "[Research Inbox] Daily digest"
    assert message.is_multipart()
    assert "Plain summary" in message.get_body(preferencelist=("plain",)).get_content()
    assert "HTML summary" in message.get_body(preferencelist=("html",)).get_content()


def test_validate_delivery_rejects_non_allowlisted_recipient() -> None:
    with pytest.raises(PermissionError):
        validate_delivery(
            recipient="other@example.com",
            subject="Digest",
            text_body="Summary",
            html_body=None,
            policy=enabled_policy(),
        )


def test_validate_delivery_rejects_disabled_delivery() -> None:
    policy_value = EmailDeliveryPolicy(
        enabled=False,
        allowed_recipients=frozenset({"destination@example.com"}),
        subject_prefix="[Research Inbox]",
    )
    with pytest.raises(RuntimeError, match="disabled"):
        validate_delivery(
            recipient="destination@example.com",
            subject="Digest",
            text_body="Summary",
            html_body=None,
            policy=policy_value,
        )


def test_send_digest_calls_gmail_once() -> None:
    service = FakeService()
    result = send_digest(
        service=service,
        recipient="destination@example.com",
        subject="Daily digest",
        text_body="One paper was selected.",
        policy=enabled_policy(),
    )

    assert result == {"message_id": "gmail-message-id", "thread_id": "thread-id"}
    assert len(service.message_api.calls) == 1
    call = service.message_api.calls[0]
    assert call["userId"] == "me"

    raw = base64.urlsafe_b64decode(call["body"]["raw"])
    parsed = BytesParser(policy=policy.default).parsebytes(raw)
    assert parsed["To"] == "destination@example.com"
    assert parsed["Subject"] == "[Research Inbox] Daily digest"
    assert "One paper was selected." in parsed.get_content()


def test_encode_message_is_base64url() -> None:
    message = compose_message(
        recipient="destination@example.com",
        subject="Digest",
        text_body="Summary",
    )
    encoded = encode_message(message)
    decoded = base64.urlsafe_b64decode(encoded)
    assert b"destination@example.com" in decoded

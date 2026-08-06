from __future__ import annotations

import argparse
import base64
import os
import re
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
GMAIL_SCOPES = (GMAIL_READONLY_SCOPE, GMAIL_SEND_SCOPE)
TOKEN_URI = "https://oauth2.googleapis.com/token"

ENABLED_ENV = "RESEARCH_INBOX_EMAIL_ENABLED"
RECIPIENTS_ENV = "RESEARCH_INBOX_EMAIL_RECIPIENTS"
SUBJECT_PREFIX_ENV = "RESEARCH_INBOX_EMAIL_SUBJECT_PREFIX"
DEFAULT_SUBJECT_PREFIX = "[Research Inbox]"
MAX_BODY_CHARACTERS = 100_000
_MESSAGE_ID_PATTERN = re.compile(r"^<[A-Za-z0-9._+@-]+>$")


@dataclass(frozen=True)
class EmailDeliveryPolicy:
    enabled: bool
    allowed_recipients: frozenset[str]
    subject_prefix: str


def env_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_email_address(value: str) -> str:
    display_name, address = parseaddr(value.strip())
    del display_name
    normalized = address.strip().lower()
    if not normalized or "@" not in normalized or normalized.startswith("@"):
        raise ValueError("Invalid email address")
    local_part, domain = normalized.rsplit("@", 1)
    if not local_part or "." not in domain or domain.startswith(".") or domain.endswith("."):
        raise ValueError("Invalid email address")
    return normalized


def normalize_message_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not _MESSAGE_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Invalid deterministic Message-ID")
    return normalized


def parse_recipient_allowlist(value: str | None) -> frozenset[str]:
    if not value or not value.strip():
        return frozenset()
    recipients = {
        normalize_email_address(item)
        for item in value.split(",")
        if item.strip()
    }
    return frozenset(recipients)


def policy_from_environment() -> EmailDeliveryPolicy:
    subject_prefix = os.environ.get(SUBJECT_PREFIX_ENV, DEFAULT_SUBJECT_PREFIX).strip()
    delivery_policy = EmailDeliveryPolicy(
        enabled=env_flag(os.environ.get(ENABLED_ENV)),
        allowed_recipients=parse_recipient_allowlist(os.environ.get(RECIPIENTS_ENV)),
        subject_prefix=subject_prefix,
    )
    if delivery_policy.enabled and not delivery_policy.allowed_recipients:
        raise RuntimeError(
            f"{ENABLED_ENV} is enabled but {RECIPIENTS_ENV} is empty"
        )
    return delivery_policy


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
        scopes=list(GMAIL_SCOPES),
    )


def build_gmail_service(credentials: Credentials):
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def validate_delivery(
    *,
    recipient: str,
    subject: str,
    text_body: str,
    html_body: str | None,
    policy: EmailDeliveryPolicy,
    require_enabled: bool = True,
) -> str:
    normalized_recipient = normalize_email_address(recipient)
    if require_enabled and not policy.enabled:
        raise RuntimeError("Email delivery is disabled")
    if normalized_recipient not in policy.allowed_recipients:
        raise PermissionError("Recipient is not in the configured allowlist")
    if not subject.strip():
        raise ValueError("Email subject must not be empty")
    if not text_body.strip():
        raise ValueError("Email body must not be empty")
    if len(text_body) > MAX_BODY_CHARACTERS:
        raise ValueError("Email text body exceeds the configured safety limit")
    if html_body is not None and len(html_body) > MAX_BODY_CHARACTERS:
        raise ValueError("Email HTML body exceeds the configured safety limit")
    return normalized_recipient


def compose_message(
    *,
    recipient: str,
    subject: str,
    text_body: str,
    subject_prefix: str = DEFAULT_SUBJECT_PREFIX,
    html_body: str | None = None,
    message_id: str | None = None,
) -> EmailMessage:
    message = EmailMessage()
    message["To"] = normalize_email_address(recipient)
    full_subject = f"{subject_prefix} {subject.strip()}".strip()
    message["Subject"] = full_subject
    if message_id is not None:
        message["Message-ID"] = normalize_message_id(message_id)
    message.set_content(text_body)
    if html_body is not None:
        message.add_alternative(html_body, subtype="html")
    return message


def encode_message(message: EmailMessage) -> str:
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def decode_message(value: str) -> EmailMessage:
    encoded = str(value or "").strip()
    if not encoded:
        raise RuntimeError("Gmail sent message is missing raw MIME content")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as error:
        raise RuntimeError("Gmail sent message contains invalid raw MIME content") from error
    return BytesParser(policy=policy.default).parsebytes(raw)


def extract_plain_text(message: EmailMessage) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() != "text/plain":
                continue
            if part.get_content_disposition() == "attachment":
                continue
            value = part.get_content()
            if isinstance(value, str):
                return value
    elif message.get_content_type() == "text/plain":
        value = message.get_content()
        if isinstance(value, str):
            return value
    raise RuntimeError("Gmail sent message does not contain a text/plain body")


def fetch_sent_message(service: Any, *, gmail_message_id: str) -> dict[str, str | None]:
    response = (
        service.users()
        .messages()
        .get(userId="me", id=gmail_message_id, format="raw")
        .execute()
    )
    message = decode_message(str(response.get("raw") or ""))
    return {
        "text_body": extract_plain_text(message),
        "subject": str(message.get("Subject") or "") or None,
        "rfc822_message_id": str(message.get("Message-ID") or "") or None,
    }


def find_sent_message(service: Any, *, message_id: str) -> dict[str, str | None] | None:
    normalized = normalize_message_id(message_id)
    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            q=f"in:sent rfc822msgid:{normalized}",
            maxResults=1,
        )
        .execute()
    )
    messages = response.get("messages") or []
    if not messages:
        return None
    first = messages[0]
    gmail_message_id = str(first.get("id") or "")
    if not gmail_message_id:
        raise RuntimeError("Gmail sent-message search returned an empty message id")
    fetched = fetch_sent_message(service, gmail_message_id=gmail_message_id)
    return {
        "message_id": gmail_message_id,
        "thread_id": str(first.get("threadId") or "") or None,
        **fetched,
    }


def send_digest(
    *,
    service: Any,
    recipient: str,
    subject: str,
    text_body: str,
    policy: EmailDeliveryPolicy,
    html_body: str | None = None,
    message_id: str | None = None,
) -> dict[str, str | None]:
    normalized_recipient = validate_delivery(
        recipient=recipient,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        policy=policy,
        require_enabled=True,
    )
    message = compose_message(
        recipient=normalized_recipient,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        subject_prefix=policy.subject_prefix,
        message_id=message_id,
    )
    response = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": encode_message(message)})
        .execute()
    )
    return {
        "message_id": str(response.get("id")) if response.get("id") else None,
        "thread_id": str(response.get("threadId")) if response.get("threadId") else None,
    }


def send_digest_idempotent(
    *,
    service: Any,
    recipient: str,
    subject: str,
    text_body: str,
    policy: EmailDeliveryPolicy,
    message_id: str,
    html_body: str | None = None,
) -> dict[str, str | None]:
    validate_delivery(
        recipient=recipient,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        policy=policy,
        require_enabled=True,
    )
    existing = find_sent_message(service, message_id=message_id)
    if existing is not None:
        return {
            "status": "already_sent",
            **existing,
        }
    result = send_digest(
        service=service,
        recipient=recipient,
        subject=subject,
        text_body=text_body,
        policy=policy,
        html_body=html_body,
        message_id=message_id,
    )
    return {"status": "sent", **result}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one allowlisted Research Inbox digest through Gmail"
    )
    parser.add_argument("--to", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", type=Path, required=True)
    parser.add_argument("--html-file", type=Path)
    parser.add_argument("--message-id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate recipient and content without contacting Gmail",
    )
    args = parser.parse_args()

    text_body = args.body_file.read_text(encoding="utf-8")
    html_body = (
        args.html_file.read_text(encoding="utf-8") if args.html_file else None
    )
    delivery_policy = policy_from_environment()
    validate_delivery(
        recipient=args.to,
        subject=args.subject,
        text_body=text_body,
        html_body=html_body,
        policy=delivery_policy,
        require_enabled=not args.dry_run,
    )

    if args.dry_run:
        print("Email delivery dry run passed; no message was sent.")
        return 0

    service = build_gmail_service(credentials_from_environment())
    if args.message_id:
        send_digest_idempotent(
            service=service,
            recipient=args.to,
            subject=args.subject,
            text_body=text_body,
            html_body=html_body,
            policy=delivery_policy,
            message_id=args.message_id,
        )
    else:
        send_digest(
            service=service,
            recipient=args.to,
            subject=args.subject,
            text_body=text_body,
            html_body=html_body,
            policy=delivery_policy,
        )
    print("Email delivery completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

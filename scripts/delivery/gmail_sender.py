from __future__ import annotations

import argparse
import base64
import os
from dataclasses import dataclass
from email.message import EmailMessage
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
    policy = EmailDeliveryPolicy(
        enabled=env_flag(os.environ.get(ENABLED_ENV)),
        allowed_recipients=parse_recipient_allowlist(os.environ.get(RECIPIENTS_ENV)),
        subject_prefix=subject_prefix,
    )
    if policy.enabled and not policy.allowed_recipients:
        raise RuntimeError(
            f"{ENABLED_ENV} is enabled but {RECIPIENTS_ENV} is empty"
        )
    return policy


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
) -> EmailMessage:
    message = EmailMessage()
    message["To"] = normalize_email_address(recipient)
    full_subject = f"{subject_prefix} {subject.strip()}".strip()
    message["Subject"] = full_subject
    message.set_content(text_body)
    if html_body is not None:
        message.add_alternative(html_body, subtype="html")
    return message


def encode_message(message: EmailMessage) -> str:
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def send_digest(
    *,
    service: Any,
    recipient: str,
    subject: str,
    text_body: str,
    policy: EmailDeliveryPolicy,
    html_body: str | None = None,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one allowlisted Research Inbox digest through Gmail"
    )
    parser.add_argument("--to", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", type=Path, required=True)
    parser.add_argument("--html-file", type=Path)
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

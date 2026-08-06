from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from scripts.delivery.gmail_sender import (
    EmailDeliveryPolicy,
    build_gmail_service,
    credentials_from_environment,
    normalize_email_address,
    send_digest_idempotent,
)
from scripts.summarize.prepare_digest import atomic_write, load_json, stable_json

LOCAL_TIMEZONE = ZoneInfo("Asia/Singapore")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_delivery_config(path: Path) -> tuple[EmailDeliveryPolicy, str, bool, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Email delivery config must be a YAML object")
    if int(config.get("delivery_version") or 0) < 2:
        raise RuntimeError("Daily automatic delivery requires delivery_version 2")
    if config.get("enabled_by_default") is not True:
        raise RuntimeError("Daily email delivery is not enabled")

    recipient_config = config.get("recipient_policy") or {}
    recipients = recipient_config.get("recipients") or []
    if not isinstance(recipients, list) or len(recipients) != 1:
        raise RuntimeError("Exactly one daily digest recipient must be configured")
    recipient = normalize_email_address(str(recipients[0]))

    message_policy = config.get("message_policy") or {}
    if message_policy.get("delivery_mode") != "daily_digest_only":
        raise RuntimeError("Only daily_digest_only delivery is supported")
    if int(message_policy.get("maximum_messages_per_digest") or 0) != 1:
        raise RuntimeError("Daily delivery must allow exactly one message per digest")
    subject_prefix = str(message_policy.get("subject_prefix") or "[Research Inbox]")

    schedule = config.get("schedule") or {}
    send_empty = bool(schedule.get("send_empty_digest"))
    state_value = str((config.get("idempotency") or {}).get("state_path") or "")
    if not state_value:
        raise RuntimeError("Email idempotency state_path is required")

    policy = EmailDeliveryPolicy(
        enabled=True,
        allowed_recipients=frozenset({recipient}),
        subject_prefix=subject_prefix,
    )
    return policy, recipient, send_empty, state_value


def deterministic_message_id(digest_date: str, recipient: str) -> str:
    recipient_key = sha256_text(normalize_email_address(recipient))[:16]
    safe_date = digest_date.replace("-", "")
    return f"<research-inbox-{safe_date}-{recipient_key}@research-inbox.local>"


def load_delivery_state(path: Path) -> dict[str, Any]:
    state = load_json(
        path,
        {
            "schema_version": 1,
            "sent_digests": {},
        },
    )
    if not isinstance(state, dict):
        raise RuntimeError("Email delivery state must be a JSON object")
    state.setdefault("schema_version", 1)
    sent = state.setdefault("sent_digests", {})
    if not isinstance(sent, dict):
        raise RuntimeError("sent_digests must be an object")
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write(
        path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def resolve_digest_date(value: str | None) -> str:
    if value:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    return datetime.now(timezone.utc).astimezone(LOCAL_TIMEZONE).date().isoformat()


def empty_digest_body(state_root: Path, digest_date: str) -> str:
    selection = load_json(state_root / "state" / "selection_manifest.json", {})
    if not isinstance(selection, dict):
        selection = {}
    selected = int(selection.get("summary_slot_count") or 0)
    if selected > 0:
        raise RuntimeError(
            "Selected summaries exist, but the completed digest artifact is missing"
        )

    unified = load_json(state_root / "state" / "unified_registry_manifest.json", {})
    openalex = load_json(state_root / "state" / "openalex_discovery_manifest.json", {})
    routing = load_json(state_root / "state" / "routing_manifest.json", {})
    unified_count = int((unified or {}).get("unified_candidate_count") or 0)
    merged_count = int((unified or {}).get("merged_group_count") or 0)
    accepted_openalex = int((openalex or {}).get("accepted_count") or 0)
    eligible = int(selection.get("eligible_candidate_count") or 0)
    route_counts = (routing or {}).get("route_counts") or {}

    lines = [
        f"# 每日研究汇总 {digest_date}",
        "",
        "今天没有新的论文进入自动摘要名额。",
        "",
        "## 自动流程状态",
        "",
        "- Google Scholar 与 OpenAlex：已完成统一发现与处理",
        f"- 统一候选库论文数：`{unified_count}`",
        f"- 本次跨来源合并组数：`{merged_count}`",
        f"- OpenAlex 本次新接收：`{accepted_openalex}`",
        f"- 尚未完成且达到摘要阈值的候选：`{eligible}`",
        f"- metadata enrichment 队列：`{int(route_counts.get('metadata_enrichment_queue') or 0)}`",
        f"- manual review 队列：`{int(route_counts.get('manual_review_queue') or 0)}`",
        "",
        "系统没有调用 DeepSeek，因此本次没有产生模型 token 费用。",
    ]
    return "\n".join(lines)


def resolve_digest_content(
    *,
    state_root: Path,
    digest_date: str,
    send_empty: bool,
) -> tuple[str | None, int, str]:
    digest_json_path = state_root / "data" / "digests" / f"{digest_date}.generated.json"
    digest_markdown_path = state_root / "data" / "digests" / f"{digest_date}.generated.md"
    if not digest_json_path.exists() or not digest_markdown_path.exists():
        if not send_empty:
            return None, 0, "skipped_no_completed_digest"
        return empty_digest_body(state_root, digest_date), 0, "empty_daily_digest"

    digest = load_json(digest_json_path, {})
    if not isinstance(digest, dict):
        raise RuntimeError("Generated digest JSON must be an object")
    summary_count = int(digest.get("summary_count") or 0)
    if digest.get("status") != "completed_automatic":
        return None, summary_count, "skipped_digest_not_finalized"
    if summary_count <= 0 and not send_empty:
        return None, summary_count, "skipped_empty_digest"
    return digest_markdown_path.read_text(encoding="utf-8").rstrip(), summary_count, "completed_digest"


def digest_candidate_ids(state_root: Path, digest_date: str) -> list[str]:
    digest_path = state_root / "data" / "digests" / f"{digest_date}.generated.json"
    digest = load_json(digest_path, {})
    if not isinstance(digest, dict):
        return []
    values = digest.get("summaries") or []
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        candidate_id = str(value.get("candidate_id") or "").strip()
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            output.append(candidate_id)
    return output


def archive_paths(state_root: Path, digest_date: str) -> tuple[Path, Path]:
    year = digest_date[:4]
    root = state_root / "data" / "digest_archive" / year
    return root / f"{digest_date}.email.md", root / f"{digest_date}.delivery.json"


def archive_delivered_digest(
    *,
    state_root: Path,
    digest_date: str,
    body: str,
    subject: str,
    summary_count: int,
    content_status: str,
    candidate_ids: list[str],
    delivery_status: str,
    sent_at: str,
    recipient_sha256: str,
    message_id_header: str,
    gmail_message_id_sha256: str | None,
) -> dict[str, Any]:
    if delivery_status not in {"sent", "already_sent"}:
        raise RuntimeError("Only confirmed sent digests may be archived")

    body_path, metadata_path = archive_paths(state_root, digest_date)
    body_sha256 = sha256_text(body)
    source_path = state_root / "data" / "digests" / f"{digest_date}.generated.md"
    source_file = (
        f"data/digests/{digest_date}.generated.md"
        if content_status == "completed_digest" and source_path.exists()
        else None
    )
    source_sha256 = sha256_text(source_path.read_text(encoding="utf-8")) if source_file else None

    if body_path.exists():
        existing_body = body_path.read_text(encoding="utf-8")
        if sha256_text(existing_body) != body_sha256:
            raise RuntimeError(f"Archived digest body changed for {digest_date}")
        existing_metadata = load_json(metadata_path, {})
        if not isinstance(existing_metadata, dict):
            raise RuntimeError(f"Archived delivery metadata is invalid for {digest_date}")
        if existing_metadata.get("body_sha256") != body_sha256:
            raise RuntimeError(f"Archived digest metadata hash changed for {digest_date}")
        return {
            "status": "already_archived",
            "body_file": str(body_path.relative_to(state_root)),
            "metadata_file": str(metadata_path.relative_to(state_root)),
            "body_sha256": body_sha256,
        }
    if metadata_path.exists():
        raise RuntimeError(f"Archived delivery metadata exists without its body for {digest_date}")

    archived_at = utc_now()
    body_file = str(body_path.relative_to(state_root))
    metadata_file = str(metadata_path.relative_to(state_root))
    metadata = {
        "schema_version": 1,
        "digest_date": digest_date,
        "delivery_status": delivery_status,
        "sent_at": sent_at,
        "archived_at": archived_at,
        "subject": subject,
        "summary_count": summary_count,
        "candidate_ids": candidate_ids,
        "content_status": content_status,
        "body_sha256": body_sha256,
        "archive_body_file": body_file,
        "source_digest_file": source_file,
        "source_digest_sha256": source_sha256,
        "knowledge_index_file": "data/knowledge_base/index.md",
        "recipient_sha256": recipient_sha256,
        "message_id_header": message_id_header,
        "gmail_message_id_sha256": gmail_message_id_sha256,
        "full_text_persisted": False,
    }
    atomic_write(body_path, body)
    atomic_write(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "status": "archived",
        "body_file": body_file,
        "metadata_file": metadata_file,
        "body_sha256": body_sha256,
    }


def send_daily_digest(
    *,
    config_path: Path,
    state_root: Path,
    digest_date: str,
    service: Any | None = None,
) -> dict[str, Any]:
    policy, recipient, send_empty, state_value = load_delivery_config(config_path)
    state_path = state_root / state_value
    state = load_delivery_state(state_path)

    body, summary_count, content_status = resolve_digest_content(
        state_root=state_root,
        digest_date=digest_date,
        send_empty=send_empty,
    )
    if body is None:
        result = {
            "status": content_status,
            "digest_date": digest_date,
            "sent": False,
        }
        print(stable_json(result), flush=True)
        return result

    body = body.rstrip() + (
        "\n\n---\n"
        "长期知识库索引：automation-state/data/knowledge_base/index.md\n"
    )
    digest_sha256 = sha256_text(body)
    recipient_sha256 = sha256_text(recipient)
    message_id = deterministic_message_id(digest_date, recipient)
    subject = f"每日研究汇总 {digest_date}（{summary_count} 篇）"
    candidate_ids = digest_candidate_ids(state_root, digest_date)

    sent_digests = state["sent_digests"]
    existing = sent_digests.get(digest_date)
    if isinstance(existing, dict) and existing.get("status") in {"sent", "already_sent"}:
        archive_status = "skipped_digest_hash_mismatch"
        archive: dict[str, Any] | None = None
        if existing.get("digest_sha256") == digest_sha256:
            archive = archive_delivered_digest(
                state_root=state_root,
                digest_date=digest_date,
                body=body,
                subject=subject,
                summary_count=summary_count,
                content_status=content_status,
                candidate_ids=candidate_ids,
                delivery_status=str(existing["status"]),
                sent_at=str(existing.get("sent_at") or ""),
                recipient_sha256=str(existing.get("recipient_sha256") or recipient_sha256),
                message_id_header=str(existing.get("message_id_header") or message_id),
                gmail_message_id_sha256=(
                    str(existing.get("gmail_message_id_sha256"))
                    if existing.get("gmail_message_id_sha256")
                    else None
                ),
            )
            archive_status = str(archive["status"])
        result = {
            "status": "skipped_already_recorded",
            "digest_date": digest_date,
            "sent": False,
            "digest_sha256": digest_sha256,
            "recorded_digest_sha256": existing.get("digest_sha256"),
            "archive_status": archive_status,
        }
        if archive:
            result["archive_body_file"] = archive["body_file"]
            result["archive_metadata_file"] = archive["metadata_file"]
        print(stable_json(result), flush=True)
        return result

    try:
        gmail = service or build_gmail_service(credentials_from_environment())
        delivery = send_digest_idempotent(
            service=gmail,
            recipient=recipient,
            subject=subject,
            text_body=body,
            policy=policy,
            message_id=message_id,
        )
        status = str(delivery.get("status") or "sent")
        gmail_id = str(delivery.get("message_id") or "")
        sent_at = utc_now()
        gmail_message_id_sha256 = sha256_text(gmail_id) if gmail_id else None
        archive = archive_delivered_digest(
            state_root=state_root,
            digest_date=digest_date,
            body=body,
            subject=subject,
            summary_count=summary_count,
            content_status=content_status,
            candidate_ids=candidate_ids,
            delivery_status=status,
            sent_at=sent_at,
            recipient_sha256=recipient_sha256,
            message_id_header=message_id,
            gmail_message_id_sha256=gmail_message_id_sha256,
        )
    except Exception as error:
        state["last_failure"] = {
            "digest_date": digest_date,
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }
        write_state(state_path, state)
        raise

    sent_digests[digest_date] = {
        "status": status,
        "sent_at": sent_at,
        "digest_sha256": digest_sha256,
        "summary_count": summary_count,
        "content_status": content_status,
        "recipient_sha256": recipient_sha256,
        "message_id_header": message_id,
        "gmail_message_id_sha256": gmail_message_id_sha256,
        "archive_status": archive["status"],
        "archive_body_file": archive["body_file"],
        "archive_metadata_file": archive["metadata_file"],
    }
    state.pop("last_failure", None)
    state["last_successful_delivery_at"] = sent_digests[digest_date]["sent_at"]
    write_state(state_path, state)

    result = {
        "status": status,
        "digest_date": digest_date,
        "summary_count": summary_count,
        "content_status": content_status,
        "sent": status == "sent",
        "idempotent_recovery": status == "already_sent",
        "digest_sha256": digest_sha256,
        "archive_status": archive["status"],
        "archive_body_file": archive["body_file"],
        "archive_metadata_file": archive["metadata_file"],
    }
    print(stable_json(result), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send and archive one idempotent daily Research Inbox digest"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--digest-date")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    send_daily_digest(
        config_path=args.config,
        state_root=args.state_root,
        digest_date=resolve_digest_date(args.digest_date),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

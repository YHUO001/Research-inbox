from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.summarize.evidence_guard import enforce_onn_architecture
from scripts.summarize.generate_summaries import (
    validate_chinese_summary,
    validate_method_depth,
)
from scripts.summarize.prepare_digest import (
    atomic_write,
    load_json,
    load_jsonl,
    stable_json,
    validate_record,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_sha256(value: dict[str, Any]) -> str:
    return sha256_text(stable_json(value))


def unique_by_candidate(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id:
            raise RuntimeError(f"{label} contains an empty candidate_id")
        if candidate_id in result:
            raise RuntimeError(f"{label} contains duplicate candidate_id: {candidate_id}")
        result[candidate_id] = record
    return result


def automatic_markdown(content: str, *, completed_at: str, count: int) -> str:
    marker = "\n## 自动处理状态\n"
    if marker in content:
        content = content.split(marker, 1)[0].rstrip()
    return (
        content.rstrip()
        + marker
        + "\n- 本地结构与证据校验：`通过`\n"
        + f"- 自动完成论文数：`{count}`\n"
        + f"- 完成时间：`{completed_at}`\n"
        + "- 人工评审：`不需要`\n"
        + "- 邮件发送：`false`\n"
    )


def validate_automatic_config(config: dict[str, Any]) -> None:
    execution = config.get("execution") or {}
    automation = config.get("automation") or {}
    review = config.get("review") or {}
    if automation.get("enabled") is not True:
        raise RuntimeError("Automatic summary orchestration must be enabled")
    if automation.get("mode") != "automatic_after_discovery":
        raise RuntimeError("Automatic finalization requires automatic_after_discovery")
    if automation.get("update_summary_history_after_validation") is not True:
        raise RuntimeError("Automatic finalization must update history after validation")
    if automation.get("all_or_nothing_batch") is not True:
        raise RuntimeError("Automatic finalization requires all-or-nothing batches")
    if automation.get("review_required") or review.get("required"):
        raise RuntimeError("Human review must be disabled during finalization")
    if automation.get("email_enabled") or execution.get("email_enabled"):
        raise RuntimeError("Email delivery must remain disabled during finalization")


def finalize_automatic(
    *,
    generation_manifest_path: Path,
    history_path: Path,
    summary_schema_path: Path,
    config_path: Path,
    completed_at: str | None = None,
) -> dict[str, Any]:
    completed_at = completed_at or utc_now()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    validate_automatic_config(config)

    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict):
        raise RuntimeError("Summary generation manifest must be a JSON object")
    if generation.get("status") == "completed_automatic" and generation.get("summary_history_updated") is True:
        return generation
    if generation.get("status") != "completed":
        raise RuntimeError("Only a completed generation batch can be finalized")

    request_count = int(generation.get("request_count") or 0)
    summary_count = int(generation.get("summary_count") or 0)
    failure_count = int(generation.get("failure_count") or 0)
    if request_count <= 0 or summary_count != request_count or failure_count != 0:
        raise RuntimeError("Automatic finalization requires a complete all-or-nothing batch")
    if generation.get("numeric_grounding_scope") != "title_abstract_and_open_full_text_loose":
        raise RuntimeError("Automatic finalization requires the approved loose full-evidence grounding mode")

    request_path = Path(str(generation.get("request_file") or ""))
    summary_path = Path(str(generation.get("summary_file") or ""))
    digest_json_path = Path(str(generation.get("digest_json_file") or ""))
    digest_markdown_path = Path(str(generation.get("digest_markdown_file") or ""))
    for label, path in (
        ("request", request_path),
        ("summary", summary_path),
        ("digest JSON", digest_json_path),
        ("digest Markdown", digest_markdown_path),
    ):
        if not path.exists():
            raise RuntimeError(f"Missing {label} artifact: {path}")

    requests = load_jsonl(request_path)
    summaries = load_jsonl(summary_path)
    request_by_id = unique_by_candidate(requests, "Summary requests")
    summary_by_id = unique_by_candidate(summaries, "Summaries")
    if set(request_by_id) != set(summary_by_id) or len(requests) != request_count:
        raise RuntimeError("Request and summary candidate sets do not match")

    schema = load_json(summary_schema_path, {})
    output = config.get("output") or {}
    repaired_summaries: list[dict[str, Any]] = []
    architecture_repairs = 0
    architecture_evidence: dict[str, Any] = {}
    for request in requests:
        candidate_id = str(request["candidate_id"])
        summary = summary_by_id[candidate_id]
        summary, evidence, changed, previous = enforce_onn_architecture(
            summary,
            abstract=(request.get("source") or {}).get("abstract"),
        )
        architecture_repairs += int(changed)
        architecture_evidence[candidate_id] = {
            **evidence.as_dict(),
            "changed": changed,
            "previous_type": previous,
        }
        validate_record(summary, schema, f"summary {candidate_id}")
        language_errors = validate_chinese_summary(
            summary,
            minimum_han=int(output.get("minimum_han_characters") or 120),
        )
        method_errors = validate_method_depth(
            summary,
            minimum_paragraphs=int(output.get("method_implementation_min_paragraphs") or 2),
            maximum_paragraphs=int(output.get("method_implementation_max_paragraphs") or 6),
        )
        if language_errors or method_errors:
            raise RuntimeError(
                f"Final validation failed for {candidate_id}: "
                + "; ".join([*language_errors, *method_errors])
            )
        repaired_summaries.append(summary)

    digest = load_json(digest_json_path, {})
    if not isinstance(digest, dict):
        raise RuntimeError("Generated digest must be a JSON object")
    digest_ids = {
        str(item.get("candidate_id") or "")
        for item in digest.get("summaries") or []
        if isinstance(item, dict)
    }
    if digest_ids != set(request_by_id):
        raise RuntimeError("Digest and request candidate sets do not match")

    history = load_json(
        history_path,
        {"schema_version": 1, "completed_candidate_ids": {}, "failed_candidate_ids": {}},
    )
    if not isinstance(history, dict):
        raise RuntimeError("Summary history must be a JSON object")
    completed = history.setdefault("completed_candidate_ids", {})
    if not isinstance(completed, dict):
        raise RuntimeError("completed_candidate_ids must be an object")
    history.setdefault("failed_candidate_ids", {})

    newly_completed = 0
    for request, summary in zip(requests, repaired_summaries, strict=True):
        candidate_id = str(request["candidate_id"])
        summary_hash = record_sha256(summary)
        existing = completed.get(candidate_id)
        if isinstance(existing, dict):
            existing_hash = str(existing.get("summary_record_sha256") or "")
            if existing_hash and existing_hash != summary_hash:
                raise RuntimeError(
                    f"Completed candidate {candidate_id} has a different summary hash"
                )
        else:
            newly_completed += 1
        completed[candidate_id] = {
            "completed_at": completed_at,
            "digest_date": str(generation.get("digest_date") or ""),
            "model": str(generation.get("model") or ""),
            "provider": str(generation.get("provider") or ""),
            "output_language": str(generation.get("output_language") or "zh-CN"),
            "request_id": request.get("request_id"),
            "summary_record_sha256": summary_hash,
            "completion_mode": "automatic_after_local_validation",
            "review_required": False,
        }

    digest["status"] = "completed_automatic"
    digest["summaries"] = repaired_summaries
    digest.setdefault("safety", {})["summary_history_updated"] = True
    digest["safety"]["review_required"] = False
    digest["completion"] = {
        "mode": "automatic_after_local_validation",
        "completed_at": completed_at,
        "architecture_repairs": architecture_repairs,
    }

    generation.update(
        {
            "status": "completed_automatic",
            "summary_history_updated": True,
            "review_required": False,
            "automatic_history_pending": False,
            "completed_at": completed_at,
            "completed_candidate_ids": sorted(request_by_id),
            "newly_completed_candidate_count": newly_completed,
            "architecture_repairs": architecture_repairs,
            "architecture_evidence": architecture_evidence,
            "completion_mode": "automatic_after_local_validation",
        }
    )

    atomic_write(
        summary_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in repaired_summaries
        ),
    )
    atomic_write(
        digest_json_path,
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        digest_markdown_path,
        automatic_markdown(
            digest_markdown_path.read_text(encoding="utf-8"),
            completed_at=completed_at,
            count=len(repaired_summaries),
        ),
    )
    atomic_write(
        history_path,
        json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        generation_manifest_path,
        json.dumps(generation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(stable_json(generation), flush=True)
    return generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize a fully validated summary batch without human review"
    )
    parser.add_argument("--generation-manifest-path", type=Path, required=True)
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--summary-schema", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    finalize_automatic(
        generation_manifest_path=args.generation_manifest_path,
        history_path=args.history_path,
        summary_schema_path=args.summary_schema,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from scripts.enrich.selected_author_metadata import enrich_selected_author_context
from scripts.summarize.prepare_digest import (
    atomic_write,
    file_digest,
    load_json,
    load_jsonl,
    prepare_dry_run,
    stable_json,
)


def completed_candidate_ids(history: dict[str, Any]) -> set[str]:
    value = history.get("completed_candidate_ids") or {}
    if isinstance(value, dict):
        return {str(candidate_id) for candidate_id in value}
    if isinstance(value, list):
        return {str(candidate_id) for candidate_id in value}
    raise ValueError("completed_candidate_ids must be an object or list")


def validate_automatic_config(config: dict[str, Any]) -> None:
    execution = config.get("execution") or {}
    automation = config.get("automation") or {}
    review = config.get("review") or {}
    knowledge = config.get("knowledge_base") or {}
    delivery = config.get("delivery") or {}
    if automation.get("enabled") is not True:
        raise RuntimeError("Automatic summary orchestration must be enabled")
    if automation.get("mode") != "automatic_daily_batch":
        raise RuntimeError("Automatic preparation requires automatic_daily_batch")
    if automation.get("filter_completed_before_provider") is not True:
        raise RuntimeError("Completed candidates must be filtered before provider calls")
    if automation.get("update_summary_history_after_validation") is not True:
        raise RuntimeError("Automatic completion must update history after validation")
    if automation.get("all_or_nothing_batch") is not True:
        raise RuntimeError("Automatic completion requires all-or-nothing batches")
    if automation.get("review_required") or review.get("required"):
        raise RuntimeError("Human review must be disabled for automatic preparation")
    if execution.get("email_enabled"):
        raise RuntimeError("The generator must not send email directly")
    if automation.get("delivery_mode") != "separate_daily_digest_workflow":
        raise RuntimeError("Daily delivery must remain separate from generation")
    if knowledge.get("enabled") is not True or knowledge.get("persist_full_text"):
        raise RuntimeError("The long-term index must be enabled without persisted full text")
    if delivery.get("daily_digest_enabled") is not True:
        raise RuntimeError("Daily digest delivery must be enabled")


def attach_selected_author_contexts(
    request_path: Path,
    *,
    api_key: str | None,
) -> dict[str, int]:
    """Attach deterministic author/team context to already-selected requests only.

    The prompt has already been built before this step, so these static metadata
    fields are available to the renderer without increasing LLM prompt tokens.
    """
    requests = load_jsonl(request_path)
    enriched_count = 0
    fallback_count = 0
    for request in requests:
        source = request.get("source")
        if not isinstance(source, dict):
            continue
        context = enrich_selected_author_context(source, api_key=api_key)
        source["author_context"] = context
        if context.get("lookup_status") == "enriched":
            enriched_count += 1
        else:
            fallback_count += 1
    atomic_write(
        request_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in requests
        ),
    )
    return {
        "author_metadata_enriched_count": enriched_count,
        "author_metadata_fallback_count": fallback_count,
    }


def prepare_automatic(
    *,
    queue_path: Path,
    history_path: Path,
    selection_manifest_path: Path,
    config_path: Path,
    request_schema_path: Path,
    summary_schema_path: Path,
    output_root: Path,
    manifest_path: Path,
    filtered_queue_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    validate_automatic_config(config)

    queue = load_jsonl(queue_path)
    history = load_json(
        history_path,
        {"schema_version": 1, "completed_candidate_ids": {}, "failed_candidate_ids": {}},
    )
    if not isinstance(history, dict):
        raise ValueError("Summary history must be a JSON object")
    completed = completed_candidate_ids(history)
    filtered = [
        item
        for item in queue
        if str(item.get("candidate_id") or "") not in completed
    ]
    filtered_out = len(queue) - len(filtered)
    atomic_write(
        filtered_queue_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in filtered
        ),
    )

    manifest = prepare_dry_run(
        queue_path=filtered_queue_path,
        selection_manifest_path=selection_manifest_path,
        config_path=config_path,
        request_schema_path=request_schema_path,
        summary_schema_path=summary_schema_path,
        output_root=output_root,
        state_manifest_path=manifest_path,
    )

    author_counts = {
        "author_metadata_enriched_count": 0,
        "author_metadata_fallback_count": 0,
    }
    request_path_value = str(manifest.get("request_file") or "")
    if int(manifest.get("request_count") or 0) > 0 and request_path_value:
        request_path = Path(request_path_value)
        author_counts = attach_selected_author_contexts(
            request_path,
            api_key=os.environ.get("OPENALEX_API_KEY"),
        )
        manifest["request_sha256"] = file_digest(request_path)

    manifest.update(
        {
            "status": "no_candidates" if int(manifest.get("request_count") or 0) == 0 else "automatic_requests_prepared",
            "execution_mode": "automatic_daily_batch",
            "llm_enabled": True,
            "review_required": False,
            "summary_history_updated": False,
            "source_queue_candidate_count": len(queue),
            "completed_candidate_filtered_count": filtered_out,
            "automatic_queue_candidate_count": len(filtered),
            "history_path": str(history_path),
            "knowledge_base_pending": int(manifest.get("request_count") or 0) > 0,
            "daily_digest_pending": int(manifest.get("request_count") or 0) > 0,
            "author_metadata_mode": "selected_openalex_work_metadata_with_safe_fallback",
            **author_counts,
        }
    )
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(stable_json(manifest), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare automatic summary requests after filtering completed candidates"
    )
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--selection-manifest-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request-schema", type=Path, required=True)
    parser.add_argument("--summary-schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--filtered-queue-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prepare_automatic(
        queue_path=args.queue_path,
        history_path=args.history_path,
        selection_manifest_path=args.selection_manifest_path,
        config_path=args.config,
        request_schema_path=args.request_schema,
        summary_schema_path=args.summary_schema,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
        filtered_queue_path=args.filtered_queue_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

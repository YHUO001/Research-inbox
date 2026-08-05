from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.summarize.finalize_automatic import finalize_automatic
from scripts.summarize.prepare_digest import atomic_write, load_json, load_jsonl, stable_json


_ALLOWED_MIGRATION_MODES = {
    "migrated_from_user_reviewed_batch",
    "automatic_daily_batch_after_local_validation",
}


def unique_candidate_ids(records: list[dict[str, Any]], label: str) -> set[str]:
    values: set[str] = set()
    for record in records:
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id:
            raise RuntimeError(f"{label} contains an empty candidate_id")
        if candidate_id in values:
            raise RuntimeError(f"{label} contains duplicate candidate_id: {candidate_id}")
        values.add(candidate_id)
    return values


def bootstrap_approved_batch(
    *,
    generation_manifest_path: Path,
    state_root: Path,
    history_path: Path,
    summary_schema_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict):
        raise RuntimeError("Summary generation manifest must be a JSON object")

    if generation.get("status") == "completed_automatic":
        result = {"status": "skipped_already_automatic", "bootstrapped": False}
        print(stable_json(result), flush=True)
        return result
    if generation.get("status") != "completed":
        result = {"status": "skipped_no_legacy_completed_batch", "bootstrapped": False}
        print(stable_json(result), flush=True)
        return result

    request_count = int(generation.get("request_count") or 0)
    summary_count = int(generation.get("summary_count") or 0)
    failure_count = int(generation.get("failure_count") or 0)
    digest_date = str(generation.get("digest_date") or "")
    if not digest_date or request_count <= 0 or summary_count != request_count or failure_count:
        raise RuntimeError("Legacy batch is not a complete all-or-nothing generation")

    request_path_value = str(generation.get("request_file") or "")
    request_path = Path(request_path_value) if request_path_value else Path()
    if not request_path_value or not request_path.exists():
        request_path = state_root / "data" / "summary_requests" / f"{digest_date}.jsonl"
    if not request_path.exists():
        raise RuntimeError(f"Legacy summary request file is missing: {request_path}")

    requests = load_jsonl(request_path)
    request_ids = unique_candidate_ids(requests, "Legacy summary requests")
    if len(request_ids) != request_count:
        raise RuntimeError("Legacy request count does not match the generation manifest")

    history = load_json(history_path, {})
    if not isinstance(history, dict):
        raise RuntimeError("Summary history must be a JSON object")
    completed = history.get("completed_candidate_ids") or {}
    if not isinstance(completed, dict):
        raise RuntimeError("completed_candidate_ids must be an object")

    for candidate_id in request_ids:
        entry = completed.get(candidate_id)
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Legacy candidate is not recorded as user-approved: {candidate_id}"
            )
        if entry.get("review_required") is not False:
            raise RuntimeError(
                f"Legacy candidate still requires review: {candidate_id}"
            )
        if str(entry.get("completion_mode") or "") not in _ALLOWED_MIGRATION_MODES:
            raise RuntimeError(
                f"Legacy candidate has an unsupported completion mode: {candidate_id}"
            )

    generation["request_file"] = str(request_path)
    generation["bootstrap_source"] = "existing_user_approved_batch"
    generation["bootstrap_candidate_ids"] = sorted(request_ids)
    atomic_write(
        generation_manifest_path,
        json.dumps(generation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    finalized = finalize_automatic(
        generation_manifest_path=generation_manifest_path,
        history_path=history_path,
        summary_schema_path=summary_schema_path,
        config_path=config_path,
    )
    finalized["bootstrap_source"] = "existing_user_approved_batch"
    finalized["bootstrapped"] = True
    finalized["new_provider_calls"] = 0
    atomic_write(
        generation_manifest_path,
        json.dumps(finalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    result = {
        "status": "completed",
        "bootstrapped": True,
        "digest_date": digest_date,
        "candidate_count": len(request_ids),
        "new_provider_calls": 0,
    }
    print(stable_json(result), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap an existing user-approved summary batch into automation"
    )
    parser.add_argument("--generation-manifest-path", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--summary-schema", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bootstrap_approved_batch(
        generation_manifest_path=args.generation_manifest_path,
        state_root=args.state_root,
        history_path=args.history_path,
        summary_schema_path=args.summary_schema,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

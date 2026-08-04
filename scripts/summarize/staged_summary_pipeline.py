from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from scripts.summarize import generate_summaries as summary_core
from scripts.summarize.deepseek_provider import DeepSeekClient, DeepSeekResponse
from scripts.summarize.fulltext_methods import MethodContext, candidate_urls, collect_method_context
from scripts.summarize.generate_summaries_production import (
    ProductionNormalizingClient,
    TransportRepairDiagnostics,
    diagnostic_payload,
    expected_example,
    generation_manifest_copy,
    request_abstracts,
)
from scripts.summarize.prepare_digest import (
    atomic_write,
    load_json,
    stable_json,
    validate_numeric_grounding,
)


def log_event(stage: str, event: str, **details: Any) -> None:
    payload = {
        "stage": stage,
        "event": event,
        **details,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def validate_full_text_safety(config: dict[str, Any]) -> None:
    execution = config.get("execution") or {}
    full_text = config.get("full_text") or {}
    if execution.get("use_full_text") and not full_text.get("open_access_only"):
        raise RuntimeError("Full-text method context must remain open-access-only")
    if full_text.get("persist_extracted_text"):
        raise RuntimeError("Extracted full text must not be persisted")
    if full_text.get("numeric_grounding_scope") != "title_and_abstract_only":
        raise RuntimeError("Numeric grounding must remain limited to title and abstract")


def shared_numeric_grounding(
    summary: dict[str, Any], *, title: str, abstract: str | None
) -> list[str]:
    narrative = {
        key: summary.get(key)
        for key in summary_core.NARRATIVE_FIELDS
    }
    return validate_numeric_grounding(
        narrative,
        title=title,
        abstract=abstract,
    )


def prepare_stage(
    *,
    dry_run_manifest_path: Path,
    config_path: Path,
    prepared_root: Path,
    method_context_loader: Callable[..., MethodContext] = collect_method_context,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    validate_full_text_safety(config)
    prepared_root.mkdir(parents=True, exist_ok=True)

    per_candidate_seconds: dict[str, float] = {}

    def timed_loader(source: dict[str, Any], *, config: dict[str, Any]) -> MethodContext:
        candidate_id = str(source.get("candidate_id") or "unknown")
        urls = candidate_urls(source, int(config.get("candidate_url_limit") or 3))
        log_event(
            "full_text",
            "candidate_start",
            candidate_id=candidate_id,
            candidate_urls=urls,
        )
        started = time.monotonic()
        try:
            context = method_context_loader(source, config=config)
        except Exception as error:
            elapsed = round(time.monotonic() - started, 3)
            per_candidate_seconds[candidate_id] = elapsed
            log_event(
                "full_text",
                "candidate_error",
                candidate_id=candidate_id,
                elapsed_seconds=elapsed,
                error=f"{type(error).__name__}: {str(error)[:300]}",
            )
            raise
        elapsed = round(time.monotonic() - started, 3)
        per_candidate_seconds[candidate_id] = elapsed
        log_event(
            "full_text",
            "candidate_complete",
            candidate_id=candidate_id,
            elapsed_seconds=elapsed,
            status=context.status,
            source_url=context.source_url,
            media_type=context.media_type,
            character_count=context.character_count,
            section_headings=context.section_headings,
        )
        return context

    started = time.monotonic()
    log_event("full_text", "stage_start")
    generation_manifest, alias_count, contexts = generation_manifest_copy(
        dry_run_manifest_path,
        prepared_root,
        config=config,
        method_context_loader=timed_loader,
    )
    elapsed = round(time.monotonic() - started, 3)
    for candidate_id, record in contexts.items():
        record["elapsed_seconds"] = per_candidate_seconds.get(candidate_id, 0.0)

    audit_path = prepared_root / "fulltext_preparation.json"
    audit = {
        "status": "prepared",
        "prepared_generation_manifest_path": str(generation_manifest),
        "preparation_audit_path": str(audit_path),
        "elapsed_seconds": elapsed,
        "tops_alias_expansions": alias_count,
        "full_text_method_contexts": contexts,
        "full_text_used": any(
            item.get("status") == "used" for item in contexts.values()
        ),
        "full_text_persisted": False,
        "numeric_grounding_scope": "title_and_abstract_only",
    }
    atomic_write(
        audit_path,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    log_event(
        "full_text",
        "stage_complete",
        elapsed_seconds=elapsed,
        candidate_count=len(contexts),
        full_text_used=audit["full_text_used"],
        prepared_generation_manifest_path=str(generation_manifest),
    )
    return audit


class TimedDeepSeekClient:
    def __init__(self, base_client: Any) -> None:
        self.base_client = base_client
        self.attempts: dict[str, int] = {}
        self.call_records: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> DeepSeekResponse:
        expected = expected_example(str(kwargs.get("system_prompt") or ""))
        candidate_id = str(expected.get("candidate_id") or "unknown")
        attempt = self.attempts.get(candidate_id, 0) + 1
        self.attempts[candidate_id] = attempt
        user_prompt = str(kwargs.get("user_prompt") or "")
        retry_marker = "上一份 JSON 未通过本地校验："
        retry_reason = None
        if retry_marker in user_prompt:
            retry_reason = user_prompt.rsplit(retry_marker, 1)[1].split("。请修正后", 1)[0][:300]
        log_event(
            "model",
            "request_start",
            candidate_id=candidate_id,
            attempt=attempt,
            validation_retry=retry_reason is not None,
            retry_reason=retry_reason,
            model=str(kwargs.get("model") or ""),
            max_tokens=int(kwargs.get("max_tokens") or 0),
        )
        started = time.monotonic()
        try:
            response = self.base_client.complete_json(**kwargs)
        except Exception as error:
            elapsed = round(time.monotonic() - started, 3)
            record = {
                "candidate_id": candidate_id,
                "attempt": attempt,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": f"{type(error).__name__}: {str(error)[:300]}",
            }
            self.call_records.append(record)
            log_event("model", "request_error", **record)
            raise
        elapsed = round(time.monotonic() - started, 3)
        record = {
            "candidate_id": candidate_id,
            "attempt": attempt,
            "status": "response_received",
            "elapsed_seconds": elapsed,
            "usage": response.usage,
        }
        self.call_records.append(record)
        log_event("model", "response_received", **record)
        return response


def generate_stage(
    *,
    dry_run_manifest_path: Path,
    summary_schema_path: Path,
    config_path: Path,
    prepared_root: Path,
    output_root: Path,
    manifest_path: Path,
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    validate_full_text_safety(config)
    provider = config["provider"]

    preparation_audit_path = prepared_root / "fulltext_preparation.json"
    audit = load_json(preparation_audit_path, {})
    if not isinstance(audit, dict) or audit.get("status") != "prepared":
        raise RuntimeError("A completed full-text preparation stage is required")
    prepared_manifest_path = Path(
        str(audit.get("prepared_generation_manifest_path") or "")
    )
    if not prepared_manifest_path.exists():
        raise RuntimeError("Prepared generation manifest is missing")

    key = api_key or os.environ.get(str(provider["api_key_env"]))
    if not key:
        raise RuntimeError("Missing required DeepSeek API key")
    base_client = client or DeepSeekClient(
        api_key=key,
        base_url=str(provider["base_url"]),
        timeout_seconds=float(provider["timeout_seconds"]),
        max_attempts=int(provider["http_max_attempts"]),
    )
    timed_client = TimedDeepSeekClient(base_client)
    diagnostics = TransportRepairDiagnostics()
    diagnostics.tops_alias_expansions = int(audit.get("tops_alias_expansions") or 0)
    contexts = audit.get("full_text_method_contexts") or {}
    diagnostics.full_text_method_contexts = (
        dict(contexts) if isinstance(contexts, dict) else {}
    )
    wrapped_client = ProductionNormalizingClient(
        base_client=timed_client,
        diagnostics=diagnostics,
        abstracts=request_abstracts(dry_run_manifest_path),
    )

    previous_validator = summary_core.validate_summary_numeric_grounding
    summary_core.validate_summary_numeric_grounding = shared_numeric_grounding
    started = time.monotonic()
    log_event(
        "model",
        "stage_start",
        prepared_generation_manifest_path=str(prepared_manifest_path),
        candidate_count=len(diagnostics.full_text_method_contexts),
    )
    state: dict[str, Any] | None = None
    try:
        state = summary_core.generate(
            dry_run_manifest_path=prepared_manifest_path,
            summary_schema_path=summary_schema_path,
            config_path=config_path,
            output_root=output_root,
            manifest_path=manifest_path,
            api_key=key,
            client=wrapped_client,
        )
        return state
    finally:
        summary_core.validate_summary_numeric_grounding = previous_validator
        elapsed = round(time.monotonic() - started, 3)
        persisted = load_json(manifest_path, {})
        if not isinstance(persisted, dict):
            persisted = {}
        persisted["transport_repairs"] = diagnostic_payload(diagnostics)
        persisted["full_text_method_contexts"] = diagnostics.full_text_method_contexts
        persisted["full_text_used"] = bool(audit.get("full_text_used"))
        persisted["full_text_persisted"] = False
        persisted["numeric_grounding_scope"] = "title_and_abstract_only"
        persisted["stage_timings"] = {
            "full_text_preparation_seconds": float(audit.get("elapsed_seconds") or 0),
            "model_generation_and_validation_seconds": elapsed,
            "model_api_calls": timed_client.call_records,
        }
        atomic_write(
            manifest_path,
            json.dumps(persisted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        log_event(
            "model",
            "stage_complete",
            elapsed_seconds=elapsed,
            status=persisted.get("status"),
            summary_count=persisted.get("summary_count"),
            failure_count=persisted.get("failure_count"),
            api_call_count=len(timed_client.call_records),
        )
        if state is not None:
            state.clear()
            state.update(persisted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-text preparation and DeepSeek generation as separately timed stages"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--dry-run-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    prepare_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/summary_generation.yaml"),
    )
    prepare_parser.add_argument("--prepared-root", type=Path, required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--dry-run-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    generate_parser.add_argument(
        "--summary-schema",
        type=Path,
        default=Path("schemas/paper_summary.schema.json"),
    )
    generate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/summary_generation.yaml"),
    )
    generate_parser.add_argument("--prepared-root", type=Path, required=True)
    generate_parser.add_argument(
        "--output-root", type=Path, default=Path("runtime-state/data")
    )
    generate_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_stage(
            dry_run_manifest_path=args.dry_run_manifest_path,
            config_path=args.config,
            prepared_root=args.prepared_root,
        )
    else:
        result = generate_stage(
            dry_run_manifest_path=args.dry_run_manifest_path,
            summary_schema_path=args.summary_schema,
            config_path=args.config,
            prepared_root=args.prepared_root,
            output_root=args.output_root,
            manifest_path=args.manifest_path,
        )
    print(stable_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

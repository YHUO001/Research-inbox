from __future__ import annotations

import argparse
import json
import multiprocessing
import re
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from scripts.summarize import staged_summary_pipeline as pipeline
from scripts.summarize.abstract_fallback_policy import abstract_fallback_notice
from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_digest import (
    atomic_write,
    file_digest,
    load_json,
    load_jsonl,
    stable_json,
)
from scripts.summarize.springer_openaccess import (
    api_audit_url,
    collect_official_or_open_context,
    normalize_doi,
)


_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z0-9])(?:~|≈|∼|±)?\s*"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_CONFIG_GROUNDING_SCOPE = "title_and_abstract_only"
_AUDIT_GROUNDING_SCOPE = "title_abstract_and_open_full_text_loose"
_ORIGINAL_GENERATION_MANIFEST_COPY = pipeline.generation_manifest_copy


@dataclass(frozen=True)
class RetriedMethodContext(MethodContext):
    attempt_count: int = 1

    def audit_record(self) -> dict[str, Any]:
        record = super().audit_record()
        record["attempt_count"] = max(1, int(self.attempt_count or 1))
        return record


def retried_context(context: MethodContext, attempt_count: int, *, final: bool = False) -> RetriedMethodContext:
    error = context.error
    if final and context.status != "used":
        suffix = f"full-text retrieval failed after {attempt_count} attempts; fell back to abstract"
        error = "; ".join(value for value in (error, suffix) if value)
    return RetriedMethodContext(
        candidate_id=context.candidate_id,
        status=context.status,
        source_url=context.source_url,
        media_type=context.media_type,
        section_headings=list(context.section_headings),
        text=context.text,
        error=error,
        attempt_count=max(1, int(attempt_count or 1)),
    )


def full_text_numeric_aliases(text: str) -> list[str]:
    """Return distinct decimal literals occurring in temporary full text.

    The aliases are written only to the runner's temporary request snapshot.
    They are not included in the model prompt and are never persisted to the
    automation-state branch.
    """

    aliases: list[str] = []
    seen: set[str] = set()
    for match in _NUMERIC_LITERAL.finditer(text or ""):
        value = match.group("number").lstrip("+")
        if value.startswith("."):
            value = "0" + value
        key = value.lower()
        if key not in seen:
            seen.add(key)
            aliases.append(value)
    return aliases


def _replace_numeric_prompt_rules(prompt: str) -> str:
    updated = prompt.replace(
        "公开正文中的方法相关上下文（仅用于定性解释方法；不得据此新增标题或摘要中没有的数字）",
        "公开正文中的方法相关上下文（可用于方法解释及其中明确出现的数值；不得新增全部证据中都没有的数字）",
    )
    updated = updated.replace(
        "所有数字仍必须出现在标题或摘要中；即使正文上下文包含额外数字，也不要把这些数字写入摘要。",
        "数字可以来自标题、摘要或追加的公开正文方法上下文；允许约数、精确写法和轻微四舍五入之间的自然转换，但不得新增全部证据中都没有出现的数字。",
    )
    updated = updated.replace(
        "标题或摘要中的近似、约数和正负范围必须保留其语义，例如使用“约”“近似”或“正负”；不得把约数改写为精确值，也不得把正负范围改写为单点值。",
        "近似词、正负符号和单位排版可以按中文表达调整；不要改变数量级，也不要把一个数改成明显不同的数。",
    )
    return updated


def generation_manifest_copy_with_full_text_numbers(
    dry_run_manifest_path: Path,
    temporary_root: Path,
    *,
    config: dict[str, Any],
    method_context_loader: Callable[..., MethodContext],
) -> tuple[Path, int, dict[str, dict[str, Any]]]:
    """Create an effective temporary batch with retries, fallback, and skips."""

    captured_text: dict[str, str] = {}

    def capturing_loader(source: dict[str, Any], *, config: dict[str, Any]) -> MethodContext:
        context = method_context_loader(source, config=config)
        if context.status == "used" and context.text:
            captured_text[str(context.candidate_id)] = context.text
        return context

    manifest_path, alias_count, contexts = _ORIGINAL_GENERATION_MANIFEST_COPY(
        dry_run_manifest_path,
        temporary_root,
        config=config,
        method_context_loader=capturing_loader,
    )
    manifest = load_json(manifest_path, {})
    request_path = Path(str(manifest.get("request_file") or ""))
    requests = load_jsonl(request_path)
    full_text_alias_count = 0
    effective_requests: list[dict[str, Any]] = []
    skipped_no_abstract: list[str] = []
    abstract_fallbacks: list[str] = []
    full_text_config = config.get("full_text") or {}
    configured_attempts = max(1, int(full_text_config.get("retrieval_attempts") or 3))
    skip_without_abstract = bool(full_text_config.get("skip_when_abstract_missing", True))

    for request in requests:
        candidate_id = str(request.get("candidate_id") or "")
        source = request.get("source")
        if not isinstance(source, dict):
            continue
        record = contexts.setdefault(candidate_id, {"candidate_id": candidate_id})
        status = str(record.get("status") or "not_available")
        attempts = max(1, int(record.get("attempt_count") or configured_attempts))
        abstract = str(source.get("abstract") or "").strip()

        if status != "used" and not abstract and skip_without_abstract:
            record["fallback_decision"] = "skipped_no_abstract"
            record["abstract_available"] = False
            skipped_no_abstract.append(candidate_id)
            continue

        request["prompt"] = _replace_numeric_prompt_rules(str(request.get("prompt") or ""))
        if status != "used":
            notice = abstract_fallback_notice(attempts)
            request["full_text_fallback"] = True
            request["full_text_retrieval_attempts"] = attempts
            request["abstract_fallback_notice"] = notice
            request["prompt"] = (
                str(request.get("prompt") or "")
                + "\n\n证据降级要求："
                + notice
                + " 不得为了满足篇幅要求补写摘要没有提供的实现或实验细节；"
                "可以生成较短但结构完整的摘要级短讯，并明确列出缺失信息。"
            )
            record["fallback_decision"] = "generate_from_abstract"
            record["abstract_available"] = True
            abstract_fallbacks.append(candidate_id)
        else:
            request["full_text_fallback"] = False
            request["full_text_retrieval_attempts"] = attempts
            record["fallback_decision"] = "use_full_text"
            record["abstract_available"] = bool(abstract)

        aliases = full_text_numeric_aliases(captured_text.get(candidate_id, ""))
        if aliases:
            source["abstract"] = (
                str(source.get("abstract") or "")
                + "\nMachine-only open-full-text numeric grounding aliases: "
                + "; ".join(aliases)
                + "."
            )
            full_text_alias_count += len(aliases)
        effective_requests.append(request)

    atomic_write(
        request_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in effective_requests
        ),
    )
    manifest["request_count"] = len(effective_requests)
    manifest["skipped_no_abstract_candidate_ids"] = skipped_no_abstract
    manifest["skipped_no_abstract_count"] = len(skipped_no_abstract)
    manifest["abstract_fallback_candidate_ids"] = abstract_fallbacks
    manifest["abstract_fallback_count"] = len(abstract_fallbacks)
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    persistent_manifest = load_json(dry_run_manifest_path, {})
    persistent_request_path = Path(str(persistent_manifest.get("request_file") or ""))
    persistent_requests = load_jsonl(persistent_request_path)
    retained_ids = {str(item.get("candidate_id") or "") for item in effective_requests}
    retained_persistent = [
        item
        for item in persistent_requests
        if str(item.get("candidate_id") or "") in retained_ids
    ]
    atomic_write(
        persistent_request_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in retained_persistent
        ),
    )
    persistent_manifest["request_count"] = len(retained_persistent)
    persistent_manifest["request_sha256"] = file_digest(persistent_request_path)
    persistent_manifest["skipped_no_abstract_candidate_ids"] = skipped_no_abstract
    persistent_manifest["skipped_no_abstract_count"] = len(skipped_no_abstract)
    persistent_manifest["abstract_fallback_candidate_ids"] = abstract_fallbacks
    persistent_manifest["abstract_fallback_count"] = len(abstract_fallbacks)
    if not retained_persistent:
        persistent_manifest["status"] = "no_eligible_evidence"
        persistent_manifest["automatic_history_pending"] = False
        persistent_manifest["knowledge_base_pending"] = False
        persistent_manifest["daily_digest_pending"] = True
    atomic_write(
        dry_run_manifest_path,
        json.dumps(persistent_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest_path, alias_count + full_text_alias_count, contexts


def validate_full_text_safety(config: dict[str, Any]) -> None:
    execution = config.get("execution") or {}
    full_text = config.get("full_text") or {}
    grounding = config.get("grounding") or {}
    if execution.get("use_full_text") and not full_text.get("open_access_only"):
        raise RuntimeError("Full-text method context must remain open-access-only")
    if full_text.get("persist_extracted_text"):
        raise RuntimeError("Extracted full text must not be persisted")
    if full_text.get("numeric_grounding_scope") != _CONFIG_GROUNDING_SCOPE:
        raise RuntimeError("Core numeric grounding configuration must remain backward compatible")
    if grounding.get("numeric_matching_mode") != "loose_full_evidence":
        raise RuntimeError("The staged workflow must explicitly enable loose full-evidence matching")
    if int(full_text.get("retrieval_attempts") or 0) != 3:
        raise RuntimeError("Production full-text fallback must use exactly three attempts")
    if full_text.get("fallback_to_abstract_after_attempts") is not True:
        raise RuntimeError("Abstract fallback must be enabled")
    if full_text.get("skip_when_abstract_missing") is not True:
        raise RuntimeError("Candidates without full text or abstract must be skipped")


def _loader_worker(
    source: dict[str, Any],
    config: dict[str, Any],
    loader: Callable[..., MethodContext],
    output: Any,
) -> None:
    try:
        output.put(("ok", loader(source, config=config)))
    except BaseException as error:  # The parent converts optional retrieval failures to fallback.
        output.put(("error", type(error).__name__, str(error)[:300]))


def _timeout_source_url(source: dict[str, Any], config: dict[str, Any]) -> str | None:
    doi = normalize_doi(source.get("doi"))
    if not doi:
        return None
    endpoint = str(
        config.get("springer_openaccess_endpoint")
        or "https://api.springernature.com/openaccess/jats"
    ).rstrip("?")
    return api_audit_url(endpoint, doi)


def _bounded_collect_once(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    loader: Callable[..., MethodContext],
) -> MethodContext:
    candidate_id = str(source.get("candidate_id") or source.get("id") or "unknown")
    timeout_seconds = float(config.get("candidate_timeout_seconds") or 30)
    if timeout_seconds <= 0:
        return loader(source, config=config)

    available = multiprocessing.get_all_start_methods()
    if "fork" not in available:
        return loader(source, config=config)

    context = multiprocessing.get_context("fork")
    output = context.Queue(maxsize=1)
    process = context.Process(
        target=_loader_worker,
        args=(source, config, loader, output),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(5)
        output.close()
        return MethodContext(
            candidate_id=candidate_id,
            status="timed_out",
            source_url=_timeout_source_url(source, config),
            media_type="application/xml+jats",
            section_headings=[],
            text="",
            error=f"full-text child process exceeded {timeout_seconds:g} seconds",
        )

    try:
        result = output.get(timeout=1)
    except Empty:
        return MethodContext(
            candidate_id=candidate_id,
            status="not_available",
            source_url=_timeout_source_url(source, config),
            media_type="application/xml+jats",
            section_headings=[],
            text="",
            error="full-text child process exited without a result",
        )
    finally:
        output.close()

    if result[0] == "ok":
        return result[1]
    return MethodContext(
        candidate_id=candidate_id,
        status="not_available",
        source_url=_timeout_source_url(source, config),
        media_type="application/xml+jats",
        section_headings=[],
        text="",
        error=f"full-text child error {result[1]}",
    )


def bounded_collect_method_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    loader: Callable[..., MethodContext] = collect_official_or_open_context,
) -> MethodContext:
    """Retry optional full-text lookup three times, then return an auditable fallback."""

    maximum_attempts = max(1, int(config.get("retrieval_attempts") or 3))
    last: MethodContext | None = None
    for attempt in range(1, maximum_attempts + 1):
        last = _bounded_collect_once(source, config=config, loader=loader)
        if last.status == "used" and last.text:
            return retried_context(last, attempt)
    if last is None:
        last = MethodContext(
            candidate_id=str(source.get("candidate_id") or source.get("id") or "unknown"),
            status="not_available",
            source_url=_timeout_source_url(source, config),
            media_type=None,
            section_headings=[],
            text="",
            error="full-text retrieval did not start",
        )
    return retried_context(last, maximum_attempts, final=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare optional method context through official OA APIs with hard fallback"
    )
    parser.add_argument(
        "--dry-run-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/summary_generation.yaml"),
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pipeline.generation_manifest_copy = generation_manifest_copy_with_full_text_numbers
    pipeline.validate_full_text_safety = validate_full_text_safety
    result = pipeline.prepare_stage(
        dry_run_manifest_path=args.dry_run_manifest_path,
        config_path=args.config,
        prepared_root=args.prepared_root,
        method_context_loader=bounded_collect_method_context,
    )
    result["numeric_grounding_scope"] = _AUDIT_GROUNDING_SCOPE
    prepared_manifest = load_json(
        Path(str(result.get("prepared_generation_manifest_path") or "")),
        {},
    )
    result["prepared_request_count"] = int(prepared_manifest.get("request_count") or 0)
    result["skipped_no_abstract_candidate_ids"] = list(
        prepared_manifest.get("skipped_no_abstract_candidate_ids") or []
    )
    result["skipped_no_abstract_count"] = len(result["skipped_no_abstract_candidate_ids"])
    result["abstract_fallback_candidate_ids"] = list(
        prepared_manifest.get("abstract_fallback_candidate_ids") or []
    )
    result["abstract_fallback_count"] = len(result["abstract_fallback_candidate_ids"])
    audit_path = Path(str(result.get("preparation_audit_path") or ""))
    if audit_path:
        atomic_write(
            audit_path,
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    print(stable_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

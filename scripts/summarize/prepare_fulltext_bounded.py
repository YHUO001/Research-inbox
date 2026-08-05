from __future__ import annotations

import argparse
import json
import multiprocessing
import re
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from scripts.summarize import staged_summary_pipeline as pipeline
from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_digest import atomic_write, load_json, load_jsonl, stable_json
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
    """Add machine-only full-text numeric aliases to the temporary snapshot."""

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

    for request in requests:
        candidate_id = str(request.get("candidate_id") or "")
        request["prompt"] = _replace_numeric_prompt_rules(str(request.get("prompt") or ""))
        aliases = full_text_numeric_aliases(captured_text.get(candidate_id, ""))
        if not aliases:
            continue
        source = request.get("source")
        if not isinstance(source, dict):
            continue
        abstract = str(source.get("abstract") or "")
        source["abstract"] = (
            abstract
            + "\nMachine-only open-full-text numeric grounding aliases: "
            + "; ".join(aliases)
            + "."
        )
        full_text_alias_count += len(aliases)

    atomic_write(
        request_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in requests
        ),
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


def bounded_collect_method_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    loader: Callable[..., MethodContext] = collect_official_or_open_context,
) -> MethodContext:
    """Run one optional full-text lookup in a killable child process."""

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
            error=(
                f"full-text child process exceeded {timeout_seconds:g} seconds; "
                "fell back to abstract"
            ),
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
            error="full-text child process exited without a result; fell back to abstract",
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
        error=f"full-text child error {result[1]}; fell back to abstract",
    )


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

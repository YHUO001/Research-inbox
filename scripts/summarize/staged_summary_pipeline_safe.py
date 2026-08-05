from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from scripts.summarize import staged_summary_pipeline as pipeline
from scripts.summarize.abstract_fallback_policy import (
    ABSTRACT_ONLY_BASIS,
    abstract_fallback_notice,
    normalize_abstract_fallback_summary,
    validate_method_depth_by_evidence,
)
from scripts.summarize.prepare_digest import atomic_write, load_json, stable_json


_GROUNDING_SCOPE = "title_abstract_and_open_full_text_loose"
_NUMBER_LITERAL = re.compile(
    r"(?<![A-Za-z0-9])(?:~|≈|∼|±)?\s*"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_FALLBACK_ATTEMPTS = re.compile(r"公开全文已尝试获取\s*(\d+)\s*次")
_ORIGINAL_SYSTEM_PROMPT = pipeline.summary_core.system_prompt
_ORIGINAL_RENDER_MARKDOWN = pipeline.summary_core.render_markdown
_ORIGINAL_COMPLETE_JSON = pipeline.ProductionNormalizingClient.complete_json


def _numeric_occurrences(text: str) -> list[tuple[str, Decimal]]:
    values: list[tuple[str, Decimal]] = []
    for match in _NUMBER_LITERAL.finditer(text or ""):
        raw = match.group("number")
        try:
            values.append((raw, Decimal(raw)))
        except InvalidOperation:
            continue
    return values


def _close_enough(output: Decimal, source: Decimal) -> bool:
    """Allow natural approximation and small rounding differences."""

    tolerance = max(Decimal("0.02"), abs(source) * Decimal("0.02"))
    return abs(output - source) <= tolerance


def shared_numeric_grounding(
    summary: dict[str, Any], *, title: str, abstract: str | None
) -> list[str]:
    narrative = {
        key: summary.get(key)
        for key in pipeline.summary_core.NARRATIVE_FIELDS
    }
    source_values = [
        value for _, value in _numeric_occurrences(f"{title}\n{abstract or ''}")
    ]
    unsupported: set[str] = set()
    for raw, output_value in _numeric_occurrences(stable_json(narrative)):
        if not any(_close_enough(output_value, source_value) for source_value in source_values):
            unsupported.add(raw)
    return sorted(unsupported)


def system_prompt(*args: Any, **kwargs: Any) -> str:
    prompt = _ORIGINAL_SYSTEM_PROMPT(*args, **kwargs)
    prompt = prompt.replace(
        "公开正文上下文仅用于定性解释方法，不得据此新增标题或摘要中没有的数字。",
        "数字可以来自标题、摘要或追加的公开正文方法上下文。约数、精确写法、单位排版和轻微四舍五入差异可以自然转换；不得新增全部证据中都没有出现的数字。",
    )
    if str(kwargs.get("information_basis") or "") == ABSTRACT_ONLY_BASIS:
        prompt += (
            "\n当前没有可用的公开全文方法上下文。请生成摘要级短讯：保持结构完整，"
            "但不要为了达到全文级篇幅而扩写摘要未提供的实验、装置、训练或实现细节。"
        )
    return prompt


def _fallback_attempts(user_prompt: str) -> int:
    match = _FALLBACK_ATTEMPTS.search(user_prompt or "")
    return int(match.group(1)) if match else 3


def complete_json_with_fallback_normalization(
    self: Any, **kwargs: Any
) -> pipeline.DeepSeekResponse:
    response = _ORIGINAL_COMPLETE_JSON(self, **kwargs)
    try:
        value = json.loads(response.content)
    except json.JSONDecodeError:
        return response
    if not isinstance(value, dict):
        return response

    value, changed = normalize_abstract_fallback_summary(value)
    verification = value.get("verification")
    if not isinstance(verification, dict):
        verification = {}
        value["verification"] = verification
        changed = True
    if str(verification.get("information_basis") or "") == ABSTRACT_ONLY_BASIS:
        notice = abstract_fallback_notice(
            _fallback_attempts(str(kwargs.get("user_prompt") or ""))
        )
        missing = verification.get("missing_information")
        if not isinstance(missing, list):
            missing = []
            changed = True
        if notice not in missing:
            missing.append(notice)
            changed = True
        verification["missing_information"] = missing

    if changed and hasattr(self, "diagnostics"):
        self.diagnostics.generation_metadata_repair_responses += 1
    return pipeline.DeepSeekResponse(
        content=json.dumps(value, ensure_ascii=False, sort_keys=True),
        usage=response.usage,
        model=response.model,
    )


def render_markdown(*args: Any, **kwargs: Any) -> str:
    markdown = _ORIGINAL_RENDER_MARKDOWN(*args, **kwargs)
    markdown = markdown.replace(
        "> 数字结果仍仅允许来自标题或摘要。",
        "> 数字可以来自标题、摘要或公开正文方法章节，并采用宽松的近似与四舍五入匹配。",
    )
    requests = args[1] if len(args) > 1 else kwargs.get("requests")
    if not isinstance(requests, list) or not requests:
        return markdown

    parts = markdown.split("\n## ")
    if len(parts) != len(requests) + 1:
        return markdown
    rebuilt = [parts[0]]
    for request, section in zip(requests, parts[1:], strict=True):
        if request.get("full_text_fallback") is True:
            attempts = int(request.get("full_text_retrieval_attempts") or 3)
            section = section.replace(
                "- 方法证据：标题、元数据和摘要",
                "- 方法证据：标题、元数据和摘要\n"
                f"- 证据提示：{abstract_fallback_notice(attempts)}",
                1,
            )
        rebuilt.append(section)
    return "\n## ".join(rebuilt)


def _argument_value(argv: list[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _manifest_path(argv: list[str]) -> Path:
    value = _argument_value(argv, "--manifest-path")
    return Path(value) if value else Path("runtime-state/state/summary_generation_manifest.json")


def _mark_manifest_scope(path: Path, argv: list[str]) -> None:
    manifest = load_json(path, {})
    if not isinstance(manifest, dict) or not manifest:
        return
    manifest["numeric_grounding_scope"] = _GROUNDING_SCOPE
    manifest["numeric_matching"] = {
        "approximation_markers_ignored": True,
        "unit_format_ignored": True,
        "relative_tolerance": 0.02,
        "absolute_tolerance": 0.02,
        "evidence_sources": ["title", "abstract", "temporary_open_full_text_methods"],
    }

    dry_run_value = _argument_value(argv, "--dry-run-manifest-path")
    if dry_run_value:
        dry_run = load_json(Path(dry_run_value), {})
        if isinstance(dry_run, dict):
            for key in (
                "request_file",
                "request_sha256",
                "skipped_no_abstract_candidate_ids",
                "skipped_no_abstract_count",
                "abstract_fallback_candidate_ids",
                "abstract_fallback_count",
            ):
                if key in dry_run:
                    manifest[key] = dry_run[key]

    prepared_root_value = _argument_value(argv, "--prepared-root")
    if prepared_root_value:
        audit = load_json(Path(prepared_root_value) / "fulltext_preparation.json", {})
        if isinstance(audit, dict):
            for key in (
                "skipped_no_abstract_candidate_ids",
                "skipped_no_abstract_count",
                "abstract_fallback_candidate_ids",
                "abstract_fallback_count",
                "prepared_request_count",
            ):
                if key in audit:
                    manifest[key] = audit[key]
    atomic_write(
        path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def main() -> int:
    pipeline.shared_numeric_grounding = shared_numeric_grounding
    pipeline.summary_core.system_prompt = system_prompt
    pipeline.summary_core.render_markdown = render_markdown
    pipeline.summary_core.validate_method_depth = validate_method_depth_by_evidence
    pipeline.ProductionNormalizingClient.complete_json = complete_json_with_fallback_normalization
    argv = sys.argv[1:]
    manifest_path = _manifest_path(argv)
    try:
        return pipeline.main()
    finally:
        _mark_manifest_scope(manifest_path, argv)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from scripts.summarize import staged_summary_pipeline as pipeline
from scripts.summarize.abstract_fallback_policy import (
    ABSTRACT_ONLY_BASIS,
    FULL_TEXT_BASIS,
    abstract_fallback_notice,
    normalize_abstract_fallback_summary,
    validate_method_depth_by_evidence,
)
from scripts.summarize.numeric_grounding_repair import (
    format_numeric_diagnostics,
    full_text_evidence_from_user_prompt,
    repair_summary_numeric_grounding,
    source_evidence_from_user_prompt,
    unsupported_numeric_diagnostics,
)
from scripts.summarize.prepare_digest import atomic_write, load_json


_GROUNDING_SCOPE = "title_abstract_and_open_full_text_loose"
_FALLBACK_ATTEMPTS = re.compile(r"公开全文已尝试获取\s*(\d+)\s*次")
_ORIGINAL_SYSTEM_PROMPT = pipeline.summary_core.system_prompt
_ORIGINAL_RENDER_MARKDOWN = pipeline.summary_core.render_markdown
_ORIGINAL_COMPLETE_JSON = pipeline.ProductionNormalizingClient.complete_json
_ORIGINAL_DIAGNOSTIC_PAYLOAD = pipeline.diagnostic_payload
_FULL_TEXT_VALIDATED_HASHES: dict[str, str] = {}


def _numeric_fingerprint(summary: dict[str, Any]) -> str:
    narrative = {
        field: summary.get(field)
        for field in pipeline.summary_core.NARRATIVE_FIELDS
    }
    payload = json.dumps(
        narrative,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shared_numeric_grounding(
    summary: dict[str, Any], *, title: str, abstract: str | None
) -> list[str]:
    verification = summary.get("verification")
    candidate_id = str(summary.get("candidate_id") or "")
    if (
        isinstance(verification, dict)
        and verification.get("full_text_method_context_used") is True
        and candidate_id
        and _FULL_TEXT_VALIDATED_HASHES.get(candidate_id)
        == _numeric_fingerprint(summary)
    ):
        return []

    records = unsupported_numeric_diagnostics(
        summary,
        title=title,
        abstract=abstract,
    )
    return format_numeric_diagnostics(records)


def system_prompt(*args: Any, **kwargs: Any) -> str:
    prompt = _ORIGINAL_SYSTEM_PROMPT(*args, **kwargs)
    prompt = prompt.replace(
        "公开正文上下文仅用于定性解释方法，不得据此新增标题或摘要中没有的数字。",
        "数字可以来自标题、摘要或追加的公开正文方法上下文。约数、精确写法、单位排版和轻微四舍五入差异可以自然转换；不得新增全部证据中都没有出现的数字。",
    )
    if str(kwargs.get("information_basis") or "") == ABSTRACT_ONLY_BASIS:
        fallback_instruction = (
            "\n当前没有可用的公开全文方法上下文。请生成摘要级短讯：保持结构完整，"
            "但不要为了达到全文级篇幅而扩写摘要未提供的实验、装置、训练或实现细节。"
            "不要用阿拉伯数字给步骤或段落编号；摘要只写“常数级”或“有界”时，"
            "不得自行补成 O(1)、[0,1] 或其他具体数学常数。"
        )
        contract_marker = "\nJSON Schema:\n"
        if contract_marker not in prompt:
            raise RuntimeError("Summary system prompt is missing its JSON contract marker")
        prompt = prompt.replace(
            contract_marker,
            fallback_instruction + contract_marker,
            1,
        )
    return prompt


def _fallback_attempts(user_prompt: str) -> int:
    match = _FALLBACK_ATTEMPTS.search(user_prompt or "")
    return int(match.group(1)) if match else 3


def _record_numeric_repairs(
    diagnostics: Any,
    *,
    candidate_id: str,
    repairs: list[dict[str, Any]],
) -> None:
    if not repairs:
        return
    count = int(getattr(diagnostics, "numeric_grounding_repair_responses", 0))
    setattr(diagnostics, "numeric_grounding_repair_responses", count + 1)
    records = getattr(diagnostics, "numeric_grounding_repairs", None)
    if not isinstance(records, dict):
        records = {}
    existing = records.get(candidate_id)
    if not isinstance(existing, list):
        existing = []
    existing.extend(repairs)
    records[candidate_id] = existing
    setattr(diagnostics, "numeric_grounding_repairs", records)


def _candidate_id(value: dict[str, Any], system_prompt_value: str) -> str:
    candidate_id = str(value.get("candidate_id") or "")
    if candidate_id:
        return candidate_id
    try:
        expected = pipeline.expected_example(system_prompt_value)
    except (RuntimeError, ValueError, json.JSONDecodeError):
        return "unknown"
    return str(expected.get("candidate_id") or "unknown")


def _relax_numeric_user_instruction(user_prompt: str) -> str:
    strict = (
        "所有数字仍必须出现在标题或摘要中；即使正文上下文包含额外数字，"
        "也不要把这些数字写入摘要。"
    )
    relaxed = (
        "数字必须有证据支撑：没有公开全文时只能来自标题或摘要；若追加了公开正文方法上下文，"
        "方法说明中的参数数字也可以来自该上下文。近似值、单位排版和轻微四舍五入差异可以自然转换；"
        "不得新增全部证据中都没有出现的数字。"
    )
    return (user_prompt or "").replace(strict, relaxed)


def complete_json_with_fallback_normalization(
    self: Any, **kwargs: Any
) -> pipeline.DeepSeekResponse:
    call_kwargs = dict(kwargs)
    user_prompt = _relax_numeric_user_instruction(str(call_kwargs.get("user_prompt") or ""))
    call_kwargs["user_prompt"] = user_prompt
    response = _ORIGINAL_COMPLETE_JSON(self, **call_kwargs)
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

    information_basis = str(verification.get("information_basis") or "")
    candidate_id = _candidate_id(value, str(call_kwargs.get("system_prompt") or ""))
    title, abstract = source_evidence_from_user_prompt(user_prompt)

    if information_basis == ABSTRACT_ONLY_BASIS:
        notice = abstract_fallback_notice(_fallback_attempts(user_prompt))
        missing = verification.get("missing_information")
        if not isinstance(missing, list):
            missing = []
            changed = True
        if notice not in missing:
            missing.append(notice)
            changed = True
        verification["missing_information"] = missing

        repaired, repairs = repair_summary_numeric_grounding(
            value,
            title=title,
            abstract=abstract,
        )
        if repairs:
            value = repaired
            changed = True
            if hasattr(self, "diagnostics"):
                _record_numeric_repairs(
                    self.diagnostics,
                    candidate_id=candidate_id,
                    repairs=repairs,
                )

    elif information_basis == FULL_TEXT_BASIS:
        full_text_evidence = full_text_evidence_from_user_prompt(user_prompt)
        if full_text_evidence:
            repaired, repairs = repair_summary_numeric_grounding(
                value,
                title=title,
                abstract=abstract,
                extra_evidence=full_text_evidence,
                evidence_scope="标题、摘要或公开正文方法上下文",
            )
            if repairs:
                value = repaired
                changed = True
                if hasattr(self, "diagnostics"):
                    _record_numeric_repairs(
                        self.diagnostics,
                        candidate_id=candidate_id,
                        repairs=repairs,
                    )
            remaining = unsupported_numeric_diagnostics(
                value,
                title=title,
                abstract=abstract,
                extra_evidence=full_text_evidence,
            )
            if not remaining and candidate_id != "unknown":
                _FULL_TEXT_VALIDATED_HASHES[candidate_id] = _numeric_fingerprint(value)

    if changed and hasattr(self, "diagnostics"):
        self.diagnostics.generation_metadata_repair_responses += 1
    return pipeline.DeepSeekResponse(
        content=json.dumps(value, ensure_ascii=False, sort_keys=True),
        usage=response.usage,
        model=response.model,
    )


def diagnostic_payload_with_numeric_repairs(diagnostics: Any) -> dict[str, Any]:
    payload = _ORIGINAL_DIAGNOSTIC_PAYLOAD(diagnostics)
    payload["numeric_grounding_repair_responses"] = int(
        getattr(diagnostics, "numeric_grounding_repair_responses", 0)
    )
    records = getattr(diagnostics, "numeric_grounding_repairs", {})
    payload["numeric_grounding_repairs"] = records if isinstance(records, dict) else {}
    return payload


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
        "unsupported_claim_policy": "full_evidence_deterministic_redaction_then_reject",
        "full_text_unsupported_sentence_action": "deterministic_redaction",
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
    _FULL_TEXT_VALIDATED_HASHES.clear()
    pipeline.shared_numeric_grounding = shared_numeric_grounding
    pipeline.summary_core.system_prompt = system_prompt
    pipeline.summary_core.render_markdown = render_markdown
    pipeline.summary_core.validate_method_depth = validate_method_depth_by_evidence
    pipeline.ProductionNormalizingClient.complete_json = complete_json_with_fallback_normalization
    pipeline.diagnostic_payload = diagnostic_payload_with_numeric_repairs
    argv = sys.argv[1:]
    manifest_path = _manifest_path(argv)
    try:
        return pipeline.main()
    finally:
        _mark_manifest_scope(manifest_path, argv)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Any

ABSTRACT_ONLY_BASIS = "title_metadata_and_abstract_only"
FULL_TEXT_BASIS = "title_metadata_abstract_and_open_full_text_methods"


def abstract_fallback_notice(attempt_count: int) -> str:
    attempts = max(1, int(attempt_count or 1))
    return (
        f"公开全文已尝试获取 {attempts} 次但仍未取得；"
        "本条为摘要级短讯，仅依据题目、元数据和当前可用摘要或摘要片段生成。"
    )


def evidence_basis(summary: dict[str, Any]) -> str:
    verification = summary.get("verification") or {}
    if not isinstance(verification, dict):
        return ABSTRACT_ONLY_BASIS
    return str(verification.get("information_basis") or ABSTRACT_ONLY_BASIS)


def validate_method_depth_by_evidence(
    summary: dict[str, Any],
    *,
    minimum_paragraphs: int = 2,
    maximum_paragraphs: int = 6,
    full_text_principle_minimum: int = 160,
    abstract_principle_minimum: int = 60,
    abstract_minimum_paragraphs: int = 1,
) -> list[str]:
    """Apply strict depth only when method-oriented full text was actually used."""

    basis = evidence_basis(summary)
    principle_minimum = (
        full_text_principle_minimum
        if basis == FULL_TEXT_BASIS
        else abstract_principle_minimum
    )
    required_paragraphs = (
        minimum_paragraphs
        if basis == FULL_TEXT_BASIS
        else abstract_minimum_paragraphs
    )

    errors: list[str] = []
    principle = str(summary.get("method_principle") or "")
    paragraphs = summary.get("method_implementation") or []
    if len(principle) < principle_minimum:
        errors.append(
            f"method_principle is too short for {basis}: "
            f"{len(principle)} < {principle_minimum}"
        )
    if (
        not isinstance(paragraphs, list)
        or not required_paragraphs <= len(paragraphs) <= maximum_paragraphs
    ):
        errors.append(
            "method_implementation must contain "
            f"{required_paragraphs}-{maximum_paragraphs} paragraphs for {basis}"
        )
    return errors


def normalize_abstract_fallback_summary(value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Repair safe shape-only variations without adding scientific content."""

    changed = False
    details = value.get("optical_neural_network_analysis")
    if isinstance(details, dict):
        tasks = details.get("application_tasks")
        if isinstance(tasks, str):
            text = tasks.strip()
            details["application_tasks"] = [] if text in {"", "未提供", "not_available"} else [text]
            changed = True

    implementation = value.get("method_implementation")
    if isinstance(implementation, str):
        text = implementation.strip()
        value["method_implementation"] = [text] if text else []
        changed = True

    return value, changed

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any


NARRATIVE_FIELDS = (
    "core_problem",
    "method_and_architecture",
    "method_principle",
    "method_implementation",
    "main_contributions",
    "reported_results",
    "distinction_from_prior_work",
    "research_value",
    "limitations_and_open_questions",
    "optical_neural_network_analysis",
    "zeroth_order_analysis",
)

_NUMBER_LITERAL = re.compile(
    r"(?<![A-Za-z0-9])(?:~|≈|∼|±)?\s*"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_SENTENCE_PART = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]*")
_ORDERED_PREFIX = re.compile(
    r"^\s*(?:第\s*)?(?P<number>\d+)\s*(?:步(?:骤)?|[.)、:：])\s*"
)
_BIG_O_ONE = re.compile(r"(?i)O\s*[（(]\s*1\s*[)）]")
_BOUNDED_ZERO_ONE = re.compile(
    r"(?:[\[（(]\s*0\s*[,，]\s*1\s*[\]）)]|"
    r"0\s*(?:到|至|[-–—~～])\s*1|介于\s*0\s*(?:和|与|到|至)\s*1\s*之间)"
)
_BOUNDED_CONTEXT = re.compile(
    r"收缩|因子|权重|系数|概率|比例|门控|shrink|factor|weight|coefficient",
    re.IGNORECASE,
)
_SOURCE_MARKER = "来源记录：\n"


def _numeric_occurrences(text: str) -> list[tuple[str, Decimal, int, int]]:
    values: list[tuple[str, Decimal, int, int]] = []
    for match in _NUMBER_LITERAL.finditer(text or ""):
        raw = match.group("number")
        try:
            values.append((raw, Decimal(raw), match.start(), match.end()))
        except InvalidOperation:
            continue
    return values


def _close_enough(output: Decimal, source: Decimal) -> bool:
    tolerance = max(Decimal("0.02"), abs(source) * Decimal("0.02"))
    return abs(output - source) <= tolerance


def _source_values(title: str, abstract: str | None) -> list[Decimal]:
    return [
        value
        for _, value, _, _ in _numeric_occurrences(f"{title}\n{abstract or ''}")
    ]


def _unsupported_in_text(text: str, source_values: list[Decimal]) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    for raw, value, start, end in _numeric_occurrences(text):
        if any(_close_enough(value, source) for source in source_values):
            continue
        unsupported.append(
            {
                "token": raw,
                "start": start,
                "end": end,
            }
        )
    return unsupported


def _excerpt(text: str, start: int, end: int, radius: int = 60) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _walk_strings(value: Any, path: str) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, list):
        return [
            pair
            for index, item in enumerate(value)
            for pair in _walk_strings(item, f"{path}[{index}]")
        ]
    if isinstance(value, dict):
        return [
            pair
            for key, item in value.items()
            for pair in _walk_strings(item, f"{path}.{key}" if path else str(key))
        ]
    return []


def unsupported_numeric_diagnostics(
    summary: dict[str, Any], *, title: str, abstract: str | None
) -> list[dict[str, Any]]:
    source_values = _source_values(title, abstract)
    diagnostics: list[dict[str, Any]] = []
    for field in NARRATIVE_FIELDS:
        for path, text in _walk_strings(summary.get(field), field):
            for occurrence in _unsupported_in_text(text, source_values):
                diagnostics.append(
                    {
                        "path": path,
                        "token": occurrence["token"],
                        "excerpt": _excerpt(
                            text,
                            int(occurrence["start"]),
                            int(occurrence["end"]),
                        ),
                    }
                )
    return diagnostics


def format_numeric_diagnostics(records: list[dict[str, Any]]) -> list[str]:
    return [
        f"{record['token']} at {record['path']}: {record['excerpt']}"
        for record in records
    ]


def _safe_idiom_rewrites(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    updated = text

    if _ORDERED_PREFIX.search(updated):
        updated = _ORDERED_PREFIX.sub("", updated, count=1)
        actions.append("removed_numeric_sequence_prefix")

    if _BIG_O_ONE.search(updated):
        updated = _BIG_O_ONE.sub("常数级", updated)
        actions.append("rewrote_big_o_one")

    if _BOUNDED_CONTEXT.search(updated) and _BOUNDED_ZERO_ONE.search(updated):
        updated = _BOUNDED_ZERO_ONE.sub("一个有界范围", updated)
        actions.append("rewrote_bounded_zero_one_interval")

    return updated, actions


def _redact_unsupported_sentences(
    text: str,
    *,
    source_values: list[Decimal],
    result_field: bool,
) -> tuple[str, list[dict[str, Any]]]:
    repairs: list[dict[str, Any]] = []
    pieces = _SENTENCE_PART.findall(text)
    if not pieces:
        pieces = [text]
    output: list[str] = []
    placeholder = (
        "相关定量结果未在标题或摘要中提供。"
        if result_field
        else "相关定量细节未在标题或摘要中提供。"
    )
    for piece in pieces:
        unsupported = _unsupported_in_text(piece, source_values)
        if not unsupported:
            output.append(piece)
            continue
        repairs.append(
            {
                "action": "redacted_unsupported_sentence",
                "tokens": sorted({str(item["token"]) for item in unsupported}),
                "original_excerpt": re.sub(r"\s+", " ", piece).strip()[:240],
                "replacement": placeholder,
            }
        )
        if not output or output[-1] != placeholder:
            output.append(placeholder)
    return "".join(output).strip(), repairs


def _repair_value(
    value: Any,
    *,
    path: str,
    source_values: list[Decimal],
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, str):
        updated, actions = _safe_idiom_rewrites(value)
        repairs: list[dict[str, Any]] = []
        if actions:
            repairs.append(
                {
                    "path": path,
                    "action": "+".join(actions),
                    "original_excerpt": re.sub(r"\s+", " ", value).strip()[:240],
                    "replacement": re.sub(r"\s+", " ", updated).strip()[:240],
                }
            )
        updated, redactions = _redact_unsupported_sentences(
            updated,
            source_values=source_values,
            result_field=path.startswith("reported_results"),
        )
        for record in redactions:
            record["path"] = path
        repairs.extend(redactions)
        return updated, repairs

    if isinstance(value, list):
        output: list[Any] = []
        repairs: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            repaired, child_repairs = _repair_value(
                item,
                path=f"{path}[{index}]",
                source_values=source_values,
            )
            output.append(repaired)
            repairs.extend(child_repairs)
        return output, repairs

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        repairs: list[dict[str, Any]] = []
        for key, item in value.items():
            repaired, child_repairs = _repair_value(
                item,
                path=f"{path}.{key}" if path else str(key),
                source_values=source_values,
            )
            output[str(key)] = repaired
            repairs.extend(child_repairs)
        return output, repairs

    return value, []


def repair_summary_numeric_grounding(
    summary: dict[str, Any], *, title: str, abstract: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_values = _source_values(title, abstract)
    output = dict(summary)
    repairs: list[dict[str, Any]] = []
    for field in NARRATIVE_FIELDS:
        if field not in output:
            continue
        repaired, field_repairs = _repair_value(
            output[field],
            path=field,
            source_values=source_values,
        )
        output[field] = repaired
        repairs.extend(field_repairs)
    return output, repairs


def source_evidence_from_user_prompt(user_prompt: str) -> tuple[str, str | None]:
    if _SOURCE_MARKER not in user_prompt:
        return "", None
    tail = user_prompt.split(_SOURCE_MARKER, 1)[1].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(tail)
    except (json.JSONDecodeError, TypeError):
        return "", None
    if not isinstance(value, dict):
        return "", None
    return str(value.get("title") or ""), (
        str(value.get("abstract")) if value.get("abstract") is not None else None
    )

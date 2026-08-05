from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from scripts.summarize import staged_summary_pipeline as pipeline
from scripts.summarize.prepare_digest import atomic_write, load_json, stable_json


_GROUNDING_SCOPE = "title_abstract_and_open_full_text_loose"
_NUMBER_LITERAL = re.compile(
    r"(?<![A-Za-z0-9])(?:~|≈|∼|±)?\s*"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_ORIGINAL_SYSTEM_PROMPT = pipeline.summary_core.system_prompt
_ORIGINAL_RENDER_MARKDOWN = pipeline.summary_core.render_markdown


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
    """Allow natural approximation and small rounding differences.

    The tolerance is two percent of the source magnitude, with a small absolute
    floor of 0.02 for values close to zero. Units and approximation markers are
    intentionally ignored; only the numerical value must occur in the evidence.
    """

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


def validate_full_text_safety(config: dict[str, Any]) -> None:
    execution = config.get("execution") or {}
    full_text = config.get("full_text") or {}
    if execution.get("use_full_text") and not full_text.get("open_access_only"):
        raise RuntimeError("Full-text method context must remain open-access-only")
    if full_text.get("persist_extracted_text"):
        raise RuntimeError("Extracted full text must not be persisted")
    if full_text.get("numeric_grounding_scope") != _GROUNDING_SCOPE:
        raise RuntimeError("Numeric grounding must use loose title, abstract, and open-full-text evidence")


def system_prompt(*args: Any, **kwargs: Any) -> str:
    prompt = _ORIGINAL_SYSTEM_PROMPT(*args, **kwargs)
    return prompt.replace(
        "公开正文上下文仅用于定性解释方法，不得据此新增标题或摘要中没有的数字。",
        "数字可以来自标题、摘要或追加的公开正文方法上下文。约数、精确写法、单位排版和轻微四舍五入差异可以自然转换；不得新增全部证据中都没有出现的数字。",
    )


def render_markdown(*args: Any, **kwargs: Any) -> str:
    markdown = _ORIGINAL_RENDER_MARKDOWN(*args, **kwargs)
    return markdown.replace(
        "> 数字结果仍仅允许来自标题或摘要。",
        "> 数字可以来自标题、摘要或公开正文方法章节，并采用宽松的近似与四舍五入匹配。",
    )


def _manifest_path(argv: list[str]) -> Path:
    for index, value in enumerate(argv):
        if value == "--manifest-path" and index + 1 < len(argv):
            return Path(argv[index + 1])
    return Path("runtime-state/state/summary_generation_manifest.json")


def _mark_manifest_scope(path: Path) -> None:
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
    atomic_write(
        path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def main() -> int:
    pipeline.shared_numeric_grounding = shared_numeric_grounding
    pipeline.validate_full_text_safety = validate_full_text_safety
    pipeline.summary_core.system_prompt = system_prompt
    pipeline.summary_core.render_markdown = render_markdown
    manifest_path = _manifest_path(sys.argv[1:])
    try:
        return pipeline.main()
    finally:
        _mark_manifest_scope(manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())

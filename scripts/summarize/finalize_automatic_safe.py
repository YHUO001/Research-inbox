from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.summarize import finalize_automatic as base
from scripts.summarize.abstract_fallback_policy import validate_method_depth_by_evidence
from scripts.summarize.prepare_digest import atomic_write, load_json


_ORIGINAL_FINALIZE = base.finalize_automatic
_REQUIRED_ARTIFACTS = (
    ("request_file", "request"),
    ("summary_file", "summary"),
    ("digest_json_file", "digest JSON"),
    ("digest_markdown_file", "digest Markdown"),
)


def validate_required_artifacts(generation_manifest_path: Path) -> None:
    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict):
        raise RuntimeError("Summary generation manifest must be a JSON object")
    for field, label in _REQUIRED_ARTIFACTS:
        raw = generation.get(field)
        value = str(raw).strip() if raw is not None else ""
        if not value:
            raise RuntimeError(f"Summary generation manifest is missing {field}")
        path = Path(value)
        if not path.is_file():
            raise RuntimeError(f"Missing {label} artifact file: {path}")


def finalize_with_fallback(**kwargs: Any) -> dict[str, Any]:
    generation_manifest_path = Path(str(kwargs.get("generation_manifest_path") or ""))
    validate_required_artifacts(generation_manifest_path)
    result = _ORIGINAL_FINALIZE(**kwargs)
    skipped = [
        str(item)
        for item in result.get("skipped_no_abstract_candidate_ids") or []
        if str(item)
    ]
    if not skipped:
        return result

    digest_markdown_path = Path(str(result.get("digest_markdown_file") or ""))
    digest_json_path = Path(str(result.get("digest_json_file") or ""))
    if digest_markdown_path.exists():
        content = digest_markdown_path.read_text(encoding="utf-8").rstrip()
        marker = "\n## 证据不足而跳过\n"
        if marker not in content:
            content += (
                marker
                + "\n"
                + f"- 跳过论文数：`{len(skipped)}`\n"
                + "- 原因：公开全文获取三次仍失败，且没有可用摘要或摘要片段；"
                "未调用模型，也未写入完成历史。\n"
            )
            atomic_write(digest_markdown_path, content)

    digest = load_json(digest_json_path, {})
    if isinstance(digest, dict) and digest:
        digest["skipped_no_abstract_candidate_ids"] = skipped
        digest["skipped_no_abstract_count"] = len(skipped)
        atomic_write(
            digest_json_path,
            json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    return result


def main() -> int:
    base.validate_method_depth = validate_method_depth_by_evidence
    base.finalize_automatic = finalize_with_fallback
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

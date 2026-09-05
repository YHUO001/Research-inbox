from __future__ import annotations

from typing import Any


def render_author_context(author_context: dict[str, Any] | None) -> str:
    """Render deterministic author metadata for user-facing digests.

    This formatter only displays supplied metadata. It does not infer reputation,
    background, or affiliations that are not present in the source record.
    """
    if not author_context:
        return ""

    lines: list[str] = ["## 作者信息"]

    first_author = author_context.get("first_author")
    if isinstance(first_author, dict):
        name = str(first_author.get("name") or "未知")
        institutions = ", ".join(first_author.get("institutions") or []) or "未提供"
        lines.extend([
            f"- 第一作者：{name}",
            f"- 所属机构：{institutions}",
        ])

    corresponding = author_context.get("corresponding_authors") or []
    if corresponding:
        lines.append("- 通讯作者：")
        for author in corresponding:
            if isinstance(author, dict):
                name = str(author.get("name") or "未知")
                institutions = ", ".join(author.get("institutions") or []) or "未提供"
                lines.append(f"  - {name}（{institutions}）")

    return "\n".join(lines)

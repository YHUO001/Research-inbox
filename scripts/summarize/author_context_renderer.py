from __future__ import annotations

from typing import Any


def _author_label(author: dict[str, Any]) -> str:
    name = str(author.get("name") or "未知").strip() or "未知"
    institutions = [str(item).strip() for item in author.get("institutions") or [] if str(item).strip()]
    return f"{name}（{'；'.join(institutions)}）" if institutions else name


def render_author_context(author_context: dict[str, Any] | None) -> str:
    """Render deterministic author/team metadata for user-facing digests.

    The formatter only displays supplied metadata. It never guesses a
    corresponding author from author order and never invents affiliations.
    """
    if not author_context:
        return ""

    authors = [item for item in author_context.get("authors") or [] if isinstance(item, dict)]
    first_author = author_context.get("first_author")
    corresponding = [
        item for item in author_context.get("corresponding_authors") or [] if isinstance(item, dict)
    ]
    institutions = [
        str(item).strip()
        for item in author_context.get("institutions") or []
        if str(item).strip()
    ]

    lines: list[str] = ["### 作者与团队", ""]
    if authors:
        lines.append("- 作者：" + "；".join(_author_label(author) for author in authors))
    elif isinstance(first_author, dict):
        lines.append(f"- 作者：{_author_label(first_author)}")
    else:
        lines.append("- 作者：公开元数据未提供")

    if isinstance(first_author, dict):
        lines.append(f"- 第一作者：{_author_label(first_author)}")
    else:
        lines.append("- 第一作者：公开元数据未提供")

    if corresponding:
        lines.append("- 通讯作者：" + "；".join(_author_label(author) for author in corresponding))
    else:
        lines.append("- 通讯作者：公开元数据未明确标注（不按末位作者推断）")

    lines.append("- 主要机构：" + ("；".join(institutions) if institutions else "公开元数据未提供"))
    if len(institutions) > 1:
        lines.append(
            f"- 团队概况：公开元数据显示为跨机构合作，涉及 {len(institutions)} 个主要机构；这里只描述元数据，不推断 PI 关系或团队历史。"
        )
    elif len(institutions) == 1:
        lines.append(
            "- 团队概况：公开元数据显示作者主要归属于同一主要机构；这里只描述元数据，不推断 PI 关系或团队历史。"
        )
    else:
        lines.append("- 团队概况：当前公开元数据不足以判断团队机构构成。")

    return "\n".join(lines)

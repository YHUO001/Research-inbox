"""Digest integration helper for author metadata sections.

Keeps author metadata rendering separate from generated scientific content.
"""

from __future__ import annotations

from typing import Any

from scripts.summarize.author_context_renderer import render_author_context



def append_author_section(markdown: str, summary: dict[str, Any]) -> str:
    """Append author metadata to a digest entry when metadata exists.

    The summary body is preserved; only deterministic metadata is appended.
    """
    section = render_author_context(summary.get("author_context"))
    if not section:
        return markdown

    markdown = markdown.rstrip()
    if "## 作者信息" in markdown:
        return markdown

    return f"{markdown}\n\n{section}\n"

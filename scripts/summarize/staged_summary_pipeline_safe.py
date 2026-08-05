from __future__ import annotations

from typing import Any

from scripts.summarize import staged_summary_pipeline as pipeline
from scripts.summarize.prepare_digest import numeric_tokens


_ORIGINAL_SHARED_NUMERIC_GROUNDING = pipeline.shared_numeric_grounding


def shared_numeric_grounding(
    summary: dict[str, Any], *, title: str, abstract: str | None
) -> list[str]:
    """Keep strict grounding while allowing exact-source values to be weakened.

    An output such as ``~65 TOPS`` is a weaker statement than an exact source
    value of ``65 TOPS`` and is therefore safe. The reverse direction remains
    prohibited: an approximate source value such as ``~5 GHz`` may not become
    the exact claim ``5 GHz``.
    """

    unsupported = _ORIGINAL_SHARED_NUMERIC_GROUNDING(
        summary,
        title=title,
        abstract=abstract,
    )
    source_tokens = numeric_tokens(f"{title}\n{abstract or ''}")
    return [
        token
        for token in unsupported
        if not (token.startswith("~") and token[1:] in source_tokens)
    ]


# generate_stage resolves this module global at runtime before installing the
# validator into generate_summaries, so the compatibility wrapper remains small.
pipeline.shared_numeric_grounding = shared_numeric_grounding


def main() -> int:
    return pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Any, Callable

from scripts.summarize import prepare_fulltext_bounded as bounded
from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.publisher_fulltext import collect_publisher_routed_context


TERMINAL_RETRIEVAL_STATUSES = {
    "authentication_failed",
    "forbidden",
    "not_applicable",
}


def routed_bounded_collect_method_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    loader: Callable[..., MethodContext] = collect_publisher_routed_context,
) -> MethodContext:
    """Retry transient retrieval failures, but stop immediately on terminal failures."""

    maximum_attempts = max(1, int(config.get("retrieval_attempts") or 3))
    last: MethodContext | None = None
    for attempt in range(1, maximum_attempts + 1):
        last = bounded._bounded_collect_once(source, config=config, loader=loader)
        if last.status == "used" and last.text:
            return bounded.retried_context(last, attempt)
        if last.status in TERMINAL_RETRIEVAL_STATUSES:
            return bounded.retried_context(last, attempt, final=True)

    if last is None:
        last = MethodContext(
            candidate_id=str(source.get("candidate_id") or source.get("id") or "unknown"),
            status="not_available",
            source_url=bounded._timeout_source_url(source, config),
            media_type=None,
            section_headings=[],
            text="",
            error="full-text retrieval did not start",
        )
    return bounded.retried_context(last, maximum_attempts, final=True)


def main() -> int:
    bounded.bounded_collect_method_context = routed_bounded_collect_method_context
    return bounded.main()


if __name__ == "__main__":
    raise SystemExit(main())

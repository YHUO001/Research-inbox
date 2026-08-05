from __future__ import annotations

import time

from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_fulltext_bounded import bounded_collect_method_context


def source() -> dict:
    return {
        "candidate_id": "candidate-timeout",
        "open_access_url": "https://doi.org/10.1000/slow",
    }


def test_slow_candidate_is_interrupted_and_falls_back() -> None:
    def slow_loader(item: dict, *, config: dict) -> MethodContext:
        del item, config
        time.sleep(2)
        raise AssertionError("hard timeout did not interrupt the loader")

    started = time.monotonic()
    context = bounded_collect_method_context(
        source(),
        config={"candidate_timeout_seconds": 0.05, "candidate_url_limit": 3},
        loader=slow_loader,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert context.status == "timed_out"
    assert context.source_url == "https://doi.org/10.1000/slow"
    assert context.text == ""
    assert context.error and "fell back to abstract" in context.error
    assert context.audit_record()["text_persisted"] is False


def test_fast_candidate_preserves_method_context() -> None:
    expected = MethodContext(
        candidate_id="candidate-timeout",
        status="used",
        source_url="https://example.org/article",
        media_type="text/html",
        section_headings=["Methods"],
        text="The input is encoded optically and the detected output is used for classification.",
    )

    def fast_loader(item: dict, *, config: dict) -> MethodContext:
        del item, config
        return expected

    context = bounded_collect_method_context(
        source(),
        config={"candidate_timeout_seconds": 1},
        loader=fast_loader,
    )
    assert context == expected

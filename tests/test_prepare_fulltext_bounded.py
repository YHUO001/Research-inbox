from __future__ import annotations

import time

from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_fulltext_bounded import bounded_collect_method_context
from scripts.summarize.springer_openaccess import api_audit_url


def source() -> dict:
    return {
        "candidate_id": "candidate-timeout",
        "doi": "10.1000/slow",
        "open_access_url": "https://doi.org/10.1000/slow",
    }


def test_slow_candidate_is_interrupted_three_times_then_falls_back() -> None:
    def slow_loader(item: dict, *, config: dict) -> MethodContext:
        del item, config
        time.sleep(2)
        raise AssertionError("hard timeout did not interrupt the loader")

    started = time.monotonic()
    context = bounded_collect_method_context(
        source(),
        config={
            "candidate_timeout_seconds": 0.05,
            "candidate_url_limit": 3,
            "retrieval_attempts": 3,
        },
        loader=slow_loader,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert context.status == "timed_out"
    assert context.source_url == api_audit_url(
        "https://api.springernature.com/openaccess/jats",
        "10.1000/slow",
    )
    assert "api_key" not in context.source_url
    assert context.text == ""
    assert context.error and "failed after 3 attempts" in context.error
    assert context.audit_record()["attempt_count"] == 3
    assert context.audit_record()["text_persisted"] is False


def test_third_attempt_can_recover_method_context() -> None:
    attempts = 0
    expected = MethodContext(
        candidate_id="candidate-timeout",
        status="used",
        source_url="https://example.org/article",
        media_type="text/html",
        section_headings=["Methods"],
        text="The input is encoded optically and the detected output is used for classification.",
    )

    def recovering_loader(item: dict, *, config: dict) -> MethodContext:
        nonlocal attempts
        del item, config
        attempts += 1
        if attempts < 3:
            return MethodContext(
                candidate_id="candidate-timeout",
                status="not_available",
                source_url="https://example.org/article",
                media_type="text/html",
                section_headings=[],
                text="",
                error="temporary failure",
            )
        return expected

    context = bounded_collect_method_context(
        source(),
        config={"candidate_timeout_seconds": 0, "retrieval_attempts": 3},
        loader=recovering_loader,
    )
    assert attempts == 3
    assert context.status == expected.status
    assert context.source_url == expected.source_url
    assert context.text == expected.text
    assert context.audit_record()["attempt_count"] == 3


def test_fast_candidate_stops_after_first_attempt() -> None:
    attempts = 0

    def fast_loader(item: dict, *, config: dict) -> MethodContext:
        nonlocal attempts
        del item, config
        attempts += 1
        return MethodContext(
            candidate_id="candidate-timeout",
            status="used",
            source_url="https://example.org/article",
            media_type="text/html",
            section_headings=["Methods"],
            text="The input is encoded optically and the detected output is used for classification.",
        )

    context = bounded_collect_method_context(
        source(),
        config={"candidate_timeout_seconds": 0, "retrieval_attempts": 3},
        loader=fast_loader,
    )
    assert attempts == 1
    assert context.status == "used"
    assert context.audit_record()["attempt_count"] == 1

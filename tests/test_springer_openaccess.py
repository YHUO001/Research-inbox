from __future__ import annotations

import time

from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_fulltext_bounded import bounded_collect_method_context
from scripts.summarize.springer_openaccess import (
    api_audit_url,
    collect_official_or_open_context,
    collect_springer_openaccess_context,
    extract_jats_method_context,
    non_springer_open_source,
)


JATS_WITH_METHODS = b"""<?xml version='1.0' encoding='UTF-8'?>
<response>
  <records>
    <article>
      <body>
        <sec sec-type='intro'>
          <title>Introduction</title>
          <p>This background paragraph should not be used as method context.</p>
        </sec>
        <sec sec-type='methods'>
          <title>Materials and methods</title>
          <p>The optical input is encoded by a programmable modulator and propagated through the computing stage.</p>
          <p>The detector records the output before an electronic readout produces the final prediction.</p>
        </sec>
        <sec sec-type='results'>
          <title>Results</title>
          <p>This result paragraph should not be included in method context.</p>
        </sec>
      </body>
    </article>
  </records>
</response>
"""

JATS_WITHOUT_RECORDS = b"""<?xml version='1.0' encoding='UTF-8'?>
<response><result><total>0</total></result><records /></response>
"""


def _slow_loader(source: dict, *, config: dict) -> MethodContext:
    time.sleep(5)
    return MethodContext(
        str(source.get("candidate_id") or "unknown"),
        "used",
        "https://example.org/late",
        "text/html",
        ["Methods"],
        "This result should never be returned because the child must be terminated.",
    )


def _fast_loader(source: dict, *, config: dict) -> MethodContext:
    return MethodContext(
        str(source.get("candidate_id") or "unknown"),
        "used",
        "https://example.org/open.xml",
        "application/xml+jats",
        ["Methods"],
        "The optical input is processed by the documented experimental implementation.",
    )


def test_extract_jats_method_sections_only() -> None:
    text, headings = extract_jats_method_context(
        JATS_WITH_METHODS,
        maximum_characters=2000,
    )
    assert headings == ["Materials and methods"]
    assert "programmable modulator" in text
    assert "electronic readout" in text
    assert "background paragraph" not in text
    assert "result paragraph" not in text


def test_openaccess_query_uses_key_but_audit_url_does_not() -> None:
    captured: list[str] = []

    def fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int) -> bytes:
        captured.append(url)
        return JATS_WITH_METHODS

    context = collect_springer_openaccess_context(
        {
            "candidate_id": "paper-1",
            "doi": "10.1038/s41377-026-02394-3",
        },
        config={
            "springer_openaccess_endpoint": "https://api.springernature.com/openaccess/jats",
            "maximum_method_characters": 2000,
        },
        api_key="private-test-key",
        fetcher=fetcher,
    )
    assert context.status == "used"
    assert captured and "api_key=private-test-key" in captured[0]
    assert context.source_url == api_audit_url(
        "https://api.springernature.com/openaccess/jats",
        "10.1038/s41377-026-02394-3",
    )
    assert "private-test-key" not in str(context.audit_record())
    assert context.audit_record()["text_persisted"] is False


def test_openaccess_no_record_falls_back_without_direct_nature_request() -> None:
    calls: list[str] = []

    def fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int) -> bytes:
        calls.append(url)
        return JATS_WITHOUT_RECORDS

    direct_calls: list[dict] = []

    def direct_loader(source: dict, *, config: dict) -> MethodContext:
        direct_calls.append(source)
        raise AssertionError("Nature and DOI pages must not be fetched directly")

    context = collect_official_or_open_context(
        {
            "candidate_id": "paper-nature",
            "doi": "10.1038/s41467-026-76128-9",
            "open_access_url": "https://www.nature.com/articles/s41467-026-76128-9",
            "landing_page": "https://doi.org/10.1038/s41467-026-76128-9",
        },
        config={"allow_non_springer_open_urls": True},
        api_key="private-test-key",
        api_fetcher=fetcher,
        direct_loader=direct_loader,
    )
    assert context.status == "not_available"
    assert calls
    assert direct_calls == []
    assert context.error and "fell back to abstract" in context.error


def test_non_springer_repository_url_remains_available_as_fallback() -> None:
    source = non_springer_open_source(
        {
            "candidate_id": "paper-repository",
            "doi": "10.1234/example",
            "open_access_url": "https://repository.example.edu/paper.pdf",
            "landing_page": "https://doi.org/10.1234/example",
        }
    )
    assert source == {
        "candidate_id": "paper-repository",
        "open_access_url": "https://repository.example.edu/paper.pdf",
        "landing_page": None,
        "doi": None,
    }


def test_official_api_failure_can_use_non_springer_repository() -> None:
    def fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int) -> bytes:
        return JATS_WITHOUT_RECORDS

    def direct_loader(source: dict, *, config: dict) -> MethodContext:
        assert source["open_access_url"] == "https://repository.example.edu/paper.xml"
        assert source["doi"] is None
        return MethodContext(
            "paper-repository",
            "used",
            source["open_access_url"],
            "application/xml",
            ["Methods"],
            "The repository record provides the open method implementation context.",
        )

    context = collect_official_or_open_context(
        {
            "candidate_id": "paper-repository",
            "doi": "10.1234/example",
            "open_access_url": "https://repository.example.edu/paper.xml",
        },
        config={"allow_non_springer_open_urls": True},
        api_key="private-test-key",
        api_fetcher=fetcher,
        direct_loader=direct_loader,
    )
    assert context.status == "used"
    assert context.source_url == "https://repository.example.edu/paper.xml"


def test_child_process_enforces_wall_clock_timeout() -> None:
    started = time.monotonic()
    context = bounded_collect_method_context(
        {
            "candidate_id": "paper-timeout",
            "doi": "10.1038/example",
        },
        config={
            "candidate_timeout_seconds": 0.1,
            "springer_openaccess_endpoint": "https://api.springernature.com/openaccess/jats",
        },
        loader=_slow_loader,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 2
    assert context.status == "timed_out"
    assert context.text == ""
    assert context.error and "fell back to abstract" in context.error


def test_child_process_returns_fast_context() -> None:
    context = bounded_collect_method_context(
        {"candidate_id": "paper-fast", "doi": "10.1038/example"},
        config={"candidate_timeout_seconds": 2},
        loader=_fast_loader,
    )
    assert context.status == "used"
    assert context.section_headings == ["Methods"]

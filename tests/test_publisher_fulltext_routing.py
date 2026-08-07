from __future__ import annotations

from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_fulltext_routed import (
    routed_bounded_collect_method_context,
)
from scripts.summarize.publisher_fulltext import (
    arxiv_public_urls,
    collect_publisher_routed_context,
    is_springer_nature_source,
    routed_open_source,
)


JATS_WITH_METHODS = b"""<?xml version='1.0' encoding='UTF-8'?>
<response><records><article><body>
<sec sec-type='methods'><title>Methods</title>
<p>The optical system uses a documented open implementation and measured output.</p>
</sec></body></article></records></response>
"""


def test_springer_detection_uses_doi_host_or_venue() -> None:
    assert is_springer_nature_source({"doi": "10.1007/s12345-001-0001-1"})
    assert is_springer_nature_source({"doi": "10.1038/example"})
    assert is_springer_nature_source(
        {"landing_page": "https://link.springer.com/article/example"}
    )
    assert is_springer_nature_source({"venue": "Nature Photonics"})
    assert not is_springer_nature_source({"doi": "10.1364/OPTICA.591264"})
    assert not is_springer_nature_source({"doi": "10.1117/1.OE.65.7.073102"})


def test_arxiv_doi_derives_public_html_and_pdf_urls() -> None:
    assert arxiv_public_urls("10.48550/arXiv.2608.03277") == [
        "https://arxiv.org/html/2608.03277",
        "https://arxiv.org/pdf/2608.03277",
    ]
    source = routed_open_source(
        {
            "candidate_id": "paper-arxiv",
            "doi": "10.48550/arXiv.2608.03277",
            "open_access_url": "https://doi.org/10.48550/arXiv.2608.03277",
        },
        include_doi_fallback=True,
    )
    assert source is not None
    assert source["open_access_url"] == "https://arxiv.org/html/2608.03277"
    assert source["landing_page"] == "https://arxiv.org/pdf/2608.03277"


def test_non_springer_doi_skips_springer_api_and_uses_public_route() -> None:
    api_calls: list[str] = []
    direct_calls: list[dict] = []

    def api_fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int) -> bytes:
        api_calls.append(url)
        raise AssertionError("Non-Springer DOI must not call Springer API")

    def direct_loader(source: dict, *, config: dict) -> MethodContext:
        direct_calls.append(source)
        return MethodContext(
            "paper-optica",
            "used",
            source["open_access_url"],
            "text/html",
            ["Methods"],
            "The public publisher page provides the method implementation.",
        )

    context = collect_publisher_routed_context(
        {
            "candidate_id": "paper-optica",
            "doi": "10.1364/OPTICA.591264",
            "open_access_url": "https://doi.org/10.1364/OPTICA.591264",
            "landing_page": "https://doi.org/10.1364/OPTICA.591264",
            "venue": "Optica",
        },
        config={"allow_non_springer_open_urls": True},
        api_key="unused-key",
        api_fetcher=api_fetcher,
        direct_loader=direct_loader,
    )

    assert context.status == "used"
    assert api_calls == []
    assert direct_calls[0]["open_access_url"] == (
        "https://doi.org/10.1364/optica.591264"
    )


def test_springer_source_uses_official_jats_api() -> None:
    api_calls: list[str] = []

    def api_fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int) -> bytes:
        api_calls.append(url)
        return JATS_WITH_METHODS

    def direct_loader(source: dict, *, config: dict) -> MethodContext:
        raise AssertionError("Successful Springer JATS lookup must not use direct fetch")

    context = collect_publisher_routed_context(
        {
            "candidate_id": "paper-springer",
            "doi": "10.1007/s12345-001-0001-1",
            "venue": "Springer Journal",
        },
        config={"allow_non_springer_open_urls": True},
        api_key="private-test-key",
        api_fetcher=api_fetcher,
        direct_loader=direct_loader,
    )

    assert context.status == "used"
    assert api_calls and "api.springernature.com/openaccess/jats" in api_calls[0]


def test_authentication_failure_is_not_retried_three_times() -> None:
    calls = 0

    def auth_loader(source: dict, *, config: dict) -> MethodContext:
        nonlocal calls
        calls += 1
        return MethodContext(
            str(source["candidate_id"]),
            "authentication_failed",
            "https://api.springernature.com/openaccess/jats?q=doi:test&p=1",
            "application/xml+jats",
            [],
            "",
            "Springer Open Access HTTP 401; fell back to abstract",
        )

    context = routed_bounded_collect_method_context(
        {"candidate_id": "paper-auth", "doi": "10.1007/example"},
        config={"retrieval_attempts": 3, "candidate_timeout_seconds": -1},
        loader=auth_loader,
    )

    assert calls == 1
    assert context.status == "authentication_failed"
    assert context.attempt_count == 1
    assert context.error and "after 1 attempts" in context.error


def test_transient_failure_still_uses_configured_retry_budget() -> None:
    calls = 0

    def transient_loader(source: dict, *, config: dict) -> MethodContext:
        nonlocal calls
        calls += 1
        return MethodContext(
            str(source["candidate_id"]),
            "not_available",
            "https://repository.example/paper",
            "text/html",
            [],
            "",
            "temporary network failure",
        )

    context = routed_bounded_collect_method_context(
        {"candidate_id": "paper-transient", "doi": "10.1234/example"},
        config={"retrieval_attempts": 3, "candidate_timeout_seconds": -1},
        loader=transient_loader,
    )

    assert calls == 3
    assert context.attempt_count == 3
    assert context.error and "after 3 attempts" in context.error

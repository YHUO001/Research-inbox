from __future__ import annotations

from scripts.summarize.fulltext_methods import (
    MethodContext,
    candidate_urls,
    collect_method_context,
    extract_html_method_context,
)


def test_candidate_urls_prefers_open_access_and_deduplicates() -> None:
    source = {
        "open_access_url": "https://example.org/open",
        "landing_page": "https://doi.org/10.1000/example",
        "doi": "10.1000/example",
    }
    assert candidate_urls(source) == [
        "https://example.org/open",
        "https://doi.org/10.1000/example",
    ]


def test_html_extractor_keeps_method_sections_and_discovers_pdf() -> None:
    html = b"""
    <html><head><meta name='citation_pdf_url' content='/paper.pdf'></head>
    <body><article>
      <h2>Introduction</h2><p>This paragraph is not method evidence.</p>
      <h2>Methods</h2>
      <p>The input is encoded on a spatial light modulator before optical propagation.</p>
      <p>The detector measures the output and an electronic layer performs the final readout.</p>
      <h2>Results</h2><p>This paragraph should not be included.</p>
    </article></body></html>
    """
    text, headings, pdf_urls = extract_html_method_context(
        html,
        source_url="https://example.org/article",
        maximum_characters=2000,
    )
    assert headings == ["Methods"]
    assert "spatial light modulator" in text
    assert "final readout" in text
    assert "should not be included" not in text
    assert pdf_urls == ["https://example.org/paper.pdf"]


def test_collection_records_audit_without_persisting_text() -> None:
    html = b"""
    <html><body><h2>Experimental setup</h2>
    <p>The optical signal passes through the programmable device and is measured at the output.</p>
    </body></html>
    """

    def fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int):
        return html, "text/html", url

    context = collect_method_context(
        {
            "candidate_id": "candidate-1",
            "open_access_url": "https://example.org/article",
        },
        config={"maximum_method_characters": 1000},
        fetcher=fetcher,
    )
    assert isinstance(context, MethodContext)
    assert context.status == "used"
    assert context.section_headings == ["Experimental setup"]
    audit = context.audit_record()
    assert audit["text_persisted"] is False
    assert audit["character_count"] > 0
    assert audit["content_sha256"]
    assert "text" not in audit


def test_collection_falls_back_without_raising_when_no_context() -> None:
    def fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int):
        return b"<html><body><h2>Introduction</h2><p>Only background text is available.</p></body></html>", "text/html", url

    context = collect_method_context(
        {
            "candidate_id": "candidate-2",
            "landing_page": "https://example.org/article",
        },
        config={},
        fetcher=fetcher,
    )
    assert context.status == "not_available"
    assert context.text == ""
    assert context.error

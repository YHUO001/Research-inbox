from __future__ import annotations

from scripts.summarize.fulltext_methods import (
    MethodContext,
    candidate_urls,
    collect_method_context,
    derived_nature_pdf_urls,
    discover_pdf_urls,
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


def test_nature_pdf_urls_include_regular_and_article_in_press_forms() -> None:
    article = "https://www.nature.com/articles/s41467-026-76128-9?error=cookies"
    assert derived_nature_pdf_urls(article) == [
        "https://www.nature.com/articles/s41467-026-76128-9.pdf",
        "https://www.nature.com/articles/s41467-026-76128-9_reference.pdf",
    ]
    html = b"<html><body><a href='/download'>Download PDF file</a></body></html>"
    from bs4 import BeautifulSoup

    urls = discover_pdf_urls(BeautifulSoup(html, "html.parser"), article)
    assert "https://www.nature.com/download" in urls
    assert "https://www.nature.com/articles/s41467-026-76128-9_reference.pdf" in urls


def test_html_text_fallback_finds_nature_like_method_headings() -> None:
    html = b"""
    <html><body><main>
      <div>Introduction</div>
      <p>Background text does not describe implementation.</p>
      <div>All-optical computer architecture</div>
      <p>Input pulse amplitudes enter a recurrent optical cavity with interferometric linear weights.</p>
      <p>A nonlinear optical waveguide applies the activation before the optical output is read.</p>
      <div>Experimental setup</div>
      <p>The temporal delay is tuned to determine the recurrent connection topology.</p>
    </main></body></html>
    """
    text, headings, _ = extract_html_method_context(
        html,
        source_url="https://www.nature.com/articles/s41377-026-02314-5",
        maximum_characters=2000,
    )
    assert "All-optical computer architecture" in headings
    assert "Experimental setup" in headings
    assert "recurrent optical cavity" in text
    assert "connection topology" in text


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


def test_collection_uses_nature_reference_pdf_fallback() -> None:
    article_url = "https://www.nature.com/articles/s41467-026-76128-9"
    attempted: list[str] = []
    no_methods = b"<html><body><h2>Abstract</h2><p>Only an abstract is available.</p></body></html>"
    method_html = b"""
    <html><body><h2>Experimental implementation</h2>
    <p>The optical carrier is divided into parallel branches and delayed before entering weighted interferometers.</p>
    <p>The weighted optical signals are coherently combined and detected to obtain the matrix-computing output.</p>
    </body></html>
    """

    def fetcher(url: str, *, timeout_seconds: float, maximum_bytes: int):
        attempted.append(url)
        if url.endswith("_reference.pdf"):
            return method_html, "text/html", url
        return no_methods, "text/html", article_url

    context = collect_method_context(
        {
            "candidate_id": "candidate-nature",
            "open_access_url": article_url,
        },
        config={"maximum_method_characters": 2000},
        fetcher=fetcher,
    )
    assert context.status == "used"
    assert context.source_url and context.source_url.endswith("_reference.pdf")
    assert "parallel branches" in context.text
    assert f"{article_url}.pdf" in attempted
    assert f"{article_url}_reference.pdf" in attempted
    assert context.audit_record()["text_persisted"] is False


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

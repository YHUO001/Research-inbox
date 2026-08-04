from pathlib import Path

from scripts.ingest.scholar_parser import SourceContext, parse_alert_body


def context() -> SourceContext:
    return SourceContext(
        message_id="production-message",
        received_at="2026-08-03T18:38:08Z",
        sender="scholaralerts-noreply@google.com",
        subject='"optical neural network" - 新的结果',
        spf="pass",
        dkim="pass",
    )


def test_production_table_html() -> None:
    fixture = Path(__file__).parent / "fixtures" / "scholar_production_table.html"
    records = parse_alert_body(
        fixture.read_text(encoding="utf-8"),
        context(),
        content_type="html",
        extracted_at="2026-08-04T00:00:00Z",
    )
    assert len(records) == 3

    first = records[0]
    assert first["venue"]["normalized"] == "Optical Engineering"
    assert first["year"] == 2026
    assert len(first["authors"]) == 8
    assert first["parse_status"]["state"] == "complete"

    second = records[1]
    assert second["title"].startswith("65 TOPS")
    assert "[PDF]" not in (second["snippet"] or "")
    assert second["venue"]["normalized"] == "Nature Communications"
    assert second["parse_status"]["state"] == "complete"

    third = records[2]
    assert third["parse_status"]["state"] == "manual_review"
    assert "possible_document_cover_title" in third["parse_status"]["warnings"]


def test_plain_text_resource_prefix_is_removed() -> None:
    body = """
[PDF] [A paper title](https://scholar.google.com/scholar_url?url=https%3A%2F%2Fexample.org%2Fpaper.pdf)

A Author - Optica, 2026

A concise abstract fragment.
"""
    records = parse_alert_body(
        body,
        context(),
        content_type="text",
        extracted_at="2026-08-04T00:00:00Z",
    )
    assert len(records) == 1
    assert records[0]["title"] == "A paper title"
    assert records[0]["venue"]["normalized"] == "Optica"

from __future__ import annotations

from scripts.enrich.metadata import crossref_date


def test_crossref_date_keeps_year_when_month_and_day_are_null() -> None:
    assert crossref_date(
        {"published-print": {"date-parts": [[2026, None, None]]}}
    ) == ("2026", 2026)


def test_crossref_date_keeps_month_when_day_is_null() -> None:
    assert crossref_date(
        {"published-online": {"date-parts": [["2026", "8", None]]}}
    ) == ("2026-08", 2026)


def test_crossref_date_skips_source_without_a_valid_year() -> None:
    assert crossref_date(
        {
            "published-print": {"date-parts": [[None, None, None]]},
            "issued": {"date-parts": [[2025, 12, 31]]},
        }
    ) == ("2025-12-31", 2025)


def test_crossref_date_degrades_invalid_calendar_day_to_month() -> None:
    assert crossref_date(
        {"published": {"date-parts": [[2026, 2, 31]]}}
    ) == ("2026-02", 2026)


def test_crossref_date_returns_unresolved_for_invalid_sources() -> None:
    assert crossref_date(
        {"issued": {"date-parts": [["not-a-year", 1, 1]]}}
    ) == (None, None)

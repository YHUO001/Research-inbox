from scripts.ingest.scholar_parser import PARSER_VERSION, parse_metadata_line


def test_leading_year_in_conference_name_is_preserved() -> None:
    authors, venue, year = parse_metadata_line(
        "VA Puligandla, N Popov, T Knežević - "
        "2026 49th MIPRO ICT and Electronics …, 2026"
    )
    assert PARSER_VERSION == 3
    assert len(authors) == 3
    assert venue == "2026 49th MIPRO ICT and Electronics"
    assert year == 2026


def test_duplicate_event_year_at_venue_end_is_removed() -> None:
    authors, venue, year = parse_metadata_line(
        "L Rantz, A Maeder - Photonics for Quantum 2026, 2026"
    )
    assert len(authors) == 2
    assert venue == "Photonics for Quantum"
    assert year == 2026

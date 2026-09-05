from scripts.summarize.author_context import attach_author_context


def test_attach_author_context_preserves_summary_and_adds_metadata():
    result = attach_author_context(
        {"candidate_id": "paper-1", "core_problem": "example"},
        {
            "authors": [{"name": "Alice", "institutions": ["Example Lab"]}],
            "corresponding_authors": [],
        },
    )

    assert result["candidate_id"] == "paper-1"
    assert result["author_context"]["first_author"]["name"] == "Alice"
    assert result["author_context"]["source"] == "metadata_only"

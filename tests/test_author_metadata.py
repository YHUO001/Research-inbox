from scripts.enrich.author_metadata import build_author_context


def test_build_author_context_keeps_metadata_roles():
    context = build_author_context(
        {
            "authors": [
                {
                    "name": "Alice Zhang",
                    "institutions": ["Example University"],
                    "orcid": "0000-0000-0000-0000",
                }
            ],
            "corresponding_authors": [
                {
                    "name": "Prof. Bob Li",
                    "institutions": ["Example Institute"],
                }
            ],
        }
    )

    assert context["first_author"]["name"] == "Alice Zhang"
    assert context["first_author"]["institutions"] == ["Example University"]
    assert context["corresponding_authors"][0]["name"] == "Prof. Bob Li"
    assert context["source"] == "metadata_only"


def test_build_author_context_does_not_guess_missing_roles():
    context = build_author_context({"authors": []})
    assert context["first_author"] is None
    assert context["corresponding_authors"] == []

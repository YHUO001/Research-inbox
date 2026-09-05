from scripts.summarize.author_context_renderer import render_author_context


def test_render_author_context_displays_metadata_only():
    result = render_author_context(
        {
            "first_author": {
                "name": "Alice Zhang",
                "institutions": ["Example University"],
            },
            "corresponding_authors": [
                {
                    "name": "Bob Li",
                    "institutions": ["Example Institute"],
                }
            ],
        }
    )

    assert "第一作者：Alice Zhang" in result
    assert "Example University" in result
    assert "Bob Li" in result


def test_render_author_context_handles_missing_metadata():
    assert render_author_context(None) == ""

from scripts.enrich.selected_author_metadata import (
    context_from_crossref_message,
    context_from_openalex_work,
)
from scripts.summarize.add_digest_doi_links import add_author_sections_to_digest


def test_openalex_context_preserves_affiliations_and_explicit_corresponding_role():
    context = context_from_openalex_work(
        {
            "authorships": [
                {
                    "author_position": "first",
                    "is_corresponding": False,
                    "author": {
                        "display_name": "Alice Zhang",
                        "id": "https://openalex.org/A1",
                        "orcid": "https://orcid.org/0000-0001",
                    },
                    "institutions": [{"display_name": "Example University"}],
                },
                {
                    "author_position": "last",
                    "is_corresponding": True,
                    "author": {
                        "display_name": "Bob Li",
                        "id": "https://openalex.org/A2",
                        "orcid": None,
                    },
                    "institutions": [{"display_name": "Example Institute"}],
                },
            ]
        }
    )

    assert context["first_author"]["name"] == "Alice Zhang"
    assert context["corresponding_authors"][0]["name"] == "Bob Li"
    assert context["institutions"] == ["Example University", "Example Institute"]
    assert context["team_scope"] == "multi_institution"


def test_crossref_context_keeps_affiliations_without_guessing_corresponding_author():
    context = context_from_crossref_message(
        {
            "author": [
                {
                    "given": "Alice",
                    "family": "Zhang",
                    "affiliation": [{"name": "Example University"}],
                },
                {
                    "given": "Bob",
                    "family": "Li",
                    "affiliation": [{"name": "Example Institute"}],
                },
            ]
        }
    )

    assert context["first_author"]["name"] == "Alice Zhang"
    assert context["corresponding_authors"] == []
    assert context["corresponding_author_status"] == "not_explicitly_available"


def test_final_digest_inserts_author_team_section_before_core_problem():
    markdown = "\n".join(
        [
            "# Research Inbox — 2026-10-11",
            "",
            "## 1. Example paper",
            "",
            "- DOI：10.1000/example",
            "- 方法证据：标题、元数据和摘要",
            "",
            "### 核心问题",
            "",
            "示例核心问题。",
        ]
    )
    requests = [
        {
            "candidate_id": "candidate-1",
            "source": {
                "author_context": {
                    "authors": [
                        {
                            "name": "Alice Zhang",
                            "institutions": ["Example University"],
                        },
                        {
                            "name": "Bob Li",
                            "institutions": ["Example Institute"],
                        },
                    ],
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
                    "institutions": ["Example University", "Example Institute"],
                }
            },
        }
    ]

    output, count = add_author_sections_to_digest(markdown, requests)
    assert count == 1
    assert "### 作者与团队" in output
    assert "通讯作者：Bob Li（Example Institute）" in output
    assert "跨机构合作" in output
    assert output.index("### 作者与团队") < output.index("### 核心问题")

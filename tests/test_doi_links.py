from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize.add_doi_links import (
    add_links_to_digest,
    add_links_to_review,
    apply_doi_links,
    doi_markdown,
    normalize_doi,
)


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_doi_accepts_identifiers_and_resolver_urls() -> None:
    assert normalize_doi("10.1038/s41467-026-76128-9") == "10.1038/s41467-026-76128-9"
    assert normalize_doi("doi: 10.1038/s41377-026-02314-5") == "10.1038/s41377-026-02314-5"
    assert normalize_doi("https://doi.org/10.1000/example%281%29") == "10.1000/example(1)"
    assert normalize_doi("https://example.org/10.1000/example") is None
    assert normalize_doi("") is None
    assert doi_markdown(None) == "未提供"
    assert doi_markdown("10.1000/example(1)") == (
        "[10.1000/example(1)](https://doi.org/10.1000/example%281%29)"
    )


def test_markdown_helpers_add_clickable_doi_and_keep_original_link() -> None:
    requests = [
        {
            "source": {
                "doi": "10.1038/s41467-026-76128-9",
                "landing_page": "https://www.nature.com/articles/s41467-026-76128-9",
            }
        }
    ]
    digest, digest_count = add_links_to_digest(
        "- 链接：https://www.nature.com/articles/s41467-026-76128-9\n", requests
    )
    review, review_count = add_links_to_review(
        "- DOI：10.1038/s41467-026-76128-9\n", requests
    )
    link = "[10.1038/s41467-026-76128-9](https://doi.org/10.1038/s41467-026-76128-9)"
    assert digest_count == 1
    assert review_count == 1
    assert f"- DOI：{link}" in digest
    assert "- 原始链接：https://www.nature.com/articles/s41467-026-76128-9" in digest
    assert f"- DOI：{link}" in review


def test_apply_doi_links_updates_both_user_facing_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "data"
    request_path = output_root / "summary_requests/2026-08-05.jsonl"
    digest_path = output_root / "digests/2026-08-05.generated.md"
    review_path = output_root / "reviews/2026-08-05.review.md"
    request_path.parent.mkdir(parents=True)
    digest_path.parent.mkdir(parents=True)
    review_path.parent.mkdir(parents=True)
    request = {
        "candidate_id": "paper-1",
        "source": {
            "doi": "doi: 10.1038/s41377-026-02394-3",
            "landing_page": "https://doi.org/10.1038/s41377-026-02394-3",
        },
    }
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    digest_path.write_text(
        "# 摘要\n\n- 链接：https://doi.org/10.1038/s41377-026-02394-3\n",
        encoding="utf-8",
    )
    review_path.write_text(
        "# 评审\n\n- DOI：doi: 10.1038/s41377-026-02394-3\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "state/summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"digest_date": "2026-08-05"}), encoding="utf-8")

    state = apply_doi_links(
        generation_manifest_path=manifest_path,
        output_root=output_root,
    )

    expected = (
        "[10.1038/s41377-026-02394-3]"
        "(https://doi.org/10.1038/s41377-026-02394-3)"
    )
    assert state["digest_doi_links"] == 1
    assert state["review_doi_links"] == 1
    assert state["model_generated_doi"] is False
    assert expected in digest_path.read_text(encoding="utf-8")
    assert expected in review_path.read_text(encoding="utf-8")


def test_automatic_generation_applies_digest_doi_links_without_review() -> None:
    generation = (
        ROOT / ".github/workflows/generate-deepseek-summaries.yml"
    ).read_text(encoding="utf-8")
    assert "scripts.summarize.add_digest_doi_links" in generation
    assert generation.index("scripts.summarize.add_digest_doi_links") < generation.index(
        "scripts.summarize.finalize_automatic"
    )
    assert "scripts.summarize.add_doi_links" not in generation
    assert not (
        ROOT / ".github/workflows/prepare-human-summary-review.yml"
    ).exists()

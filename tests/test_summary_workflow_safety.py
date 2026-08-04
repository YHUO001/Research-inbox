from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_summary_dry_run_remains_manual_and_provider_free() -> None:
    text = workflow("build-summary-dry-run.yml")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "GMAIL_CLIENT_SECRET" not in text
    assert "summary_history.json" not in text


def test_deepseek_generation_builds_review_packet_but_cannot_finalize() -> None:
    text = workflow("generate-deepseek-summaries.yml")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}" in text
    assert "OPENAI_API_KEY" not in text
    assert "GMAIL_CLIENT_SECRET" not in text
    assert "gmail_sender" not in text
    assert "summary_history.json" not in text
    assert "scripts.summarize.generate_summaries_production" in text
    assert "scripts.summarize.build_review_packet" in text
    assert "data/reviews" in text
    assert "state/summary_review_manifest.json" in text
    assert "Fail after persisting validation diagnostics" in text


def test_offline_review_preparation_uses_no_provider_secret() -> None:
    text = workflow("prepare-human-summary-review.yml")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "scripts.summarize.build_review_packet" in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "GMAIL_CLIENT_SECRET" not in text
    assert "summary_history.json" not in text


def test_review_finalization_is_manual_and_has_explicit_confirmation() -> None:
    text = workflow("finalize-reviewed-summaries.yml")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "approve_all" in text
    assert "hold_for_revision" in text
    assert "Type REVIEWED exactly" in text
    assert "scripts.summarize.finalize_review" in text
    assert "state/summary_history.json" in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "GMAIL_CLIENT_SECRET" not in text
    assert "gmail_sender" not in text


def test_summary_configuration_requires_chinese_open_fulltext_human_review() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "summary_generation.yaml").read_text(encoding="utf-8")
    )
    execution = config["execution"]
    provider = config["provider"]
    review = config["review"]
    output = config["output"]
    full_text = config["full_text"]
    assert config["summary_generation_version"] == 5
    assert config["prompt_version"] == 2
    assert execution["mode"] == "manual_provider_validation"
    assert execution["provider"] == "deepseek"
    assert execution["llm_enabled"] is False
    assert execution["manual_provider_calls_allowed"] is True
    assert execution["email_enabled"] is False
    assert execution["update_summary_history"] is False
    assert execution["use_full_text"] is True
    assert execution["full_text_open_access_only"] is True
    assert review["required"] is True
    assert review["approval_mode"] == "manual_all_or_nothing"
    assert review["confirmation_phrase"] == "REVIEWED"
    assert review["email_after_approval"] is False
    assert provider["model"] == "deepseek-v4-pro"
    assert provider["thinking_enabled"] is False
    assert provider["response_format"] == "json_object"
    assert provider["pricing"]["input_cache_hit_cny_per_million"] == 0.025
    assert provider["pricing"]["input_cache_miss_cny_per_million"] == 3.0
    assert provider["pricing"]["output_cny_per_million"] == 6.0
    assert output["language"] == "zh-CN"
    assert output["require_chinese_narrative"] is True
    assert output["method_implementation_min_paragraphs"] == 2
    assert full_text["enabled"] is True
    assert full_text["open_access_only"] is True
    assert full_text["persist_extracted_text"] is False
    assert full_text["qualitative_methods_only"] is True
    assert full_text["numeric_grounding_scope"] == "title_and_abstract_only"
    assert config["limits"]["maximum_summaries_per_run"] == 3
    assert config["grounding"]["enforce_onn_architecture_evidence"] is True

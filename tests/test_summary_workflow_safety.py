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
    assert "SPRINGER_NATURE_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "GMAIL_CLIENT_SECRET" not in text
    assert "summary_history.json" not in text


def test_deepseek_generation_is_automatic_and_transactional() -> None:
    text = workflow("generate-deepseek-summaries.yml")
    assert "workflow_dispatch:" in text
    assert "workflow_run:" in text
    assert '"Daily Research Inbox"' in text
    assert '"OpenAlex Research Discovery"' in text
    assert "schedule:" not in text
    assert "timeout-minutes: 30" in text
    assert "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}" in text
    assert "SPRINGER_NATURE_API_KEY: ${{ secrets.SPRINGER_NATURE_API_KEY }}" in text
    assert "OPENAI_API_KEY" not in text
    assert "GMAIL_CLIENT_SECRET" not in text
    assert "gmail_sender" not in text
    assert "scripts.summarize.prepare_automatic_digest" in text
    assert "scripts.summarize.generate_automatic_summaries" in text
    assert "scripts.summarize.add_digest_doi_links" in text
    assert "scripts.summarize.finalize_automatic" in text
    assert "state/summary_history.json" in text
    assert "scripts.summarize.build_review_packet" not in text
    assert "data/reviews" not in text
    assert "state/summary_review_manifest.json" not in text
    assert "Fail after persisting validation diagnostics" in text


def test_human_review_workflows_are_removed() -> None:
    workflow_root = ROOT / ".github" / "workflows"
    assert not (workflow_root / "prepare-human-summary-review.yml").exists()
    assert not (workflow_root / "finalize-reviewed-summaries.yml").exists()


def test_summary_configuration_enables_validated_automation() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "summary_generation.yaml").read_text(encoding="utf-8")
    )
    execution = config["execution"]
    automation = config["automation"]
    provider = config["provider"]
    review = config["review"]
    output = config["output"]
    full_text = config["full_text"]
    assert config["summary_generation_version"] == 7
    assert config["prompt_version"] == 2

    # The reusable core remains isolated and cannot write history by itself.
    assert execution["mode"] == "manual_provider_validation"
    assert execution["provider"] == "deepseek"
    assert execution["llm_enabled"] is False
    assert execution["manual_provider_calls_allowed"] is True
    assert execution["email_enabled"] is False
    assert execution["update_summary_history"] is False
    assert execution["use_full_text"] is True
    assert execution["full_text_open_access_only"] is True

    # The production workflow orchestration is fully automatic.
    assert automation["enabled"] is True
    assert automation["mode"] == "automatic_after_discovery"
    assert automation["trigger_workflows"] == [
        "Daily Research Inbox",
        "OpenAlex Research Discovery",
    ]
    assert automation["filter_completed_before_provider"] is True
    assert automation["update_summary_history_after_validation"] is True
    assert automation["all_or_nothing_batch"] is True
    assert automation["review_required"] is False
    assert automation["email_enabled"] is False

    assert review["required"] is False
    assert review["approval_mode"] == "disabled"
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
    assert full_text["source_strategy"] == "springer_openaccess_jats_then_non_springer_open_url"
    assert full_text["springer_api_key_env"] == "SPRINGER_NATURE_API_KEY"
    assert full_text["springer_openaccess_endpoint"] == "https://api.springernature.com/openaccess/jats"
    assert full_text["allow_direct_nature_urls"] is False
    assert full_text["allow_non_springer_open_urls"] is True
    assert full_text["persist_extracted_text"] is False
    assert full_text["qualitative_methods_only"] is True
    assert full_text["numeric_grounding_scope"] == "title_and_abstract_only"
    assert full_text["timeout_seconds"] == 10
    assert full_text["candidate_timeout_seconds"] == 30
    assert config["limits"]["maximum_summaries_per_run"] == 3
    assert config["grounding"]["enforce_onn_architecture_evidence"] is True

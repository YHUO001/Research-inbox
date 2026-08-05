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
    assert "GMAIL_CLIENT_SECRET" not in text
    assert "summary_history.json" not in text


def test_daily_pipeline_unifies_discovery_and_cross_source_reconciliation() -> None:
    text = workflow("daily-research-inbox.yml")
    assert "scripts.ingest.gmail_collector" in text
    assert "scripts.discovery.openalex_discovery" in text
    assert "scripts.pipeline.daily_source_gate" in text
    assert "scripts.pipeline.reconcile_registry" in text
    assert "data/unified_paper_registry.jsonl" in text
    assert "scripts.pipeline.route_registry" in text
    assert "scripts.enrich.enrich_registry" in text
    assert "scripts.pipeline.score_registry" in text
    assert "state/unified_registry_manifest.json" in text
    assert "20:47 retries only missing sources" in text
    assert not (
        ROOT / ".github" / "workflows" / "openalex-research-discovery.yml"
    ).exists()


def test_deepseek_generation_is_in_the_one_daily_transaction() -> None:
    text = workflow("daily-research-inbox.yml")
    assert 'cron: "17 0 * * *"' in text
    assert 'cron: "47 12 * * *"' in text
    assert "timeout-minutes: 60" in text
    assert "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}" in text
    assert "SPRINGER_NATURE_API_KEY: ${{ secrets.SPRINGER_NATURE_API_KEY }}" in text
    assert "scripts.summarize.prepare_automatic_digest" in text
    assert "scripts.summarize.generate_automatic_summaries" in text
    assert "scripts.summarize.add_digest_doi_links" in text
    assert "scripts.summarize.finalize_automatic" in text
    assert "scripts.knowledge.build_index" in text
    assert "state/summary_history.json" in text
    assert "state/knowledge_base_manifest.json" in text
    assert "data/knowledge_base" in text
    assert "scripts.summarize.build_review_packet" not in text
    assert "data/reviews" not in text
    assert "state/summary_review_manifest.json" not in text
    assert "Fail after persisting summary validation diagnostics" in text

    manual = workflow("generate-deepseek-summaries.yml")
    assert "workflow_dispatch:" in manual
    assert "schedule:" not in manual


def test_daily_email_is_idempotent_and_part_of_unified_transaction() -> None:
    text = workflow("daily-research-inbox.yml")
    assert "scripts.delivery.send_daily_digest" in text
    assert "config/email_delivery.yaml" in text
    assert "GMAIL_CLIENT_ID: ${{ secrets.GMAIL_CLIENT_ID }}" in text
    assert "GMAIL_CLIENT_SECRET: ${{ secrets.GMAIL_CLIENT_SECRET }}" in text
    assert "GMAIL_REFRESH_TOKEN: ${{ secrets.GMAIL_REFRESH_TOKEN }}" in text
    assert "state/email_delivery_state.json" not in text
    assert "Fail after persisting email delivery diagnostics" in text

    manual = workflow("send-daily-research-digest.yml")
    assert "workflow_dispatch:" in manual
    assert "schedule:" not in manual
    assert "state/email_delivery_state.json" in manual
    assert "DEEPSEEK_API_KEY" not in manual
    assert "SPRINGER_NATURE_API_KEY" not in manual


def test_human_review_workflows_are_removed() -> None:
    workflow_root = ROOT / ".github" / "workflows"
    assert not (workflow_root / "prepare-human-summary-review.yml").exists()
    assert not (workflow_root / "finalize-reviewed-summaries.yml").exists()


def test_summary_configuration_enables_daily_automation_and_indexing() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "summary_generation.yaml").read_text(encoding="utf-8")
    )
    execution = config["execution"]
    automation = config["automation"]
    knowledge = config["knowledge_base"]
    delivery = config["delivery"]
    review = config["review"]
    provider = config["provider"]
    full_text = config["full_text"]

    assert config["summary_generation_version"] == 8
    assert execution["mode"] == "manual_provider_validation"
    assert execution["llm_enabled"] is False
    assert execution["email_enabled"] is False
    assert execution["update_summary_history"] is False

    assert automation["enabled"] is True
    assert automation["mode"] == "automatic_daily_batch"
    assert automation["timezone"] == "Asia/Singapore"
    assert automation["local_time"] == "08:17"
    assert automation["retry_local_time"] == "20:47"
    assert automation["filter_completed_before_provider"] is True
    assert automation["update_summary_history_after_validation"] is True
    assert automation["all_or_nothing_batch"] is True
    assert automation["review_required"] is False
    assert automation["delivery_mode"] == "separate_daily_digest_workflow"

    assert review == {"required": False, "approval_mode": "disabled"}
    assert knowledge["enabled"] is True
    assert knowledge["update_after_automatic_completion"] is True
    assert knowledge["persist_full_text"] is False
    assert delivery["daily_digest_enabled"] is True
    assert delivery["mode"] == "end_of_unified_daily_pipeline"
    assert delivery["send_only_completed_automatic"] is True
    assert delivery["send_empty_digest"] is True

    assert provider["model"] == "deepseek-v4-pro"
    assert provider["thinking_enabled"] is False
    assert full_text["enabled"] is True
    assert full_text["open_access_only"] is True
    assert full_text["persist_extracted_text"] is False
    assert full_text["candidate_timeout_seconds"] == 30
    assert config["limits"]["maximum_summaries_per_run"] == 3
    assert config["grounding"]["numeric_matching_mode"] == "loose_full_evidence"

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_summary_dry_run_remains_manual_and_provider_free() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "build-summary-dry-run.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "DEEPSEEK_API_KEY" not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "GMAIL_CLIENT_SECRET" not in workflow
    assert "gmail_sender" not in workflow
    assert "summary_history.json" not in workflow


def test_deepseek_generation_is_manual_and_cannot_send_email() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "generate-deepseek-summaries.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}" in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "GMAIL_CLIENT_SECRET" not in workflow
    assert "gmail_sender" not in workflow
    assert "summary_history.json" not in workflow
    assert "data/summaries" in workflow
    assert "scripts.summarize.generate_summaries_production" in workflow
    assert "cat runtime-state/state/summary_generation_manifest.json" in workflow
    assert "mkdir -p data/summary_requests data/summaries data/digests" in workflow
    assert "git add -A --" in workflow
    assert "Fail after persisting validation diagnostics" in workflow


def test_summary_configuration_uses_manual_deepseek_pro_validation() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "summary_generation.yaml").read_text(encoding="utf-8")
    )
    execution = config["execution"]
    provider = config["provider"]
    assert execution["mode"] == "manual_provider_validation"
    assert execution["provider"] == "deepseek"
    assert execution["llm_enabled"] is False
    assert execution["manual_provider_calls_allowed"] is True
    assert execution["email_enabled"] is False
    assert execution["update_summary_history"] is False
    assert execution["use_full_text"] is False
    assert provider["model"] == "deepseek-v4-pro"
    assert provider["thinking_enabled"] is False
    assert provider["response_format"] == "json_object"
    assert provider["pricing"]["input_cache_hit_cny_per_million"] == 0.025
    assert provider["pricing"]["input_cache_miss_cny_per_million"] == 3.0
    assert provider["pricing"]["output_cny_per_million"] == 6.0
    assert config["limits"]["maximum_summaries_per_run"] == 3

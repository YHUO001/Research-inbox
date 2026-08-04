from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_summary_workflow_is_manual_and_has_no_provider_or_gmail_secrets() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "build-summary-dry-run.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "ANTHROPIC_API_KEY" not in workflow
    assert "GMAIL_CLIENT_SECRET" not in workflow
    assert "gmail_sender" not in workflow
    assert "summary_history.json" not in workflow


def test_summary_configuration_keeps_model_and_email_disabled() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "summary_generation.yaml").read_text(encoding="utf-8")
    )
    execution = config["execution"]
    assert execution["mode"] == "manual_dry_run"
    assert execution["provider"] == "not_configured"
    assert execution["llm_enabled"] is False
    assert execution["email_enabled"] is False
    assert execution["update_summary_history"] is False
    assert execution["use_full_text"] is False
    assert config["limits"]["maximum_summaries_per_run"] == 3

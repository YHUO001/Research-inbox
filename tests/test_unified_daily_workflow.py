from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_workflow(name: str) -> dict:
    value = yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def workflow_text(name: str) -> str:
    return (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")


def test_daily_workflow_contains_one_end_to_end_pipeline() -> None:
    text = workflow_text("daily-research-inbox.yml")
    for required in (
        "scripts.ingest.gmail_collector",
        "scripts.discovery.openalex_discovery",
        "scripts.pipeline.reconcile_registry",
        "unified_paper_registry.jsonl",
        "scripts.enrich.enrich_registry",
        "scripts.pipeline.route_registry",
        "scripts.pipeline.score_registry",
        "scripts.summarize.generate_automatic_summaries",
        "scripts.summarize.finalize_automatic",
        "scripts.knowledge.build_index",
        "scripts.delivery.send_daily_digest",
    ):
        assert required in text
    assert 'cron: "17 0 * * *"' in text
    assert 'cron: "47 12 * * *"' in text


def test_old_component_workflows_cannot_run_on_an_independent_schedule() -> None:
    assert not (ROOT / ".github/workflows/openalex-research-discovery.yml").exists()
    for name in (
        "generate-deepseek-summaries.yml",
        "send-daily-research-digest.yml",
    ):
        workflow = load_workflow(name)
        triggers = workflow.get("on") or workflow.get(True) or {}
        assert "workflow_dispatch" in triggers
        assert "schedule" not in triggers


def test_email_recipient_and_knowledge_paths_are_configured() -> None:
    delivery = yaml.safe_load((ROOT / "config/email_delivery.yaml").read_text(encoding="utf-8"))
    assert delivery["recipient_policy"]["recipients"] == ["a209072780@126.com"]
    assert delivery["schedule"]["send_empty_digest"] is True

    summary = yaml.safe_load((ROOT / "config/summary_generation.yaml").read_text(encoding="utf-8"))
    knowledge = summary["knowledge_base"]
    assert knowledge["enabled"] is True
    assert knowledge["persist_full_text"] is False
    assert knowledge["papers_file"] == "data/knowledge_base/papers.jsonl"
    assert knowledge["index_file"] == "data/knowledge_base/index.json"
    assert knowledge["markdown_file"] == "data/knowledge_base/index.md"

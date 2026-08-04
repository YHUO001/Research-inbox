from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"


SCORING_WORKFLOWS = (
    "daily-research-inbox.yml",
    "openalex-research-discovery.yml",
    "repair-scholar-registry.yml",
    "rebuild-research-routes.yml",
    "enrich-paper-registry.yml",
)


def workflow_text(name: str) -> str:
    return (WORKFLOW_DIR / name).read_text(encoding="utf-8")


def test_all_upstream_mutation_workflows_rebuild_scoring_outputs() -> None:
    for name in SCORING_WORKFLOWS:
        text = workflow_text(name)
        assert "scripts.pipeline.score_registry" in text, name
        assert "data/scoring_results.jsonl" in text, name
        assert "data/queues/llm_candidate_queue.jsonl" in text, name
        assert "state/selection_manifest.json" in text, name
        assert "state/summary_history.json" in text, name


def test_daily_workflow_refreshes_metadata_before_scoring() -> None:
    text = workflow_text("daily-research-inbox.yml")
    enrich_position = text.index("scripts.enrich.enrich_registry")
    scoring_position = text.index("scripts.pipeline.score_registry")
    assert enrich_position < scoring_position
    assert "OPENALEX_API_KEY" in text
    assert "steps.enrich.outputs.exit_code == '0'" in text


def test_llm_and_email_remain_disabled_in_pipeline_config() -> None:
    text = (ROOT / "config" / "pipeline.yaml").read_text(encoding="utf-8")
    assert "llm:\n  enabled: false" in text
    assert "email:\n    enabled: false" in text

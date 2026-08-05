from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"


SCORING_WORKFLOWS = (
    "daily-research-inbox.yml",
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


def test_daily_workflow_unifies_discovery_and_refreshes_metadata_before_scoring() -> None:
    text = workflow_text("daily-research-inbox.yml")
    scholar_position = text.index("scripts.ingest.gmail_collector")
    openalex_position = text.index("scripts.discovery.openalex_discovery")
    reconcile_position = text.index("scripts.pipeline.reconcile_registry")
    enrich_position = text.index("scripts.enrich.enrich_registry")
    route_position = text.index("scripts.pipeline.route_registry")
    scoring_position = text.index("scripts.pipeline.score_registry")

    assert scholar_position < reconcile_position
    assert openalex_position < reconcile_position
    assert reconcile_position < enrich_position
    assert enrich_position < route_position < scoring_position
    assert "OPENALEX_API_KEY" in text
    assert "steps.readiness.outputs.ready == 'true'" in text
    assert "state/unified_registry_manifest.json" in text
    assert "data/unified_paper_registry.jsonl" in text
    assert not (WORKFLOW_DIR / "openalex-research-discovery.yml").exists()


def test_llm_and_email_remain_disabled_in_base_pipeline_config() -> None:
    text = (ROOT / "config" / "pipeline.yaml").read_text(encoding="utf-8")
    assert "llm:\n  enabled: false" in text
    assert "email:\n    enabled: false" in text

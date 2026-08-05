from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.staged_summary_pipeline import (
    prepare_stage,
    shared_numeric_grounding,
)


ROOT = Path(__file__).resolve().parents[1]


def write_dry_run(tmp_path: Path) -> Path:
    request_path = tmp_path / "runtime-state/data/summary_requests/2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    request = {
        "candidate_id": "candidate-1",
        "prompt": "请生成中文摘要。",
        "source": {
            "title": "Optical computing towards 100-GHz operation",
            "abstract": "The system operates at ~5 GHz, tolerates alignment errors within ±3 pixels, and reaches 80 GHz for one task.",
            "doi": "10.1000/example",
            "landing_page": "https://doi.org/10.1000/example",
            "open_access_url": "https://example.org/article",
        },
    }
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "runtime-state/state/summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "digest_date": "2026-08-04",
                "request_file": str(request_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_prepare_stage_records_timing_without_persisting_text(tmp_path: Path) -> None:
    manifest_path = write_dry_run(tmp_path)

    def loader(source: dict, *, config: dict) -> MethodContext:
        return MethodContext(
            str(source["candidate_id"]),
            "used",
            "https://example.org/article.pdf",
            "application/pdf",
            ["Methods"],
            "The input is encoded optically and passed through the computing hardware.",
        )

    prepared_root = tmp_path / "runner-temp"
    audit = prepare_stage(
        dry_run_manifest_path=manifest_path,
        config_path=ROOT / "config/summary_generation.yaml",
        prepared_root=prepared_root,
        method_context_loader=loader,
    )
    context = audit["full_text_method_contexts"]["candidate-1"]
    assert audit["status"] == "prepared"
    assert audit["elapsed_seconds"] >= 0
    assert audit["full_text_used"] is True
    assert audit["full_text_persisted"] is False
    assert context["elapsed_seconds"] >= 0
    assert context["text_persisted"] is False
    assert "text" not in context
    assert Path(audit["prepared_generation_manifest_path"]).exists()
    assert (prepared_root / "fulltext_preparation.json").exists()


def test_shared_grounding_preserves_semantics_and_rejects_new_values() -> None:
    supported = {
        "core_problem": "标题指向 100 GHz 级目标。",
        "method_and_architecture": "作者讨论约 5 GHz 的基线。",
        "method_principle": "系统在不同任务中达到 80 GHz，并报告对正负 3 像素偏差的容忍范围。",
        "method_implementation": [],
        "main_contributions": [],
        "reported_results": [],
        "distinction_from_prior_work": "未提供",
        "research_value": "未提供",
        "limitations_and_open_questions": [],
        "optical_neural_network_analysis": None,
        "zeroth_order_analysis": None,
    }
    assert shared_numeric_grounding(
        supported,
        title="Optical computing towards 100-GHz operation",
        abstract="The baseline is ~5 GHz, tolerance is ±3 pixels, and one task reaches 80 GHz.",
    ) == []

    unsupported = dict(supported)
    unsupported["method_principle"] = "系统使用 488 nm 光源。"
    assert "488nm" in shared_numeric_grounding(
        unsupported,
        title="Optical computing towards 100-GHz operation",
        abstract="The baseline is ~5 GHz, tolerance is ±3 pixels, and one task reaches 80 GHz.",
    )


def test_workflow_splits_full_text_and_model_stages_with_30_minute_limit() -> None:
    workflow = (
        ROOT / ".github/workflows/generate-deepseek-summaries.yml"
    ).read_text(encoding="utf-8")
    assert "timeout-minutes: 30" in workflow
    assert "Prepare open full-text method context" in workflow
    assert "Generate and locally validate DeepSeek summaries" in workflow
    assert "prepare_fulltext_bounded" in workflow
    assert "staged_summary_pipeline generate" in workflow
    assert workflow.index("prepare_fulltext_bounded") < workflow.index(
        "staged_summary_pipeline generate"
    )

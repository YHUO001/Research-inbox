from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.knowledge.build_index import update_knowledge_base


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def summary(candidate_id: str) -> dict:
    return {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": candidate_id,
        "output_language": "zh-CN",
        "core_problem": "研究光学神经网络如何在真实硬件中完成稳定的并行计算。",
        "method_and_architecture": "系统使用自由空间传播、空间光调制器和探测器形成计算链路。",
        "method_principle": "输入首先被编码为空间光场，随后经过传播和调制完成并行变换，探测结果再用于任务输出和参数校准。",
        "method_implementation": ["输入被映射为空间图案并送入光学链路。", "探测结果进入电子控制环路并形成最终输出。"],
        "main_contributions": ["给出了真实硬件实现。"],
        "reported_results": [],
        "distinction_from_prior_work": "强调真实物理系统。",
        "research_value": "为后续硬件扩展提供依据。",
        "limitations_and_open_questions": ["长期稳定性仍需验证。"],
        "optical_neural_network_analysis": {
            "architecture_type": "free_space",
            "training_method": "通过测量进行校准。",
            "optical_nonlinearity": "未提供",
            "calibration_requirements": "需要校准。",
            "application_tasks": ["分类"],
            "hardware_validation": "physical_experiment",
        },
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_abstract_and_open_full_text_methods",
            "full_text_method_context_used": True,
            "full_text_method_source_url": "https://example.org/article.xml",
            "unsupported_numbers_detected": False,
            "missing_information": [],
        },
    }


def prepared_batch(tmp_path: Path) -> tuple[Path, Path]:
    candidate_id = "candidate-kb-001"
    data = tmp_path / "data"
    request_path = data / "summary_requests/2026-08-05.jsonl"
    summary_path = data / "summaries/2026-08-05.jsonl"
    manifest_path = tmp_path / "state/summary_generation_manifest.json"
    write_jsonl(
        request_path,
        [
            {
                "candidate_id": candidate_id,
                "request_id": "request-kb-001",
                "source": {
                    "title": "Free-space optical neural network",
                    "authors": ["Alice Researcher"],
                    "venue": "Optica",
                    "year": 2026,
                    "doi": "10.1000/kb.001",
                    "openalex_id": "https://openalex.org/W1",
                    "landing_page": "https://doi.org/10.1000/kb.001",
                    "source_type": "google_scholar_email",
                    "matched_projects": ["optical-neural-networks"],
                    "mandatory": False,
                    "score": 0.88,
                    "decision": "urgent",
                },
            }
        ],
    )
    write_jsonl(summary_path, [summary(candidate_id)])
    write_json(
        manifest_path,
        {
            "status": "completed_automatic",
            "summary_history_updated": True,
            "digest_date": "2026-08-05",
            "completed_at": "2026-08-05T13:35:00Z",
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "request_file": str(request_path),
            "summary_file": str(summary_path),
        },
    )
    return manifest_path, data


def test_builds_searchable_idempotent_knowledge_index(tmp_path: Path) -> None:
    manifest_path, data = prepared_batch(tmp_path)
    state_path = tmp_path / "state/knowledge_base_manifest.json"

    result = update_knowledge_base(
        generation_manifest_path=manifest_path,
        output_root=data,
        manifest_path=state_path,
    )
    assert result["status"] == "completed"
    assert result["paper_count"] == 1
    assert result["newly_indexed_count"] == 1
    assert result["full_text_persisted"] is False

    index = json.loads((data / "knowledge_base/index.json").read_text(encoding="utf-8"))
    assert index["by_doi"]["10.1000/kb.001"] == "candidate-kb-001"
    assert index["by_project"]["optical-neural-networks"] == ["candidate-kb-001"]
    assert index["by_year"]["2026"] == ["candidate-kb-001"]
    assert index["by_tag"]["architecture:free_space"] == ["candidate-kb-001"]
    assert index["by_tag"]["hardware_validation:physical_experiment"] == ["candidate-kb-001"]

    papers = [
        json.loads(line)
        for line in (data / "knowledge_base/papers.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert papers[0]["summary"]["core_problem"].startswith("研究光学神经网络")
    assert "full_text" not in papers[0]
    assert "10.1000/kb.001" in (data / "knowledge_base/index.md").read_text(encoding="utf-8")

    repeated = update_knowledge_base(
        generation_manifest_path=manifest_path,
        output_root=data,
        manifest_path=state_path,
    )
    assert repeated["newly_indexed_count"] == 0
    assert repeated["paper_count"] == 1


def test_rejects_silent_replacement_of_completed_summary(tmp_path: Path) -> None:
    manifest_path, data = prepared_batch(tmp_path)
    state_path = tmp_path / "state/knowledge_base_manifest.json"
    update_knowledge_base(
        generation_manifest_path=manifest_path,
        output_root=data,
        manifest_path=state_path,
    )

    summary_path = data / "summaries/2026-08-05.jsonl"
    changed = summary("candidate-kb-001")
    changed["core_problem"] += "内容被修改。"
    write_jsonl(summary_path, [changed])

    with pytest.raises(RuntimeError, match="Knowledge record changed"):
        update_knowledge_base(
            generation_manifest_path=manifest_path,
            output_root=data,
            manifest_path=state_path,
        )

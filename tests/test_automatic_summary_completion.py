from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize.finalize_automatic import finalize_automatic
from scripts.summarize.prepare_automatic_digest import prepare_automatic


ROOT = Path(__file__).resolve().parents[1]


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


def queue_candidate(candidate_id: str) -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "title": f"Optical neural network {candidate_id}",
        "authors": ["Alice Researcher"],
        "venue": "Optica",
        "year": 2026,
        "source_type": "google_scholar_email",
        "doi": f"10.1000/{candidate_id}",
        "openalex_id": f"https://openalex.org/{candidate_id}",
        "landing_page": f"https://doi.org/10.1000/{candidate_id}",
        "open_access_url": None,
        "abstract": "A free-space optical neural network uses a spatial light modulator and reports 95% accuracy.",
        "matched_projects": ["optical-neural-networks"],
        "mandatory": False,
        "score": 0.88,
        "decision": "urgent",
        "selection_status": "summary_slot",
        "score_breakdown": [],
    }


def valid_summary(candidate_id: str) -> dict:
    return {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": candidate_id,
        "output_language": "zh-CN",
        "core_problem": "论文试图解决光学神经网络在实际系统中如何完成稳定计算并保持任务性能的问题。",
        "method_and_architecture": "系统采用光学输入编码、核心光场变换和探测读出组成完整计算链路，并通过电子控制协调各模块。",
        "method_principle": "该方法首先把输入数据转换为光学系统能够处理的空间分布，然后利用传播、调制和干涉过程完成并行线性变换。输出光场经过探测后转换为可供任务判断的信号，训练或校准过程则根据测量结果调整可控参数，使物理系统的实际响应逐步接近期望映射。各组成部分分别承担编码、计算、读出和误差修正功能，因此能够在同一闭环中协同工作，同时保留对真实硬件偏差的适应能力。",
        "method_implementation": [
            "实施时先对输入样本进行预处理，并把数据映射到空间光调制器能够表示的光学图案。调制后的光束进入自由空间传播与计算模块，在物理传播过程中形成任务所需的中间表示，再由探测器采集输出强度。",
            "随后将探测结果送入电子处理与控制环路，根据任务目标更新可调参数或执行校准。系统重复完成输入编码、光学变换、探测读出和参数修正，最终以稳定的测量输出形成分类或推断结果。",
        ],
        "main_contributions": ["论文给出了能够在真实光学链路中执行神经网络计算的系统化实现方案。"],
        "reported_results": [
            {
                "claim": "作者报告该系统取得了百分之九十五的任务准确率。",
                "reported_by_authors": True,
                "basis": "abstract",
            }
        ],
        "distinction_from_prior_work": "该工作强调物理光学模块与电子控制闭环的联合实现，而不仅是理想化数值模拟。",
        "research_value": "该结果有助于评估光学神经网络在真实硬件条件下的可实现性、可校准性和后续扩展方向。",
        "limitations_and_open_questions": ["现有证据尚未充分说明长期漂移、跨设备复现和大规模部署时的稳定性。"],
        "optical_neural_network_analysis": {
            "architecture_type": "integrated",
            "training_method": "通过测量输出进行参数调整和系统校准。",
            "optical_nonlinearity": "未提供",
            "calibration_requirements": "需要根据真实探测结果修正物理系统偏差。",
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


def test_automatic_preparation_filters_completed_candidates(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    write_jsonl(
        queue_path,
        [queue_candidate("candidate-old"), queue_candidate("candidate-new")],
    )
    history_path = tmp_path / "state/summary_history.json"
    write_json(
        history_path,
        {
            "schema_version": 1,
            "completed_candidate_ids": {"candidate-old": {"completed_at": "earlier"}},
            "failed_candidate_ids": {},
        },
    )
    selection_path = tmp_path / "state/selection.json"
    write_json(selection_path, {"built_at": "2026-08-05T08:00:00Z"})

    manifest = prepare_automatic(
        queue_path=queue_path,
        history_path=history_path,
        selection_manifest_path=selection_path,
        config_path=ROOT / "config/summary_generation.yaml",
        request_schema_path=ROOT / "schemas/summary_request.schema.json",
        summary_schema_path=ROOT / "schemas/paper_summary.schema.json",
        output_root=tmp_path / "data",
        manifest_path=tmp_path / "state/summary_generation_manifest.json",
        filtered_queue_path=tmp_path / "filtered.jsonl",
    )

    assert manifest["status"] == "automatic_requests_prepared"
    assert manifest["request_count"] == 1
    assert manifest["completed_candidate_filtered_count"] == 1
    requests = [
        json.loads(line)
        for line in Path(manifest["request_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [item["candidate_id"] for item in requests] == ["candidate-new"]


def automatic_batch(tmp_path: Path) -> tuple[Path, Path]:
    candidate_id = "candidate-automatic"
    data = tmp_path / "data"
    state = tmp_path / "state"
    request_path = data / "summary_requests/2026-08-05.jsonl"
    summary_path = data / "summaries/2026-08-05.jsonl"
    digest_json_path = data / "digests/2026-08-05.generated.json"
    digest_markdown_path = data / "digests/2026-08-05.generated.md"
    history_path = state / "summary_history.json"
    manifest_path = state / "summary_generation_manifest.json"

    request = {
        "candidate_id": candidate_id,
        "request_id": "request-automatic",
        "source": {
            "abstract": "A free-space optical neural network uses a spatial light modulator and reports 95% accuracy."
        },
    }
    summary = valid_summary(candidate_id)
    write_jsonl(request_path, [request])
    write_jsonl(summary_path, [summary])
    write_json(
        digest_json_path,
        {
            "status": "generated_pending_human_review",
            "summaries": [summary],
            "safety": {"summary_history_updated": False},
        },
    )
    digest_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    digest_markdown_path.write_text("# Digest\n\n- DOI：[10.1000/test](https://doi.org/10.1000/test)\n", encoding="utf-8")
    write_json(
        manifest_path,
        {
            "status": "completed",
            "digest_date": "2026-08-05",
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "output_language": "zh-CN",
            "request_count": 1,
            "summary_count": 1,
            "failure_count": 0,
            "numeric_grounding_scope": "title_abstract_and_open_full_text_loose",
            "request_file": str(request_path),
            "summary_file": str(summary_path),
            "digest_json_file": str(digest_json_path),
            "digest_markdown_file": str(digest_markdown_path),
            "email_enabled": False,
            "summary_history_updated": False,
        },
    )
    return manifest_path, history_path


def test_automatic_finalization_updates_history_and_architecture(tmp_path: Path) -> None:
    manifest_path, history_path = automatic_batch(tmp_path)
    result = finalize_automatic(
        generation_manifest_path=manifest_path,
        history_path=history_path,
        summary_schema_path=ROOT / "schemas/paper_summary.schema.json",
        config_path=ROOT / "config/summary_generation.yaml",
        completed_at="2026-08-05T09:00:00Z",
    )

    assert result["status"] == "completed_automatic"
    assert result["summary_history_updated"] is True
    assert result["architecture_repairs"] == 1
    history = json.loads(history_path.read_text(encoding="utf-8"))
    entry = history["completed_candidate_ids"]["candidate-automatic"]
    assert entry["completion_mode"] == "automatic_after_local_validation"
    summary = json.loads(
        Path(result["summary_file"]).read_text(encoding="utf-8").strip()
    )
    assert summary["optical_neural_network_analysis"]["architecture_type"] == "free_space"
    markdown = Path(result["digest_markdown_file"]).read_text(encoding="utf-8")
    assert "自动处理状态" in markdown
    assert "人工评审：`不需要`" in markdown

    repeated = finalize_automatic(
        generation_manifest_path=manifest_path,
        history_path=history_path,
        summary_schema_path=ROOT / "schemas/paper_summary.schema.json",
        config_path=ROOT / "config/summary_generation.yaml",
        completed_at="2026-08-05T10:00:00Z",
    )
    assert repeated["completed_at"] == "2026-08-05T09:00:00Z"


def test_incomplete_batch_never_updates_history(tmp_path: Path) -> None:
    manifest_path, history_path = automatic_batch(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_count"] = 0
    write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="all-or-nothing"):
        finalize_automatic(
            generation_manifest_path=manifest_path,
            history_path=history_path,
            summary_schema_path=ROOT / "schemas/paper_summary.schema.json",
            config_path=ROOT / "config/summary_generation.yaml",
        )
    assert not history_path.exists()

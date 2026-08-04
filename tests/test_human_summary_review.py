from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize.build_review_packet import build_review_packet
from scripts.summarize.finalize_review import finalize_review


ROOT = Path(__file__).resolve().parents[1]


def source_request(candidate_id: str) -> dict:
    abstract = (
        "An all-optical recurrent neural network realizes linear operations, nonlinear "
        "functions, and memory in the optical domain up to 80 GHz. It analyzes soliton "
        "states from integrated optical microresonators."
    )
    return {
        "schema_version": 1,
        "request_version": 1,
        "request_id": "request-12345678",
        "candidate_id": candidate_id,
        "prepared_at": "2026-08-04T13:23:52Z",
        "provider_status": "not_configured",
        "selection_status": "summary_slot",
        "summary_schema": "paper_summary.schema.json",
        "source": {
            "title": "All-optical computing towards 100-GHz clock rates",
            "authors": ["A. Researcher"],
            "venue": "Light Science & Applications",
            "year": 2026,
            "source_type": "google_scholar_email",
            "doi": "10.1000/example",
            "openalex_id": None,
            "landing_page": "https://doi.org/10.1000/example",
            "open_access_url": None,
            "abstract": abstract,
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.85,
            "decision": "mandatory",
            "score_breakdown": [],
        },
        "instructions": ["只依据明确证据判断架构类型。"],
        "prompt": "返回一份中文、证据受限的 JSON 摘要。",
    }


def generated_summary(candidate_id: str) -> dict:
    return {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": candidate_id,
        "output_language": "zh-CN",
        "core_problem": "该工作针对传统电子处理器时钟频率难以继续提升的问题，探索面向超快信息处理的全光计算方案。",
        "method_and_architecture": "作者实验展示了一个全光循环神经网络，使线性变换、非线性函数和记忆过程都在光域内完成。",
        "method_principle": (
            "该方法利用光学器件的高速传播和并行处理能力，把循环神经网络需要的线性变换、"
            "非线性映射和状态记忆组织成连续的光学计算链路。输入波形进入系统后，光场先经历"
            "线性组合，再经过非线性光学过程形成新的状态，同时保留与前一时刻相关的光学记忆。"
            "这种反馈和状态更新使网络能够处理具有时间结构的输入，而不需要在每一步返回电子处理器。"
            "摘要没有给出具体器件拓扑和训练细节，因此不能进一步判断其物理架构。"
        ),
        "method_implementation": [
            "实施时先把待处理的时间波形输入全光循环网络，并利用光学线性运算形成中间表示。系统随后通过非线性光学功能对中间状态进行变换，同时将部分状态保留为后续时刻的记忆，从而构成循环更新过程。",
            "网络输出用于波形分类、微腔孤子状态分析以及基于量子涨落的图像生成。摘要报告系统在不同任务上可工作到 80 GHz，但没有提供训练数据、参数调节、校准方式和器件连接细节。",
        ],
        "main_contributions": ["作者报告全光循环处理在不同任务上可工作到 80 GHz。"],
        "reported_results": [
            {
                "claim": "作者报告该系统在不同任务上可工作到 80 GHz。",
                "reported_by_authors": True,
                "basis": "abstract",
            }
        ],
        "distinction_from_prior_work": "该方案避免在循环计算过程中依赖电子处理，从而降低电子时钟频率形成的瓶颈。",
        "research_value": "该工作为超快时序信号处理和全光智能计算提供了实验依据。",
        "limitations_and_open_questions": ["摘要未提供能效、训练方法、校准流程和完整器件架构。"],
        "optical_neural_network_analysis": {
            "architecture_type": "free_space",
            "training_method": "未提供",
            "optical_nonlinearity": "作者报告使用非线性光学功能，但摘要未说明具体实现机制。",
            "calibration_requirements": "未提供",
            "application_tasks": ["微腔孤子状态分析"],
            "hardware_validation": "physical_experiment",
        },
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_and_abstract_only",
            "full_text_method_context_used": False,
            "full_text_method_source_url": None,
            "unsupported_numbers_detected": False,
            "missing_information": ["公开正文方法上下文"],
        },
    }


def prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate_id = "candidate-review-123"
    data = tmp_path / "data"
    request_path = data / "summary_requests/2026-08-04.jsonl"
    summary_path = data / "summaries/2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    summary_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(source_request(candidate_id)) + "\n")
    summary_path.write_text(json.dumps(generated_summary(candidate_id)) + "\n")
    manifest_path = tmp_path / "state/summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "digest_date": "2026-08-04",
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "output_language": "zh-CN",
                "full_text_used": False,
                "full_text_method_contexts": {},
                "email_enabled": False,
                "summary_history_updated": False,
            }
        )
    )
    review_manifest = tmp_path / "state/summary_review_manifest.json"
    return manifest_path, review_manifest, tmp_path / "state/summary_history.json"


def build(tmp_path: Path) -> tuple[dict, Path, Path]:
    manifest, review_manifest, history = prepare(tmp_path)
    state = build_review_packet(
        generation_manifest_path=manifest,
        summary_schema_path=ROOT / "schemas/paper_summary.schema.json",
        output_root=tmp_path / "data",
        review_manifest_path=review_manifest,
    )
    return state, review_manifest, history


def test_review_packet_repairs_architecture_and_exposes_method_checklist(
    tmp_path: Path,
) -> None:
    state, review_manifest, _ = build(tmp_path)
    assert state["status"] == "pending_human_review"
    assert state["architecture_repairs"] == 1
    assert state["output_language"] == "zh-CN"
    packet = json.loads((tmp_path / "data/reviews/2026-08-04.review.json").read_text())
    paper = packet["papers"][0]
    assert paper["summary"]["optical_neural_network_analysis"]["architecture_type"] == "unclear"
    assert paper["automated_checks"]["architecture_consistent"] is True
    assert paper["automated_checks"]["architecture_repaired"] is True
    assert paper["automated_checks"]["chinese_valid"] is True
    assert paper["automated_checks"]["method_depth_valid"] is True
    markdown = (tmp_path / "data/reviews/2026-08-04.review.md").read_text()
    assert "原始摘要" in markdown
    assert "方法原理已经讲清楚" in markdown
    assert "没有明确架构证据" in markdown
    assert json.loads(review_manifest.read_text())["summary_history_updated"] is False


def test_hold_for_revision_does_not_update_history(tmp_path: Path) -> None:
    _, review_manifest, history = build(tmp_path)
    result = finalize_review(
        state_root=tmp_path,
        review_manifest_path=review_manifest,
        history_path=history,
        decision="hold_for_revision",
        confirmation="REVIEWED",
        reviewer="reviewer",
        reviewed_at="2026-08-04T15:30:00Z",
        notes="方法说明需要修改。",
    )
    assert result["status"] == "revision_requested"
    assert result["summary_history_updated"] is False
    assert not history.exists()


def test_approve_all_updates_history_and_keeps_email_disabled(tmp_path: Path) -> None:
    _, review_manifest, history = build(tmp_path)
    result = finalize_review(
        state_root=tmp_path,
        review_manifest_path=review_manifest,
        history_path=history,
        decision="approve_all",
        confirmation="REVIEWED",
        reviewer="reviewer",
        reviewed_at="2026-08-04T15:30:00Z",
        notes="已核对方法原理和实施过程。",
    )
    assert result["status"] == "approved"
    assert result["summary_history_updated"] is True
    assert result["email_enabled"] is False
    completed = json.loads(history.read_text())["completed_candidate_ids"]
    assert completed["candidate-review-123"]["output_language"] == "zh-CN"
    digest = json.loads((tmp_path / "data/digests/2026-08-04.generated.json").read_text())
    assert digest["status"] == "approved_human_review"
    assert digest["safety"]["summary_history_updated"] is True
    assert digest["safety"]["email_enabled"] is False
    markdown = (tmp_path / "data/digests/2026-08-04.generated.md").read_text()
    assert "## 人工评审" in markdown


def test_finalization_rejects_artifact_tampering(tmp_path: Path) -> None:
    _, review_manifest, history = build(tmp_path)
    summary_path = tmp_path / "data/summaries/2026-08-04.jsonl"
    summary_path.write_text(summary_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed after packet creation"):
        finalize_review(
            state_root=tmp_path,
            review_manifest_path=review_manifest,
            history_path=history,
            decision="approve_all",
            confirmation="REVIEWED",
            reviewer="reviewer",
            reviewed_at="2026-08-04T15:30:00Z",
        )

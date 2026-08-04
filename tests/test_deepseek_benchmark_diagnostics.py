from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.summarize.benchmark_deepseek_models_v2 import benchmark
from scripts.summarize.deepseek_provider import DeepSeekResponse


ROOT = Path(__file__).resolve().parents[1]


def request(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt": "请只依据摘要返回中文 JSON。",
        "source": {
            "title": "Optical hardware for neural computation",
            "authors": ["Alice Researcher"],
            "venue": "Optica",
            "year": 2026,
            "source_type": "google_scholar_email",
            "doi": "10.1000/example",
            "openalex_id": None,
            "landing_page": "https://doi.org/10.1000/example",
            "open_access_url": None,
            "abstract": "The authors report experimental optical hardware validation.",
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.9,
            "decision": "mandatory",
            "score_breakdown": [],
        },
    }


def valid_summary(candidate_id: str) -> dict:
    return {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": candidate_id,
        "output_language": "zh-CN",
        "core_problem": "这项工作关注如何在真实光学硬件上完成神经计算，并确认计算链路能够被实验验证。",
        "method_and_architecture": "摘要描述了一个由输入编码、并行光学处理、输出探测和电子结果读出模块共同组成的完整硬件计算系统。",
        "method_principle": (
            "该方法把输入信息编码到光场中，并借助光学传播、干涉或器件响应完成神经网络需要的线性组合。"
            "由于多个光学通道可以同时传播，系统能够并行处理输入的不同分量。"
            "核心光学模块产生的输出随后由探测和电子读出模块转换为任务结果，从而构成从输入编码、物理计算到结果测量的完整链路。"
            "摘要只说明进行了实验性硬件验证，没有提供训练策略、器件参数和校准流程，因此这些内容必须保留为未提供。"
        ),
        "method_implementation": [
            "实施时先准备任务输入，并将其转换为光学硬件可以接收的编码形式。编码信号进入核心光学处理模块后，各通道按照预设连接关系并行完成变换，形成携带任务信息的输出光场。",
            "系统随后使用探测器或电子读出模块获取光学输出，并将测量结果用于任务判断和硬件验证。摘要没有说明参数训练、误差补偿、校准频率或长期运行方式，因此不能进一步推断。",
        ],
        "main_contributions": ["作者报告了光学硬件计算方法及其实验验证。"],
        "reported_results": [
            {
                "claim": "作者报告完成了实验性光学硬件验证。",
                "reported_by_authors": True,
                "basis": "abstract",
            }
        ],
        "distinction_from_prior_work": "该工作强调在真实硬件中执行光学计算，而不是只进行软件模拟。",
        "research_value": "该结果有助于评估光学神经硬件的工程可行性和后续研究价值。",
        "limitations_and_open_questions": ["摘要未提供训练、校准和长期稳定性信息。"],
        "optical_neural_network_analysis": {
            "architecture_type": "unclear",
            "training_method": "未提供",
            "optical_nonlinearity": "未提供",
            "calibration_requirements": "未提供",
            "application_tasks": [],
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


class FakeClient:
    def __init__(self, model: str) -> None:
        self.model = model

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        match = re.search(r'"candidate_id":\s*"([^"]+)"', kwargs["system_prompt"])
        assert match
        candidate_id = match.group(1)
        content = (
            json.dumps(valid_summary(candidate_id), ensure_ascii=False)
            if self.model == "deepseek-v4-flash"
            else json.dumps({"candidate_id": "wrong-id"})
        )
        return DeepSeekResponse(
            content=content,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
            model=self.model,
        )


def test_partial_model_failure_preserves_successful_model_output(tmp_path: Path) -> None:
    request_path = tmp_path / "state" / "request.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request("candidate-isolated")) + "\n", encoding="utf-8")
    dry_manifest = tmp_path / "state" / "dry.json"
    dry_manifest.write_text(
        json.dumps({"digest_date": "2026-08-04", "request_file": str(request_path)}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "state" / "benchmark.json"

    manifest = benchmark(
        dry_run_manifest_path=dry_manifest,
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        config_path=ROOT / "config" / "summary_generation.yaml",
        output_root=tmp_path / "data",
        manifest_path=manifest_path,
        api_key="test-key",
        client_factory=lambda model: FakeClient(model),
    )

    assert manifest["status"] == "partial_failure"
    assert manifest["candidate_level_isolation"] is True
    assert manifest["models"]["flash"]["summary_count"] == 1
    assert manifest["models"]["flash"]["failure_count"] == 0
    assert manifest["models"]["pro"]["summary_count"] == 0
    assert manifest["models"]["pro"]["failure_count"] == 1
    assert manifest["models"]["pro"]["failures"][0]["candidate_id"] == "candidate-isolated"
    assert Path(manifest["models"]["flash"]["summary_file"]).exists()
    assert manifest_path.exists()


def test_workflow_uses_candidate_isolation_and_restores_dry_run_files() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "benchmark-deepseek-models.yml"
    ).read_text(encoding="utf-8")
    assert "benchmark_deepseek_models_v3" in workflow
    assert "git restore --" in workflow
    assert "state/deepseek_benchmark_manifest.json" in workflow

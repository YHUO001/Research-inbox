from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize.benchmark_deepseek_models import benchmark
from scripts.summarize.deepseek_provider import DeepSeekResponse


ROOT = Path(__file__).resolve().parents[1]


def valid_summary(candidate_id: str, model: str) -> dict:
    return {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": candidate_id,
        "output_language": "zh-CN",
        "core_problem": "该论文关注如何在真实光学硬件中高效完成神经计算，并验证计算链路是否能够稳定运行。",
        "method_and_architecture": "摘要描述了由光学处理模块与电子读出模块共同组成的计算架构，用于执行神经网络中的核心变换。",
        "method_principle": (
            "该方法把神经网络中的线性组合映射为光场传播、干涉或器件响应，使多个输入通道能够在同一物理过程中并行完成加权变换。"
            "输入数据首先被编码成光学系统可以处理的幅度或相位表示，随后进入核心光学模块形成中间计算结果。"
            "探测和电子读出模块再把光学输出转换为任务可用的数值或类别信息。"
            "各模块分别承担输入表示、并行计算和结果读出，因此构成完整的前向计算链路；摘要没有提供训练和校准细节，不能进一步补写。"
        ),
        "method_implementation": [
            "实施时先准备神经计算任务的输入，并将其编码到可控光场或相应的硬件输入端。编码后的信号进入光学处理模块，各通道依照预设连接关系完成并行变换，形成包含任务信息的输出光场。",
            "完成光学变换后，系统使用探测器和电子读出模块获取结果，并将读出值用于任务判断及硬件性能验证。摘要只说明进行了实验性光学硬件验证，没有给出训练超参数、校准周期或长期稳定性处理。",
        ],
        "main_contributions": [f"{model} 的测试输出识别出论文报告的光学硬件方法和实验验证。"],
        "reported_results": [
            {
                "claim": "作者报告完成了实验性光学硬件验证。",
                "reported_by_authors": True,
                "basis": "abstract",
            }
        ],
        "distinction_from_prior_work": "该工作强调真实硬件实现，而不是只在软件环境中模拟光学计算过程。",
        "research_value": "这项工作有助于判断光学神经网络架构是否具备进一步工程化和扩展研究的价值。",
        "limitations_and_open_questions": ["摘要未提供长期稳定性、校准流程和完整训练方法。"],
        "optical_neural_network_analysis": {
            "architecture_type": "hybrid",
            "training_method": "未提供",
            "optical_nonlinearity": "未提供",
            "calibration_requirements": "未提供",
            "application_tasks": ["神经计算"],
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


def summary_request(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt": "请只依据标题、元数据和摘要返回中文 JSON。",
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
            "abstract": "The authors report experimental optical hardware validation for neural computation.",
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.9,
            "decision": "mandatory",
            "score_breakdown": [],
        },
    }


class FakeModelClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict] = []

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        self.calls.append(kwargs)
        candidate_id = "candidate-benchmark"
        return DeepSeekResponse(
            content=json.dumps(valid_summary(candidate_id, self.model), ensure_ascii=False),
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000,
            },
            model=self.model,
        )


def test_flash_and_pro_use_identical_requests_and_isolated_outputs(tmp_path: Path) -> None:
    request_path = tmp_path / "runtime-state" / "data" / "summary_requests" / "2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(summary_request("candidate-benchmark")) + "\n",
        encoding="utf-8",
    )
    dry_manifest = tmp_path / "runtime-state" / "state" / "summary_generation_manifest.json"
    dry_manifest.parent.mkdir(parents=True)
    dry_manifest.write_text(
        json.dumps({"digest_date": "2026-08-04", "request_file": str(request_path)}),
        encoding="utf-8",
    )
    clients = {
        "deepseek-v4-flash": FakeModelClient("deepseek-v4-flash"),
        "deepseek-v4-pro": FakeModelClient("deepseek-v4-pro"),
    }
    clock_values = iter([0.0, 1.0, 2.0, 4.0])
    manifest_path = tmp_path / "runtime-state" / "state" / "deepseek_benchmark_manifest.json"
    manifest = benchmark(
        dry_run_manifest_path=dry_manifest,
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        config_path=ROOT / "config" / "summary_generation.yaml",
        output_root=tmp_path / "runtime-state" / "data",
        manifest_path=manifest_path,
        api_key="test-key",
        client_factory=lambda model: clients[model],
        clock=lambda: next(clock_values),
    )

    assert manifest["status"] == "completed"
    assert manifest["request_count"] == 1
    assert manifest["same_prompt_for_both_models"] is True
    assert manifest["thinking_enabled"] is False
    assert manifest["email_enabled"] is False
    assert manifest["summary_history_updated"] is False
    assert manifest["models"]["flash"]["estimated_cost_cny"] == 0.002
    assert manifest["models"]["pro"]["estimated_cost_cny"] == 0.006
    assert manifest["models"]["pro"]["estimated_cost_cny"] == 3 * manifest["models"]["flash"]["estimated_cost_cny"]
    assert manifest["models"]["flash"]["elapsed_seconds"] == 1.0
    assert manifest["models"]["pro"]["elapsed_seconds"] == 2.0
    assert manifest["candidate_comparisons"][0]["architecture_agreement"] is True
    assert manifest["candidate_comparisons"][0]["hardware_validation_agreement"] is True

    flash_call = clients["deepseek-v4-flash"].calls[0]
    pro_call = clients["deepseek-v4-pro"].calls[0]
    assert flash_call["user_prompt"] == pro_call["user_prompt"]
    assert flash_call["system_prompt"] == pro_call["system_prompt"]
    assert flash_call["thinking_enabled"] is False
    assert pro_call["thinking_enabled"] is False
    assert flash_call["model"] == "deepseek-v4-flash"
    assert pro_call["model"] == "deepseek-v4-pro"

    benchmark_root = (
        tmp_path / "runtime-state" / "data" / "benchmarks" / "deepseek" / "2026-08-04"
    )
    assert (benchmark_root / "flash" / "summaries" / "2026-08-04.jsonl").exists()
    assert (benchmark_root / "pro" / "summaries" / "2026-08-04.jsonl").exists()
    assert (benchmark_root / "comparison.md").exists()
    assert manifest_path.exists()


def test_benchmark_workflow_is_manual_and_does_not_touch_delivery_or_history() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "benchmark-deepseek-models.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "DEEPSEEK_API_KEY" in workflow
    assert "GMAIL_CLIENT_SECRET" not in workflow
    assert "gmail_sender" not in workflow
    assert "summary_history.json" not in workflow
    assert "data/summaries" not in workflow
    assert "data/benchmarks/deepseek" in workflow


def test_benchmark_config_uses_current_official_models_and_nonthinking_mode() -> None:
    import yaml

    config = yaml.safe_load(
        (ROOT / "config" / "summary_generation.yaml").read_text(encoding="utf-8")
    )
    benchmark_config = config["benchmark"]
    assert benchmark_config["thinking_enabled"] is False
    assert benchmark_config["same_prompt_for_all_models"] is True
    assert [item["model"] for item in benchmark_config["models"]] == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    assert benchmark_config["models"][0]["pricing"]["input_cache_miss_cny_per_million"] == 1.0
    assert benchmark_config["models"][1]["pricing"]["input_cache_miss_cny_per_million"] == 3.0

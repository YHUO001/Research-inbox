from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize.deepseek_provider import DeepSeekResponse
from scripts.summarize.evidence_guard import resolve_onn_architecture
from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.generate_summaries_production import generate


ROOT = Path(__file__).resolve().parents[1]


def request(candidate_id: str, abstract: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt": "请依据提供的摘要返回中文 JSON。",
        "source": {
            "title": "Optical neural computing hardware",
            "authors": ["Alice Researcher"],
            "venue": "Nature Communications",
            "year": 2026,
            "source_type": "google_scholar_email",
            "doi": "10.1000/example",
            "openalex_id": None,
            "landing_page": "https://doi.org/10.1000/example",
            "open_access_url": None,
            "abstract": abstract,
            "matched_projects": ["optical-neural-networks"],
            "mandatory": True,
            "score": 0.9,
            "decision": "mandatory",
            "score_breakdown": [],
        },
    }


def summary(candidate_id: str, claim: str, architecture: str = "unclear") -> dict:
    return {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": candidate_id,
        "output_language": "zh-CN",
        "core_problem": "这项工作试图解决光学神经计算系统在实际硬件中难以稳定执行和验证的问题。",
        "method_and_architecture": "作者构建光学计算硬件，并把输入信号依次送入光学变换模块和读出模块，从而完成目标计算任务。",
        "method_principle": (
            "该方法的基本原理是把神经网络中的线性变换映射到光学传播或干涉过程，"
            "利用光场的并行性同时处理多个输入分量。系统先将输入编码到可控光场中，"
            "再通过光学器件完成加权组合，最后由探测与读出模块得到计算结果。"
            "不同模块分别承担输入表示、核心变换和结果测量，因此能够形成完整的前向计算链路。"
            "摘要没有给出更细的器件参数和校准算法，所以这些部分应保留为未提供。"
        ),
        "method_implementation": [
            "实施时首先准备输入数据，并将其转换为光学系统可以接收的编码形式。随后把编码后的信号送入核心光学硬件，使各通道按照预定连接关系完成并行变换，整个过程保留输入到中间表示的对应关系。",
            "完成光学变换后，系统通过探测器或电子读出模块获取输出，并将读出结果用于任务判断或性能评估。摘要没有说明更具体的训练超参数、校准周期和长期稳定性处理，因此不能对这些环节作进一步推断。",
        ],
        "main_contributions": ["作者报告了光学计算硬件的构建与实验验证，并展示了该系统能够完成目标任务。"],
        "reported_results": [
            {"claim": claim, "reported_by_authors": True, "basis": "abstract"}
        ],
        "distinction_from_prior_work": "该工作强调在真实光学硬件上完成计算，而不是只进行软件模拟。",
        "research_value": "该结果有助于判断光学神经计算方案是否具备进一步工程化和扩展研究的价值。",
        "limitations_and_open_questions": ["摘要未提供完整的器件参数、校准流程和长期稳定性数据。"],
        "optical_neural_network_analysis": {
            "architecture_type": architecture,
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
            "missing_information": ["完整正文方法细节"],
        },
    }


class FakeClient:
    def __init__(self, values: list[dict]) -> None:
        self.values = list(values)
        self.calls = 0
        self.requests: list[dict] = []

    def complete_json(self, **kwargs) -> DeepSeekResponse:
        self.calls += 1
        self.requests.append(dict(kwargs))
        value = self.values[min(self.calls - 1, len(self.values) - 1)]
        return DeepSeekResponse(
            content=json.dumps(value, ensure_ascii=False),
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
            model="deepseek-v4-pro",
        )


def no_full_text(source: dict, *, config: dict) -> MethodContext:
    return MethodContext(str(source["candidate_id"]), "not_available", None, None, [], "")


def used_full_text(source: dict, *, config: dict) -> MethodContext:
    return MethodContext(
        str(source["candidate_id"]),
        "used",
        "https://example.org/article",
        "text/html",
        ["Methods", "Experimental setup"],
        "The method encodes the input into an optical field and sends it through the hardware. "
        "The output is detected and used for task evaluation.",
    )


def prepare(tmp_path: Path, item: dict) -> Path:
    request_path = tmp_path / "data" / "summary_requests" / "2026-08-04.jsonl"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "state" / "summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"digest_date": "2026-08-04", "request_file": str(request_path)}),
        encoding="utf-8",
    )
    return manifest_path


def run(
    tmp_path: Path,
    manifest_path: Path,
    client: FakeClient,
    loader=no_full_text,
) -> dict:
    return generate(
        dry_run_manifest_path=manifest_path,
        summary_schema_path=ROOT / "schemas" / "paper_summary.schema.json",
        config_path=ROOT / "config" / "summary_generation.yaml",
        output_root=tmp_path / "data",
        manifest_path=manifest_path,
        api_key="test-key",
        client=client,
        method_context_loader=loader,
    )


def test_architecture_evidence_resolves_observed_cases() -> None:
    hybrid = resolve_onn_architecture(
        "We validated a programmable spatial light modulator (SLM) system. "
        "Chip-scale integration via nano printing was also demonstrated."
    )
    integrated = resolve_onn_architecture(
        "Four optical analog cores are integrated on a monolithic chip."
    )
    unclear = resolve_onn_architecture(
        "The network analyzes soliton states from integrated optical microresonators."
    )
    assert hybrid.resolved_type == "hybrid"
    assert integrated.resolved_type == "integrated"
    assert unclear.resolved_type == "unclear"


def test_production_injects_metadata_normalizes_units_and_repairs_architecture(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate-expected"
    manifest_path = prepare(
        tmp_path,
        request(candidate_id, "The authors report a compute density of 5.16 TOPS/mm2."),
    )
    value = summary(
        "wrong-model-id",
        "作者报告计算密度达到 5.16 TOPS/mm²。",
        "free_space",
    )
    value["output_language"] = "en"
    client = FakeClient([value])
    state = run(tmp_path, manifest_path, client)
    repairs = state["transport_repairs"]
    assert state["status"] == "completed"
    assert repairs["candidate_id_repair_responses"] == 1
    assert repairs["generation_metadata_repair_responses"] == 1
    assert repairs["unit_format_normalization_responses"] == 1
    assert repairs["architecture_repairs"] == 1
    assert repairs["architecture_evidence"][candidate_id]["resolved_type"] == "unclear"
    generated = json.loads((tmp_path / "data/summaries/2026-08-04.jsonl").read_text())
    assert generated["candidate_id"] == candidate_id
    assert generated["output_language"] == "zh-CN"
    assert generated["optical_neural_network_analysis"]["architecture_type"] == "unclear"
    assert "5.16 TOPS/mm2" in generated["reported_results"][0]["claim"]


def test_production_accepts_tops_alias_without_changing_persisted_prompt(tmp_path: Path) -> None:
    candidate_id = "candidate-tops"
    manifest_path = prepare(
        tmp_path,
        request(candidate_id, "The OPU achieves 65.04 trillion operations per second (TOPS)."),
    )
    client = FakeClient([summary(candidate_id, "作者报告 OPU 达到 65.04 TOPS。")])
    state = run(tmp_path, manifest_path, client)
    assert state["status"] == "completed"
    assert state["transport_repairs"]["tops_alias_expansions"] == 1
    assert "Machine-only numeric grounding aliases" not in client.requests[0]["user_prompt"]


def test_production_uses_ephemeral_full_text_and_records_only_audit_metadata(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate-fulltext"
    manifest_path = prepare(tmp_path, request(candidate_id, "The system performs optical computing."))
    client = FakeClient([summary(candidate_id, "作者报告系统完成了光学计算。")])
    state = run(tmp_path, manifest_path, client, loader=used_full_text)
    context = state["full_text_method_contexts"][candidate_id]
    assert state["full_text_used"] is True
    assert context["status"] == "used"
    assert context["source_url"] == "https://example.org/article"
    assert context["text_persisted"] is False
    assert "text" not in context
    assert "公开正文中的方法相关上下文" in client.requests[0]["user_prompt"]
    generated = json.loads((tmp_path / "data/summaries/2026-08-04.jsonl").read_text())
    assert generated["verification"]["full_text_method_context_used"] is True
    assert generated["verification"]["full_text_method_source_url"] == "https://example.org/article"


def test_production_preserves_strict_approximation_semantics(tmp_path: Path) -> None:
    candidate_id = "candidate-approximation"
    manifest_path = prepare(
        tmp_path,
        request(candidate_id, "Conventional processors remain at ~5 GHz."),
    )
    client = FakeClient(
        [
            summary(candidate_id, "作者报告传统处理器保持在 5 GHz。"),
            summary(candidate_id, "作者报告传统处理器保持在 5 GHz。"),
        ]
    )
    with pytest.raises(RuntimeError, match="failed local validation"):
        run(tmp_path, manifest_path, client)
    persisted = json.loads(manifest_path.read_text())
    assert persisted["status"] == "failed_validation"
    assert "unsupported numeric claims: 5ghz" in persisted["failures"][0]["reason"]
    assert not (tmp_path / "data/summaries/2026-08-04.jsonl").exists()

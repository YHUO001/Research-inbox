from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.summarize import staged_summary_pipeline_safe as safe_pipeline
from scripts.summarize.abstract_fallback_policy import (
    ABSTRACT_ONLY_BASIS,
    FULL_TEXT_BASIS,
)
from scripts.summarize.deepseek_provider import DeepSeekResponse
from scripts.summarize.numeric_grounding_repair import (
    full_text_evidence_from_user_prompt,
    repair_summary_numeric_grounding,
    source_evidence_from_user_prompt,
    unsupported_numeric_diagnostics,
)
from scripts.summarize.staged_summary_pipeline_safe import shared_numeric_grounding


def test_safe_numeric_idioms_are_rewritten_without_losing_meaning() -> None:
    summary = {
        "method_principle": (
            "SAGE仅引入O(1)额外状态，并把收缩因子限制在[0,1]。"
        ),
        "method_implementation": [
            "第1步：估计信号能量。",
            "2) 根据估计结果调整更新。",
            "步骤3：输出收缩后的更新。",
        ],
    }

    repaired, records = repair_summary_numeric_grounding(
        summary,
        title="Noise-Aware Shrinkage",
        abstract="The method introduces only constant additional state and a bounded shrinkage factor.",
    )

    assert "O(1)" not in repaired["method_principle"]
    assert "[0,1]" not in repaired["method_principle"]
    assert "常数级" in repaired["method_principle"]
    assert "有界范围" in repaired["method_principle"]
    assert repaired["method_implementation"] == [
        "估计信号能量。",
        "根据估计结果调整更新。",
        "输出收缩后的更新。",
    ]
    assert records
    assert unsupported_numeric_diagnostics(
        repaired,
        title="Noise-Aware Shrinkage",
        abstract="The method introduces only constant additional state and a bounded shrinkage factor.",
    ) == []


def test_remaining_unsupported_quantitative_sentence_is_redacted() -> None:
    summary = {
        "method_principle": (
            "该方法处理OPT-1.3B和OPT-6.7B。实验额外使用8个方向并执行4次估计。"
        ),
        "reported_results": [
            {
                "basis": "abstract",
                "claim": "作者报告性能提升了12%。",
                "reported_by_authors": True,
            }
        ],
    }
    abstract = "Experiments use OPT-1.3B and OPT-6.7B under the same privacy budgets."

    repaired, records = repair_summary_numeric_grounding(
        summary,
        title="Private fine-tuning of large language models",
        abstract=abstract,
    )

    assert "OPT-1.3B" in repaired["method_principle"]
    assert "OPT-6.7B" in repaired["method_principle"]
    assert "8个方向" not in repaired["method_principle"]
    assert "4次估计" not in repaired["method_principle"]
    assert "相关定量细节未在标题或摘要中提供" in repaired["method_principle"]
    assert "12%" not in repaired["reported_results"][0]["claim"]
    assert "相关定量结果未在标题或摘要中提供" in repaired["reported_results"][0]["claim"]
    assert {record["path"] for record in records} >= {
        "method_principle",
        "reported_results[0].claim",
    }
    assert shared_numeric_grounding(
        repaired,
        title="Private fine-tuning of large language models",
        abstract=abstract,
    ) == []


def test_unrepairable_diagnostic_includes_field_path_and_context() -> None:
    summary = {
        "method_principle": "系统使用8个并行方向。",
    }
    records = unsupported_numeric_diagnostics(
        summary,
        title="A method",
        abstract="The method uses structured perturbations.",
    )
    assert records == [
        {
            "path": "method_principle",
            "token": "8",
            "excerpt": "系统使用8个并行方向。",
        }
    ]


def test_full_text_numeric_evidence_supports_signed_ranges_and_unicode_minus() -> None:
    summary = {
        "method_principle": "角速度范围为−1到1 rad/s。",
    }
    assert unsupported_numeric_diagnostics(
        summary,
        title="Autonomous navigation",
        abstract="The actor outputs continuous velocity commands.",
        extra_evidence="The angular velocity is normalized to the range [-1, 1] rad/s.",
    ) == []


def test_source_evidence_parser_ignores_trailing_fallback_instruction() -> None:
    prompt = (
        "其他说明\n来源记录：\n"
        '{"title":"Paper 4","abstract":"Uses 8 evaluations."}'
        "\n\n证据降级要求：不要扩写。"
    )
    assert source_evidence_from_user_prompt(prompt) == (
        "Paper 4",
        "Uses 8 evaluations.",
    )


def test_full_text_evidence_parser_extracts_only_ephemeral_method_context() -> None:
    prompt = (
        '来源记录：\n{"title":"Paper","abstract":"Abstract."}'
        "\n\n公开正文中的方法相关上下文：\n"
        "- 来源：https://example.test/paper\n"
        "- 方法上下文开始：\n"
        "Action range is [-1, 1] rad/s.\n"
        "- 方法上下文结束。"
    )
    assert full_text_evidence_from_user_prompt(prompt) == (
        "Action range is [-1, 1] rad/s."
    )


def test_transport_repair_applies_to_abstract_fallback(monkeypatch) -> None:
    response = DeepSeekResponse(
        content=json.dumps(
            {
                "method_principle": "SAGE仅引入O(1)额外状态。",
                "verification": {
                    "information_basis": ABSTRACT_ONLY_BASIS,
                    "missing_information": [],
                },
            },
            ensure_ascii=False,
        ),
        usage={},
        model="test-model",
    )
    monkeypatch.setattr(
        safe_pipeline,
        "_ORIGINAL_COMPLETE_JSON",
        lambda self, **kwargs: response,
    )
    monkeypatch.setattr(
        safe_pipeline.pipeline,
        "expected_example",
        lambda prompt: {"candidate_id": "candidate-1"},
    )
    client = SimpleNamespace(
        diagnostics=SimpleNamespace(generation_metadata_repair_responses=0)
    )

    repaired = safe_pipeline.complete_json_with_fallback_normalization(
        client,
        system_prompt="test",
        user_prompt=(
            '来源记录：\n{"title":"SAGE","abstract":"constant additional state"}'
            "\n公开全文已尝试获取 3 次"
        ),
    )
    value = json.loads(repaired.content)

    assert "O(1)" not in value["method_principle"]
    assert "常数级" in value["method_principle"]
    assert client.diagnostics.numeric_grounding_repair_responses == 1


def test_transport_repair_preserves_grounded_full_text_numbers_and_redacts_only_unsupported(
    monkeypatch,
) -> None:
    response = DeepSeekResponse(
        content=json.dumps(
            {
                "candidate_id": "candidate-2",
                "method_principle": (
                    "公开全文给出的角速度范围为-1到1 rad/s。"
                    "模型另外声称系统使用777个并行通道。"
                ),
                "verification": {
                    "information_basis": FULL_TEXT_BASIS,
                    "full_text_method_context_used": True,
                    "missing_information": [],
                },
            },
            ensure_ascii=False,
        ),
        usage={},
        model="test-model",
    )
    captured: dict[str, str] = {}

    def fake_complete(self, **kwargs):
        captured["user_prompt"] = str(kwargs.get("user_prompt") or "")
        return response

    monkeypatch.setattr(safe_pipeline, "_ORIGINAL_COMPLETE_JSON", fake_complete)
    client = SimpleNamespace(
        diagnostics=SimpleNamespace(generation_metadata_repair_responses=0)
    )
    user_prompt = (
        "所有数字仍必须出现在标题或摘要中；即使正文上下文包含额外数字，也不要把这些数字写入摘要。\n"
        '来源记录：\n{"title":"Navigation","abstract":"Continuous control is used."}'
        "\n\n公开正文中的方法相关上下文（临时证据）：\n"
        "- 方法上下文开始：\n"
        "The angular velocity range is [-1, 1] rad/s.\n"
        "- 方法上下文结束。"
    )

    repaired = safe_pipeline.complete_json_with_fallback_normalization(
        client,
        system_prompt="test",
        user_prompt=user_prompt,
    )
    value = json.loads(repaired.content)

    assert "-1到1 rad/s" in value["method_principle"]
    assert "777" not in value["method_principle"]
    assert "标题、摘要或公开正文方法上下文" in value["method_principle"]
    assert "若追加了公开正文方法上下文" in captured["user_prompt"]
    assert "所有数字仍必须出现在标题或摘要中" not in captured["user_prompt"]
    assert client.diagnostics.numeric_grounding_repair_responses == 1
    assert shared_numeric_grounding(
        value,
        title="Navigation",
        abstract="Continuous control is used.",
    ) == []

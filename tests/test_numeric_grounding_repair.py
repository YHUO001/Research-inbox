from __future__ import annotations

from scripts.summarize.numeric_grounding_repair import (
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
    assert not repaired["method_implementation"][0].startswith("第1步")
    assert not repaired["method_implementation"][1].startswith("2)")
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

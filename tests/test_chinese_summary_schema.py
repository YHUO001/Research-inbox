from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def schema() -> dict:
    return json.loads(
        (ROOT / "schemas" / "paper_summary.schema.json").read_text(encoding="utf-8")
    )


def field_validator(field: str) -> Draft202012Validator:
    return Draft202012Validator(schema()["properties"][field])


def valid_abstract_summary() -> dict:
    return {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": "candidate-abstract",
        "output_language": "zh-CN",
        "core_problem": "论文研究光学神经网络在温度变化条件下保持稳定推理性能的问题。",
        "method_and_architecture": "系统采用可调光学器件构成计算层，但摘要没有提供完整器件布局和实验连接细节。",
        "method_principle": (
            "作者在训练过程中加入与温度变化相关的参数扰动，使模型在摘要所描述的变化范围内学习更稳健的表示。"
            "由于没有取得公开全文，这里只能说明摘要明确给出的总体机制，具体损失函数、优化器和硬件控制流程均未提供。"
        ),
        "method_implementation": [
            "按照摘要，先建立光学计算模型并在训练阶段加入温度相关扰动，再使用训练后的参数评估温度变化下的输出；更细的装置与参数设置未提供。"
        ],
        "main_contributions": ["提出面向温度变化的光学网络训练策略。"],
        "reported_results": [],
        "distinction_from_prior_work": "摘要仅表明该方法针对温度鲁棒性，未提供充分的既有方法比较。",
        "research_value": "该工作为评估光学网络在非理想环境下的稳定性提供了研究方向。",
        "limitations_and_open_questions": ["公开全文未取得，具体实现、数据集和定量结果仍需核实。"],
        "optical_neural_network_analysis": {
            "architecture_type": "unclear",
            "training_method": "摘要说明训练时加入温度相关扰动。",
            "optical_nonlinearity": "not_available",
            "calibration_requirements": "not_available",
            "application_tasks": [],
            "hardware_validation": "unclear",
        },
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_and_abstract_only",
            "full_text_method_context_used": False,
            "full_text_method_source_url": None,
            "unsupported_numbers_detected": False,
            "missing_information": ["公开全文获取失败，当前仅依据摘要生成。"],
        },
    }


def test_reader_facing_scalar_fields_require_chinese() -> None:
    validator = field_validator("research_value")
    assert list(validator.iter_errors("This work is useful for optical computing research."))
    assert not list(
        validator.iter_errors("这项工作有助于评估光学计算方法的研究价值和工程潜力。")
    )


def test_reader_facing_list_items_require_chinese_or_not_available() -> None:
    validator = field_validator("limitations_and_open_questions")
    assert list(validator.iter_errors(["Long-term stability is not discussed."]))
    assert not list(validator.iter_errors(["摘要未讨论系统的长期稳定性和校准方式。"]))
    assert not list(validator.iter_errors(["not_available"]))


def test_abstract_only_brief_accepts_one_substantive_chinese_paragraph() -> None:
    validator = Draft202012Validator(schema())
    assert not list(validator.iter_errors(valid_abstract_summary()))


def test_full_text_summary_keeps_strict_method_depth() -> None:
    validator = Draft202012Validator(schema())
    summary = deepcopy(valid_abstract_summary())
    summary["verification"] = {
        "information_basis": "title_metadata_abstract_and_open_full_text_methods",
        "full_text_method_context_used": True,
        "full_text_method_source_url": "https://example.org/article",
        "unsupported_numbers_detected": False,
        "missing_information": [],
    }
    assert list(validator.iter_errors(summary))

    summary["method_principle"] = summary["method_principle"] * 2
    summary["method_implementation"] = [
        "实施时首先将输入数据编码为光学系统能够接收的信号，并送入核心处理模块完成并行变换，从而形成包含任务信息的中间光场表示，同时保留各输入通道与输出之间的对应关系。",
        "随后使用探测与电子读出模块获取计算结果，并将测量值用于任务判断；未在证据中说明的训练参数和校准过程必须明确标记为未提供，也不能依据常识自行补全。",
    ]
    assert not list(validator.iter_errors(summary))

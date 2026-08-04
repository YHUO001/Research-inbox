from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def field_validator(field: str) -> Draft202012Validator:
    schema = json.loads(
        (ROOT / "schemas" / "paper_summary.schema.json").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema["properties"][field])


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


def test_method_implementation_requires_multiple_substantive_chinese_paragraphs() -> None:
    validator = field_validator("method_implementation")
    valid = [
        "实施时首先将输入数据编码为光学系统能够接收的信号，并送入核心处理模块完成并行变换，从而形成包含任务信息的中间光场表示。",
        "随后使用探测与电子读出模块获取计算结果，并将测量值用于任务判断；未在证据中说明的训练参数和校准过程必须明确标记为未提供。",
    ]
    assert not list(validator.iter_errors(valid))
    assert list(validator.iter_errors([valid[0]]))
    assert list(
        validator.iter_errors(
            [
                "The input is encoded and sent through the optical system for processing.",
                "The detector measures the output and produces the final prediction result.",
            ]
        )
    )

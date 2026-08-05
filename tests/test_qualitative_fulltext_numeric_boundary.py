from __future__ import annotations

from scripts.summarize.prepare_fulltext_bounded import redact_method_context_numbers
from scripts.summarize.staged_summary_pipeline_safe import shared_numeric_grounding


def test_fulltext_method_numbers_are_redacted_before_prompting() -> None:
    source = (
        "The modulator is driven at 10 GSa/s near ~1.55 um, with 1414 samples "
        "and a delay of 0.3 ns. The optical signal then enters the recurrent stage."
    )
    redacted = redact_method_context_numbers(source)
    assert "10 GSa/s" not in redacted
    assert "1.55" not in redacted
    assert "1414" not in redacted
    assert "0.3 ns" not in redacted
    assert redacted.count("[数值见正文]") >= 4
    assert "The optical signal then enters the recurrent stage." in redacted


def test_exact_source_value_may_be_weakened_to_approximate_output() -> None:
    summary = {"method_principle": "作者报告该处理器达到约 65 TOPS。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="65 TOPS optoelectronic multi-core computing",
        abstract="The system achieves 65 TOPS computational speed.",
    )
    assert unsupported == []


def test_approximate_source_value_may_not_be_strengthened_to_exact_output() -> None:
    summary = {"method_principle": "传统系统的工作频率为 5 GHz。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="All-optical computing",
        abstract="Conventional electronics are limited to ~5 GHz.",
    )
    assert unsupported == ["5ghz"]


def test_unseen_numeric_claim_remains_rejected() -> None:
    summary = {"method_principle": "系统使用 488 nm 光源。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="Optical neural network",
        abstract="The system uses a programmable optical input.",
    )
    assert unsupported == ["488nm"]

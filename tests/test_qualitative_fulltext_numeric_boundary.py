from __future__ import annotations

from scripts.summarize.prepare_fulltext_bounded import full_text_numeric_aliases
from scripts.summarize.staged_summary_pipeline_safe import shared_numeric_grounding


def test_fulltext_method_numbers_become_temporary_grounding_aliases() -> None:
    source = (
        "The modulator is driven at 10 GSa/s near ~1.55 um, with 1414 samples "
        "and a delay of 0.3 ns. The optical signal then enters the recurrent stage."
    )
    aliases = full_text_numeric_aliases(source)
    assert aliases == ["10", "1.55", "1414", "0.3"]


def test_approximate_and_exact_forms_are_treated_as_equivalent() -> None:
    summary = {"method_principle": "传统系统的工作频率为 5 GHz。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="All-optical computing",
        abstract="Conventional electronics are limited to ~5 GHz.",
    )
    assert unsupported == []


def test_small_rounding_difference_is_allowed() -> None:
    summary = {"method_principle": "系统工作波长约为 1544 nm，性能约为 65 TOPS。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="Optical computing",
        abstract="Machine-only open-full-text numeric grounding aliases: 1540; 65.04.",
    )
    assert unsupported == []


def test_units_do_not_need_to_match_when_numeric_value_exists() -> None:
    summary = {"method_principle": "系统使用 10 GHz 时钟和 100 mW 功率。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="Optical computing",
        abstract="Machine-only open-full-text numeric grounding aliases: 10; 100.",
    )
    assert unsupported == []


def test_clearly_different_numeric_claim_remains_rejected() -> None:
    summary = {"method_principle": "系统包含约 120 个并行通道。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="Optical computing",
        abstract="The system contains 100 parallel channels.",
    )
    assert unsupported == ["120"]


def test_unseen_numeric_claim_remains_rejected() -> None:
    summary = {"method_principle": "系统使用 488 nm 光源。"}
    unsupported = shared_numeric_grounding(
        summary,
        title="Optical neural network",
        abstract="The system uses a programmable optical input.",
    )
    assert unsupported == ["488"]

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.delivery.send_daily_digest_safe import evidence_aware_empty_digest_body
from scripts.summarize.abstract_fallback_policy import (
    abstract_fallback_notice,
    normalize_abstract_fallback_summary,
    validate_method_depth_by_evidence,
)
from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_digest import load_jsonl
from scripts.summarize.prepare_fulltext_bounded import (
    generation_manifest_copy_with_full_text_numbers,
    retried_context,
)


ROOT = Path(__file__).resolve().parents[1]


def write_batch(tmp_path: Path, *, abstract: str | None) -> tuple[Path, Path]:
    request_path = tmp_path / "runtime-state/data/summary_requests/2026-08-05.jsonl"
    request_path.parent.mkdir(parents=True)
    request = {
        "candidate_id": "candidate-fallback",
        "prompt": "请生成中文摘要。",
        "source": {
            "title": "Thermal-resilient optical neural network",
            "abstract": abstract,
            "doi": "10.1000/fallback",
            "landing_page": "https://doi.org/10.1000/fallback",
        },
    }
    request_path.write_text(
        json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_path = tmp_path / "runtime-state/state/summary_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "digest_date": "2026-08-05",
                "request_count": 1,
                "request_file": str(request_path),
                "request_sha256": "old",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, request_path


def failed_context() -> MethodContext:
    return retried_context(
        MethodContext(
            candidate_id="candidate-fallback",
            status="not_available",
            source_url="https://example.org/article",
            media_type="text/html",
            section_headings=[],
            text="",
            error="HTTP 401",
        ),
        3,
        final=True,
    )


def fallback_config() -> dict:
    return {
        "execution": {"use_full_text": True},
        "full_text": {
            "enabled": True,
            "retrieval_attempts": 3,
            "skip_when_abstract_missing": True,
        },
    }


def test_failed_full_text_with_abstract_retains_request_and_marks_fallback(
    tmp_path: Path,
) -> None:
    manifest_path, persistent_request = write_batch(
        tmp_path,
        abstract="The authors train an optical neural network with temperature perturbations.",
    )

    def loader(source: dict, *, config: dict) -> MethodContext:
        del source, config
        return failed_context()

    temporary_manifest, _, contexts = generation_manifest_copy_with_full_text_numbers(
        manifest_path,
        tmp_path / "temporary",
        config=fallback_config(),
        method_context_loader=loader,
    )
    temporary = json.loads(temporary_manifest.read_text(encoding="utf-8"))
    requests = load_jsonl(Path(temporary["request_file"]))
    persistent = load_jsonl(persistent_request)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(requests) == 1
    assert requests[0]["full_text_fallback"] is True
    assert requests[0]["full_text_retrieval_attempts"] == 3
    assert "摘要级短讯" in requests[0]["prompt"]
    assert len(persistent) == 1
    assert stored["abstract_fallback_count"] == 1
    assert stored["skipped_no_abstract_count"] == 0
    assert contexts["candidate-fallback"]["attempt_count"] == 3
    assert contexts["candidate-fallback"]["fallback_decision"] == "generate_from_abstract"


def test_failed_full_text_without_abstract_skips_before_model(tmp_path: Path) -> None:
    manifest_path, persistent_request = write_batch(tmp_path, abstract=None)

    def loader(source: dict, *, config: dict) -> MethodContext:
        del source, config
        return failed_context()

    temporary_manifest, _, contexts = generation_manifest_copy_with_full_text_numbers(
        manifest_path,
        tmp_path / "temporary",
        config=fallback_config(),
        method_context_loader=loader,
    )
    temporary = json.loads(temporary_manifest.read_text(encoding="utf-8"))
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert load_jsonl(Path(temporary["request_file"])) == []
    assert load_jsonl(persistent_request) == []
    assert stored["request_count"] == 0
    assert stored["status"] == "no_eligible_evidence"
    assert stored["skipped_no_abstract_candidate_ids"] == ["candidate-fallback"]
    assert contexts["candidate-fallback"]["fallback_decision"] == "skipped_no_abstract"


def test_abstract_depth_is_shorter_but_full_text_depth_remains_strict() -> None:
    summary = {
        "method_principle": "摘要说明训练阶段加入温度相关扰动，使模型学习对环境变化更稳定的参数，但没有给出更细的算法和硬件执行步骤。",
        "method_implementation": [
            "先建立摘要描述的光学网络，再在训练阶段加入温度扰动并评估输出稳定性，其他细节未提供。"
        ],
        "verification": {"information_basis": "title_metadata_and_abstract_only"},
    }
    assert validate_method_depth_by_evidence(summary) == []

    summary["verification"]["information_basis"] = (
        "title_metadata_abstract_and_open_full_text_methods"
    )
    assert validate_method_depth_by_evidence(summary)


def test_shape_normalization_does_not_invent_content() -> None:
    summary = {
        "method_implementation": "摘要只说明训练时注入温度扰动。",
        "optical_neural_network_analysis": {
            "application_tasks": "未提供",
        },
    }
    normalized, changed = normalize_abstract_fallback_summary(summary)
    assert changed is True
    assert normalized["method_implementation"] == ["摘要只说明训练时注入温度扰动。"]
    assert normalized["optical_neural_network_analysis"]["application_tasks"] == []


def test_all_skipped_batch_can_send_an_explanatory_zero_summary_digest(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "summary_generation_manifest.json").write_text(
        json.dumps(
            {
                "status": "no_eligible_evidence",
                "request_count": 0,
                "skipped_no_abstract_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (state / "selection_manifest.json").write_text(
        json.dumps({"summary_slot_count": 2}), encoding="utf-8"
    )

    body = evidence_aware_empty_digest_body(tmp_path, "2026-08-05")
    assert "公开全文连续获取三次仍失败" in body
    assert "DeepSeek：未调用" in body
    assert "跳过论文数：`2`" in body


def test_production_config_and_workflow_wire_the_fallback_policy() -> None:
    config = yaml.safe_load(
        (ROOT / "config/summary_generation.yaml").read_text(encoding="utf-8")
    )
    full_text = config["full_text"]
    assert full_text["retrieval_attempts"] == 3
    assert full_text["fallback_to_abstract_after_attempts"] is True
    assert full_text["skip_when_abstract_missing"] is True

    workflow = (
        ROOT / ".github/workflows/daily-research-inbox.yml"
    ).read_text(encoding="utf-8")
    assert "steps.fulltext.outputs.prepared_request_count" in workflow
    assert "scripts.summarize.finalize_automatic_safe" in workflow
    assert "scripts.delivery.send_daily_digest_safe" in workflow
    assert abstract_fallback_notice(3).startswith("公开全文已尝试获取 3 次")

from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize import build_review_packet_compatible as compatible
from scripts.summarize.prepare_digest import atomic_write, load_json


def loose_generation() -> dict:
    return {
        "status": "completed",
        "digest_date": "2026-08-05",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "request_count": 3,
        "summary_count": 3,
        "failure_count": 0,
        "email_enabled": False,
        "summary_history_updated": False,
        "numeric_grounding_scope": compatible.LOOSE_NUMERIC_SCOPE,
        "numeric_matching": {
            "approximation_markers_ignored": True,
            "unit_format_ignored": True,
            "relative_tolerance": 0.02,
            "absolute_tolerance": 0.02,
            "evidence_sources": [
                "title",
                "abstract",
                "temporary_open_full_text_methods",
            ],
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def test_inheritance_requires_a_complete_audited_loose_run() -> None:
    generation = loose_generation()
    assert compatible.inherits_completed_loose_validation(generation) is True

    failed = dict(generation, failure_count=1)
    assert compatible.inherits_completed_loose_validation(failed) is False

    incomplete = dict(generation, summary_count=2)
    assert compatible.inherits_completed_loose_validation(incomplete) is False

    missing_source = dict(generation)
    missing_source["numeric_matching"] = dict(generation["numeric_matching"])
    missing_source["numeric_matching"]["evidence_sources"] = ["title", "abstract"]
    assert compatible.inherits_completed_loose_validation(missing_source) is False

    excessive_tolerance = dict(generation)
    excessive_tolerance["numeric_matching"] = dict(generation["numeric_matching"])
    excessive_tolerance["numeric_matching"]["relative_tolerance"] = 0.2
    assert compatible.inherits_completed_loose_validation(excessive_tolerance) is False


def test_review_inherits_validation_and_refreshes_audit_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    generation_path = tmp_path / "state/summary_generation_manifest.json"
    review_manifest_path = tmp_path / "state/summary_review_manifest.json"
    output_root = tmp_path / "data"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    write_json(generation_path, loose_generation())

    strict_validator = lambda *args, **kwargs: ["0.1"]
    monkeypatch.setattr(
        compatible.review_core,
        "validate_summary_numeric_grounding",
        strict_validator,
    )

    def fake_build_review_packet(**kwargs):
        assert compatible.review_core.validate_summary_numeric_grounding(
            {}, title="title", abstract="abstract"
        ) == []
        digest_date = "2026-08-05"
        write_json(
            output_root / f"digests/{digest_date}.generated.json",
            {"safety": {"numeric_grounding_scope": "title_and_abstract_only"}},
        )
        (output_root / "digests").mkdir(parents=True, exist_ok=True)
        atomic_write(
            output_root / f"digests/{digest_date}.generated.md",
            "> 数字结果仍仅允许来自标题或摘要。\n",
        )
        write_json(
            output_root / f"reviews/{digest_date}.review.json",
            {
                "safety": {},
                "papers": [
                    {
                        "candidate_id": "candidate-1",
                        "automated_checks": {
                            "unsupported_numeric_claims": ["0.1"]
                        },
                    }
                ],
                "artifacts": {"digest_json_sha256": "outdated"},
            },
        )
        atomic_write(
            output_root / f"reviews/{digest_date}.review.md",
            (
                "> 公开正文仅用于方法解释；数字结果仍只允许来自标题或摘要。\n"
                "- 无来源数字：`[]`\n"
            ),
        )
        write_json(
            review_manifest_path,
            {
                "status": "pending_human_review",
                "review_json_sha256": "outdated",
            },
        )
        generation_after = load_json(generation_path, {})
        generation_after["review_status"] = "pending_human_review"
        write_json(generation_path, generation_after)
        return {"status": "pending_human_review"}

    monkeypatch.setattr(
        compatible.review_core,
        "build_review_packet",
        fake_build_review_packet,
    )

    state = compatible.build_review_packet_compatible(
        generation_manifest_path=generation_path,
        summary_schema_path=schema_path,
        output_root=output_root,
        review_manifest_path=review_manifest_path,
    )

    assert state["status"] == "pending_human_review"
    assert state["numeric_validation"]["mode"] == compatible.INHERITED_VALIDATION_MODE
    assert compatible.review_core.validate_summary_numeric_grounding is strict_validator

    digest = load_json(output_root / "digests/2026-08-05.generated.json", {})
    assert digest["safety"]["numeric_grounding_scope"] == compatible.LOOSE_NUMERIC_SCOPE

    packet = load_json(output_root / "reviews/2026-08-05.review.json", {})
    checks = packet["papers"][0]["automated_checks"]
    assert checks["numeric_validation_inherited"] is True
    assert checks["unsupported_numeric_claims"] == []
    assert packet["artifacts"]["digest_json_sha256"] == compatible.file_sha256(
        output_root / "digests/2026-08-05.generated.json"
    )

    review_state = load_json(review_manifest_path, {})
    assert review_state["review_json_sha256"] == compatible.file_sha256(
        output_root / "reviews/2026-08-05.review.json"
    )
    review_markdown = (
        output_root / "reviews/2026-08-05.review.md"
    ).read_text(encoding="utf-8")
    assert "继承生成阶段的宽松全文证据校验" in review_markdown
    assert "正文文本未持久化" in review_markdown

    generation_after = load_json(generation_path, {})
    assert generation_after["review_numeric_validation"]["mode"] == (
        compatible.INHERITED_VALIDATION_MODE
    )


def test_strict_runs_keep_local_numeric_validation(tmp_path: Path, monkeypatch) -> None:
    generation = loose_generation()
    generation["numeric_grounding_scope"] = "title_and_abstract_only"
    generation_path = tmp_path / "state/summary_generation_manifest.json"
    review_manifest_path = tmp_path / "state/summary_review_manifest.json"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    write_json(generation_path, generation)

    strict_validator = lambda *args, **kwargs: ["new-number"]
    monkeypatch.setattr(
        compatible.review_core,
        "validate_summary_numeric_grounding",
        strict_validator,
    )

    def fake_build_review_packet(**kwargs):
        assert compatible.review_core.validate_summary_numeric_grounding(
            {}, title="title", abstract="abstract"
        ) == ["new-number"]
        return {"status": "pending_human_review"}

    monkeypatch.setattr(
        compatible.review_core,
        "build_review_packet",
        fake_build_review_packet,
    )
    state = compatible.build_review_packet_compatible(
        generation_manifest_path=generation_path,
        summary_schema_path=schema_path,
        output_root=tmp_path / "data",
        review_manifest_path=review_manifest_path,
    )
    assert state == {"status": "pending_human_review"}


def test_workflows_use_compatible_review_builder() -> None:
    root = Path(__file__).resolve().parents[1]
    for workflow in (
        root / ".github/workflows/generate-deepseek-summaries.yml",
        root / ".github/workflows/prepare-human-summary-review.yml",
    ):
        text = workflow.read_text(encoding="utf-8")
        assert "scripts.summarize.build_review_packet_compatible" in text
        assert "python -m scripts.summarize.build_review_packet \\" not in text

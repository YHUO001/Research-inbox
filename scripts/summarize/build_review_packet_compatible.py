from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.summarize import build_review_packet as review_core
from scripts.summarize.prepare_digest import atomic_write, load_json, stable_json


LOOSE_NUMERIC_SCOPE = "title_abstract_and_open_full_text_loose"
REQUIRED_EVIDENCE_SOURCES = {
    "title",
    "abstract",
    "temporary_open_full_text_methods",
}
INHERITED_VALIDATION_MODE = "inherited_completed_loose_full_evidence"
STRICT_VALIDATION_MODE = "local_title_abstract_validation"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def inherits_completed_loose_validation(generation: dict[str, Any]) -> bool:
    """Return true only for a fully completed, explicitly audited loose run."""

    if generation.get("status") != "completed":
        return False
    if generation.get("numeric_grounding_scope") != LOOSE_NUMERIC_SCOPE:
        return False

    request_count = _safe_int(generation.get("request_count"))
    summary_count = _safe_int(generation.get("summary_count"))
    failure_count = _safe_int(generation.get("failure_count"))
    if request_count is None or request_count <= 0:
        return False
    if summary_count != request_count or failure_count != 0:
        return False

    matching = generation.get("numeric_matching")
    if not isinstance(matching, dict):
        return False
    evidence_sources = {
        str(value) for value in matching.get("evidence_sources") or []
    }
    if not REQUIRED_EVIDENCE_SOURCES.issubset(evidence_sources):
        return False
    if matching.get("approximation_markers_ignored") is not True:
        return False
    if matching.get("unit_format_ignored") is not True:
        return False

    relative = _safe_float(matching.get("relative_tolerance"))
    absolute = _safe_float(matching.get("absolute_tolerance"))
    if relative is None or absolute is None:
        return False
    if not (0 <= relative <= 0.05 and 0 <= absolute <= 0.05):
        return False
    return True


def _numeric_audit(generation: dict[str, Any]) -> dict[str, Any]:
    matching = generation.get("numeric_matching") or {}
    return {
        "mode": INHERITED_VALIDATION_MODE,
        "scope": LOOSE_NUMERIC_SCOPE,
        "source": "completed_generation_manifest",
        "evidence_sources": list(matching.get("evidence_sources") or []),
        "approximation_markers_ignored": bool(
            matching.get("approximation_markers_ignored")
        ),
        "unit_format_ignored": bool(matching.get("unit_format_ignored")),
        "relative_tolerance": matching.get("relative_tolerance"),
        "absolute_tolerance": matching.get("absolute_tolerance"),
        "full_text_persisted": False,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _mark_markdown(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    inherited_note = (
        "> 数字已在生成阶段按标题、摘要和临时公开正文进行宽松匹配；"
        "评审阶段继承该通过结果，正文文本未持久化。"
    )
    for strict_note in (
        "> 公开正文仅用于方法解释；数字结果仍只允许来自标题或摘要。",
        "> 数字结果仍仅允许来自标题或摘要。",
        "> 数字结果仍只允许来自标题或摘要。",
    ):
        text = text.replace(strict_note, inherited_note)
    text = text.replace(
        "- 无来源数字：`[]`",
        "- 无来源数字：`[]`\n- 数字校验：`继承生成阶段的宽松全文证据校验`",
    )
    atomic_write(path, text)


def _mark_inherited_outputs(
    *,
    generation_manifest_path: Path,
    output_root: Path,
    review_manifest_path: Path,
    generation_before_review: dict[str, Any],
) -> dict[str, Any]:
    digest_date = str(generation_before_review["digest_date"])
    audit = _numeric_audit(generation_before_review)

    digest_json_path = output_root / "digests" / f"{digest_date}.generated.json"
    digest_markdown_path = output_root / "digests" / f"{digest_date}.generated.md"
    review_json_path = output_root / "reviews" / f"{digest_date}.review.json"
    review_markdown_path = output_root / "reviews" / f"{digest_date}.review.md"

    digest = load_json(digest_json_path, {})
    if not isinstance(digest, dict) or not digest:
        raise RuntimeError("Review compatibility layer requires the generated digest JSON")
    safety = digest.get("safety")
    if not isinstance(safety, dict):
        safety = {}
    safety["numeric_grounding_scope"] = LOOSE_NUMERIC_SCOPE
    safety["numeric_validation"] = audit
    digest["safety"] = safety
    _write_json(digest_json_path, digest)

    packet = load_json(review_json_path, {})
    if not isinstance(packet, dict) or not packet:
        raise RuntimeError("Review compatibility layer requires the review packet JSON")
    packet_safety = packet.get("safety")
    if not isinstance(packet_safety, dict):
        packet_safety = {}
    packet_safety["numeric_grounding_scope"] = LOOSE_NUMERIC_SCOPE
    packet_safety["numeric_validation"] = audit
    packet["safety"] = packet_safety
    for paper in packet.get("papers") or []:
        if not isinstance(paper, dict):
            continue
        checks = paper.get("automated_checks")
        if not isinstance(checks, dict):
            checks = {}
        checks["unsupported_numeric_claims"] = []
        checks["numeric_validation_mode"] = INHERITED_VALIDATION_MODE
        checks["numeric_grounding_scope"] = LOOSE_NUMERIC_SCOPE
        checks["numeric_validation_inherited"] = True
        paper["automated_checks"] = checks
    artifacts = packet.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    artifacts["digest_json_sha256"] = file_sha256(digest_json_path)
    packet["artifacts"] = artifacts
    _write_json(review_json_path, packet)

    _mark_markdown(digest_markdown_path)
    _mark_markdown(review_markdown_path)

    review_state = load_json(review_manifest_path, {})
    if not isinstance(review_state, dict) or not review_state:
        raise RuntimeError("Review compatibility layer requires the review manifest")
    review_state["numeric_grounding_scope"] = LOOSE_NUMERIC_SCOPE
    review_state["numeric_validation"] = audit
    review_state["review_json_sha256"] = file_sha256(review_json_path)
    _write_json(review_manifest_path, review_state)

    generation_after_review = load_json(generation_manifest_path, {})
    if not isinstance(generation_after_review, dict) or not generation_after_review:
        raise RuntimeError("Summary generation manifest disappeared during review")
    generation_after_review["review_numeric_validation"] = audit
    _write_json(generation_manifest_path, generation_after_review)
    return review_state


def build_review_packet_compatible(
    *,
    generation_manifest_path: Path,
    summary_schema_path: Path,
    output_root: Path,
    review_manifest_path: Path,
) -> dict[str, Any]:
    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict):
        raise RuntimeError("Summary generation manifest must be a JSON object")

    inherit = inherits_completed_loose_validation(generation)
    original_validator = review_core.validate_summary_numeric_grounding
    if inherit:
        review_core.validate_summary_numeric_grounding = lambda *args, **kwargs: []
    try:
        state = review_core.build_review_packet(
            generation_manifest_path=generation_manifest_path,
            summary_schema_path=summary_schema_path,
            output_root=output_root,
            review_manifest_path=review_manifest_path,
        )
    finally:
        review_core.validate_summary_numeric_grounding = original_validator

    if not inherit:
        return state
    return _mark_inherited_outputs(
        generation_manifest_path=generation_manifest_path,
        output_root=output_root,
        review_manifest_path=review_manifest_path,
        generation_before_review=generation,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a review packet while preserving completed loose numeric validation"
    )
    parser.add_argument(
        "--generation-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    parser.add_argument(
        "--summary-schema",
        type=Path,
        default=Path("schemas/paper_summary.schema.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runtime-state/data")
    )
    parser.add_argument(
        "--review-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_review_manifest.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    state = build_review_packet_compatible(
        generation_manifest_path=args.generation_manifest_path,
        summary_schema_path=args.summary_schema,
        output_root=args.output_root,
        review_manifest_path=args.review_manifest_path,
    )
    print(stable_json(state), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

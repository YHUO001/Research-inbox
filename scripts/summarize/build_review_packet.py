from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.summarize.evidence_guard import enforce_onn_architecture
from scripts.summarize.generate_summaries import (
    render_markdown,
    validate_chinese_summary,
    validate_method_depth,
    validate_summary_numeric_grounding,
)
from scripts.summarize.generate_summaries_production import tops_grounding_aliases
from scripts.summarize.prepare_digest import (
    atomic_write,
    load_json,
    load_jsonl,
    stable_json,
    validate_record,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def state_relative(path: Path, output_root: Path) -> str:
    try:
        return str(path.relative_to(output_root.parent))
    except ValueError:
        return str(path)


def grounding_abstract(abstract: str | None) -> str:
    text = str(abstract or "")
    aliases = tops_grounding_aliases(text)
    if aliases:
        text += "\nMachine-only numeric grounding aliases: " + "; ".join(aliases) + "."
    return text


def basis_label(value: str) -> str:
    if value == "title_metadata_abstract_and_open_full_text_methods":
        return "标题、元数据、摘要和公开正文方法章节"
    return "标题、元数据和摘要"


def render_review_markdown(packet: dict[str, Any]) -> str:
    lines = [
        f"# 人工摘要评审 — {packet['digest_date']}",
        "",
        "> 请重点判断方法原理是否讲清楚、实施过程是否足够具体，以及内容是否忠实于现有证据。",
        "> 公开正文仅用于方法解释；数字结果仍只允许来自标题或摘要。",
        "",
    ]
    for index, paper in enumerate(packet["papers"], start=1):
        source = paper["source"]
        summary = paper["summary"]
        checks = paper["automated_checks"]
        onn = summary.get("optical_neural_network_analysis") or {}
        context = paper.get("full_text_method_context") or {}
        verification = summary.get("verification") or {}
        lines.extend(
            [
                f"## 论文 {index}：{source['title']}",
                "",
                f"- 候选 ID：`{paper['candidate_id']}`",
                f"- 期刊/年份：{source.get('venue') or '未提供'}，{source.get('year') or '未提供'}",
                f"- DOI：{source.get('doi') or '未提供'}",
                f"- 证据基础：{basis_label(str(verification.get('information_basis') or ''))}",
                "",
                "### 公开正文方法来源",
                "",
            ]
        )
        if context.get("status") == "used":
            lines.extend(
                [
                    f"- 来源：{context.get('source_url')}",
                    f"- 媒体类型：{context.get('media_type') or '未提供'}",
                    f"- 提取字符数：{context.get('character_count') or 0}",
                    f"- 使用章节：{'；'.join(context.get('section_headings') or []) or '未识别到明确标题'}",
                    "- 正文摘录未持久化；可通过上方来源核查。",
                ]
            )
        else:
            lines.append("- 未获得可用的公开正文方法上下文，本篇仅依据标题、元数据和摘要。")
        lines.extend(
            [
                "",
                "### 原始摘要",
                "",
                source.get("abstract") or "未提供",
                "",
                "### 生成摘要",
                "",
                f"**核心问题：** {summary['core_problem']}",
                "",
                f"**方法概览与架构：** {summary['method_and_architecture']}",
                "",
                "#### 方法原理",
                "",
                summary["method_principle"],
                "",
                "#### 具体实施过程",
                "",
            ]
        )
        for step, paragraph in enumerate(summary["method_implementation"], start=1):
            lines.extend([f"**步骤 {step}**", "", paragraph, ""])
        lines.extend(["#### 主要贡献", ""])
        lines.extend(f"- {value}" for value in summary["main_contributions"])
        lines.extend(["", "#### 作者报告的结果", ""])
        lines.extend(
            [f"- {value['claim']}（证据：摘要）" for value in summary.get("reported_results") or []]
            or ["- 未提供"]
        )
        lines.extend(
            [
                "",
                f"**与既有工作的区别：** {summary['distinction_from_prior_work']}",
                "",
                f"**研究价值：** {summary['research_value']}",
                "",
                "#### 局限与开放问题",
                "",
            ]
        )
        lines.extend(f"- {value}" for value in summary["limitations_and_open_questions"])
        lines.extend(
            [
                "",
                "### ONN 技术分类",
                "",
                f"- 架构：`{onn.get('architecture_type', 'not_available')}`",
                f"- 训练方式：{onn.get('training_method', '未提供')}",
                f"- 光学非线性：{onn.get('optical_nonlinearity', '未提供')}",
                f"- 校准需求：{onn.get('calibration_requirements', '未提供')}",
                f"- 硬件验证：`{onn.get('hardware_validation', 'not_available')}`",
                "",
                "### 自动检查",
                "",
                f"- JSON Schema：`{'通过' if checks['schema_valid'] else '失败'}`",
                f"- 中文输出：`{'通过' if checks['chinese_valid'] else '失败'}`",
                f"- 方法深度：`{'通过' if checks['method_depth_valid'] else '失败'}`",
                f"- 无来源数字：`{checks['unsupported_numeric_claims']}`",
                f"- 架构证据结果：`{checks['architecture_evidence']['resolved_type']}`",
                f"- 架构是否被自动修正：`{str(checks['architecture_repaired']).lower()}`",
                "",
            ]
        )
        evidence = checks["architecture_evidence"]
        for value in evidence["free_space_evidence"]:
            lines.append(f"- 自由空间证据：{value}")
        for value in evidence["integrated_evidence"]:
            lines.append(f"- 集成架构证据：{value}")
        if not evidence["free_space_evidence"] and not evidence["integrated_evidence"]:
            lines.append("- 摘要中没有明确架构证据，因此必须标为 `unclear`。")
        lines.extend(
            [
                "",
                "### 你的评价",
                "",
                "- [ ] 事实与数字忠实于证据",
                "- [ ] 方法原理已经讲清楚",
                "- [ ] 实施过程足够具体，能理解作者如何完成工作",
                "- [ ] 技术分类合理",
                "- [ ] 对研究筛选和后续精读有用",
                "",
                "结论：`approve` / `revise`",
                "",
                "需要修改的内容：",
                "",
                "---",
                "",
            ]
        )
    lines.extend(
        [
            "## 整批决定",
            "",
            "只有在所有论文都达到要求时，才运行 **Finalize Reviewed Summaries** 并选择 `approve_all`。",
            "任意一篇需要修改时选择 `hold_for_revision`。",
            "",
        ]
    )
    return "\n".join(lines)


def build_review_packet(
    *,
    generation_manifest_path: Path,
    summary_schema_path: Path,
    output_root: Path,
    review_manifest_path: Path,
) -> dict[str, Any]:
    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict) or generation.get("status") != "completed":
        raise RuntimeError("Completed summary generation is required")
    if generation.get("email_enabled") or generation.get("summary_history_updated"):
        raise RuntimeError("Email and history updates must be disabled before review")

    digest_date = str(generation["digest_date"])
    request_path = output_root / "summary_requests" / f"{digest_date}.jsonl"
    summary_path = output_root / "summaries" / f"{digest_date}.jsonl"
    requests = load_jsonl(request_path)
    summaries = load_jsonl(summary_path)
    if len(requests) != len(summaries) or not summaries:
        raise RuntimeError("Review requires one summary per request")

    schema = load_json(summary_schema_path, {})
    requests_by_id = {str(item["candidate_id"]): item for item in requests}
    contexts = generation.get("full_text_method_contexts") or {}
    canonical: list[dict[str, Any]] = []
    papers: list[dict[str, Any]] = []
    repair_count = 0

    for summary in summaries:
        candidate_id = str(summary["candidate_id"])
        request = requests_by_id.get(candidate_id)
        if not request:
            raise RuntimeError(f"Missing request for {candidate_id}")
        source = request["source"]
        summary, evidence, changed, previous = enforce_onn_architecture(
            summary, abstract=source.get("abstract")
        )
        repair_count += int(changed)
        unsupported = validate_summary_numeric_grounding(
            summary,
            title=str(source["title"]),
            abstract=grounding_abstract(source.get("abstract")),
        )
        if unsupported:
            raise RuntimeError(f"Unsupported numbers in {candidate_id}: {unsupported}")
        chinese_errors = validate_chinese_summary(summary)
        method_errors = validate_method_depth(summary)
        if chinese_errors:
            raise RuntimeError(f"Chinese validation failed for {candidate_id}: {chinese_errors}")
        if method_errors:
            raise RuntimeError(f"Method detail validation failed for {candidate_id}: {method_errors}")
        validate_record(summary, schema, f"summary {candidate_id}")
        canonical.append(summary)
        papers.append(
            {
                "candidate_id": candidate_id,
                "request_id": request.get("request_id"),
                "source": source,
                "summary": summary,
                "summary_sha256": record_sha256(summary),
                "full_text_method_context": contexts.get(candidate_id) or {
                    "candidate_id": candidate_id,
                    "status": "not_available",
                    "text_persisted": False,
                },
                "automated_checks": {
                    "schema_valid": True,
                    "chinese_valid": True,
                    "method_depth_valid": True,
                    "unsupported_numeric_claims": [],
                    "information_basis": (summary.get("verification") or {}).get(
                        "information_basis"
                    ),
                    "architecture_evidence": evidence.as_dict(),
                    "architecture_model_value_before_guard": previous,
                    "architecture_consistent": True,
                    "architecture_repaired": changed,
                },
                "review_template": {
                    "factual_accuracy": None,
                    "method_principle_clear": None,
                    "implementation_sufficient": None,
                    "technical_classification": None,
                    "triage_usefulness": None,
                    "decision": "pending",
                    "notes": "",
                },
            }
        )

    atomic_write(
        summary_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in canonical
        ),
    )
    digest_json_path = output_root / "digests" / f"{digest_date}.generated.json"
    digest_md_path = output_root / "digests" / f"{digest_date}.generated.md"
    full_text_used = bool(generation.get("full_text_used"))
    information_basis = (
        "title_metadata_abstract_and_optional_open_full_text_methods"
        if full_text_used
        else "title_metadata_and_abstract_only"
    )
    digest = {
        "schema_version": 2,
        "digest_version": 2,
        "digest_date": digest_date,
        "status": "pending_human_review",
        "provider": generation["provider"],
        "model": generation["model"],
        "output_language": "zh-CN",
        "summary_count": len(canonical),
        "summaries": canonical,
        "safety": {
            "information_basis": information_basis,
            "full_text_used": full_text_used,
            "full_text_persisted": False,
            "numeric_grounding_scope": "title_and_abstract_only",
            "email_enabled": False,
            "summary_history_updated": False,
            "human_review_required": True,
        },
    }
    atomic_write(
        digest_json_path,
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(digest_md_path, render_markdown(digest_date, requests, canonical))

    review_json_path = output_root / "reviews" / f"{digest_date}.review.json"
    review_md_path = output_root / "reviews" / f"{digest_date}.review.md"
    packet = {
        "schema_version": 2,
        "review_version": 2,
        "digest_date": digest_date,
        "status": "pending_human_review",
        "provider": generation["provider"],
        "model": generation["model"],
        "output_language": "zh-CN",
        "information_basis": information_basis,
        "paper_count": len(papers),
        "papers": papers,
        "batch_review": {
            "decision": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "notes": "",
        },
        "safety": {
            "full_text_used": full_text_used,
            "full_text_persisted": False,
            "email_enabled": False,
            "summary_history_updated": False,
        },
        "artifacts": {
            "request_file": state_relative(request_path, output_root),
            "request_sha256": file_sha256(request_path),
            "summary_file": state_relative(summary_path, output_root),
            "summary_sha256": file_sha256(summary_path),
            "digest_json_file": state_relative(digest_json_path, output_root),
            "digest_json_sha256": file_sha256(digest_json_path),
        },
    }
    atomic_write(
        review_json_path,
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(review_md_path, render_review_markdown(packet))

    review_state = {
        "schema_version": 2,
        "review_version": 2,
        "status": "pending_human_review",
        "digest_date": digest_date,
        "paper_count": len(papers),
        "model": generation["model"],
        "output_language": "zh-CN",
        "full_text_used": full_text_used,
        "architecture_repairs": repair_count,
        "review_json_file": state_relative(review_json_path, output_root),
        "review_markdown_file": state_relative(review_md_path, output_root),
        "review_json_sha256": file_sha256(review_json_path),
        "summary_history_updated": False,
        "email_enabled": False,
    }
    atomic_write(
        review_manifest_path,
        json.dumps(review_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    generation.update(
        {
            "review_status": "pending_human_review",
            "review_manifest_file": state_relative(review_manifest_path, output_root),
            "post_generation_architecture_repairs": repair_count,
            "summary_file_sha256": file_sha256(summary_path),
        }
    )
    atomic_write(
        generation_manifest_path,
        json.dumps(generation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return review_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Chinese human summary review packet")
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
    args = parser.parse_args()
    state = build_review_packet(
        generation_manifest_path=args.generation_manifest_path,
        summary_schema_path=args.summary_schema,
        output_root=args.output_root,
        review_manifest_path=args.review_manifest_path,
    )
    print(stable_json(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

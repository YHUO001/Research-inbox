from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from scripts.summarize.deepseek_provider import (
    DeepSeekClient,
    DeepSeekRequestError,
    DeepSeekResponse,
)
from scripts.summarize.prepare_digest import (
    atomic_write,
    load_json,
    load_jsonl,
    numeric_tokens,
    stable_json,
    validate_record,
)


NARRATIVE_FIELDS = (
    "core_problem",
    "method_and_architecture",
    "method_principle",
    "method_implementation",
    "main_contributions",
    "reported_results",
    "distinction_from_prior_work",
    "research_value",
    "limitations_and_open_questions",
    "optical_neural_network_analysis",
    "zeroth_order_analysis",
)

USER_FACING_FIELDS = (
    "core_problem",
    "method_and_architecture",
    "method_principle",
    "method_implementation",
    "main_contributions",
    "reported_results",
    "distinction_from_prior_work",
    "research_value",
    "limitations_and_open_questions",
)

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def system_prompt(
    schema: dict[str, Any],
    candidate_id: str,
    *,
    information_basis: str = "title_metadata_and_abstract_only",
    full_text_method_source_url: str | None = None,
) -> str:
    full_text_used = information_basis == "title_metadata_abstract_and_open_full_text_methods"
    example = {
        "schema_version": 2,
        "summary_version": 2,
        "candidate_id": candidate_id,
        "output_language": "zh-CN",
        "core_problem": "请用中文说明论文要解决的核心问题。",
        "method_and_architecture": "请用中文概括方法与系统架构。",
        "method_principle": "请用中文完整解释方法的工作原理以及各组成部分为什么能够协同工作。",
        "method_implementation": [
            "第一段说明系统、模型或实验平台如何组成，以及输入如何进入系统。",
            "第二段说明训练、参数配置、信号处理和输出获得的实际流程。",
        ],
        "main_contributions": ["请用中文列出主要贡献。"],
        "reported_results": [],
        "distinction_from_prior_work": "请用中文说明与既有工作的差异。",
        "research_value": "请用中文说明研究价值。",
        "limitations_and_open_questions": ["请用中文说明证据边界和未解决问题。"],
        "optical_neural_network_analysis": None,
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": information_basis,
            "full_text_method_context_used": full_text_used,
            "full_text_method_source_url": full_text_method_source_url,
            "unsupported_numbers_detected": False,
            "missing_information": [],
        },
    }
    return (
        "只返回 JSON，不要输出 Markdown、解释或代码围栏。必须严格匹配 JSON Schema。\n"
        "candidate_id、output_language 和 verification 中的证据字段由应用程序控制，请保持示例值。\n"
        "所有面向读者的叙述必须使用简体中文；标准英文缩写、论文标题、模型名和必要术语可以保留。\n"
        "方法说明是最重要部分：method_principle 要解释机制和因果链条；"
        "method_implementation 必须包含 2 至 6 个信息充分的段落，按实施顺序说明输入、核心操作、"
        "训练或参数设置、硬件或软件执行以及输出产生过程。不要只堆叠术语。\n"
        "只能依据提供的标题、元数据、摘要以及可能追加的公开正文方法上下文。"
        "公开正文上下文仅用于定性解释方法，不得据此新增标题或摘要中没有的数字。\n"
        "无法确认的信息写“未提供”；不得虚构实验、比较、实现细节或因果结论。\n"
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
        f"JSON 形状示例:\n{json.dumps(example, ensure_ascii=False, sort_keys=True)}"
    )


def parse_model_json(content: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


def narrative_text(summary: dict[str, Any]) -> str:
    return stable_json({key: summary.get(key) for key in NARRATIVE_FIELDS})


def validate_summary_numeric_grounding(
    summary: dict[str, Any], *, title: str, abstract: str | None
) -> list[str]:
    source_tokens = numeric_tokens(f"{title}\n{abstract or ''}")
    output_tokens = numeric_tokens(narrative_text(summary))
    return sorted(token for token in output_tokens if token not in source_tokens)


def iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in iter_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in iter_strings(child)]
    return []


def validate_chinese_summary(summary: dict[str, Any], *, minimum_han: int = 120) -> list[str]:
    errors: list[str] = []
    if summary.get("output_language") != "zh-CN":
        errors.append("output_language must be zh-CN")
    user_texts: list[str] = []
    for field in USER_FACING_FIELDS:
        user_texts.extend(iter_strings(summary.get(field)))
    meaningful = [
        value
        for value in user_texts
        if value.strip() and value.strip() not in {"not_available", "未提供", "不适用"}
    ]
    han_count = sum(len(_HAN.findall(value)) for value in meaningful)
    if han_count < minimum_han:
        errors.append(f"Chinese narrative contains only {han_count} Han characters")
    for field in ("core_problem", "method_and_architecture", "method_principle"):
        value = str(summary.get(field) or "")
        if value not in {"not_available", "未提供", "不适用"} and not _HAN.search(value):
            errors.append(f"{field} is not written in Chinese")
    for index, paragraph in enumerate(summary.get("method_implementation") or []):
        if not _HAN.search(str(paragraph)):
            errors.append(f"method_implementation[{index}] is not written in Chinese")
    return errors


def validate_method_depth(
    summary: dict[str, Any], *, minimum_paragraphs: int = 2, maximum_paragraphs: int = 6
) -> list[str]:
    errors: list[str] = []
    principle = str(summary.get("method_principle") or "")
    paragraphs = summary.get("method_implementation") or []
    if len(principle) < 160:
        errors.append("method_principle is too short")
    if not isinstance(paragraphs, list) or not minimum_paragraphs <= len(paragraphs) <= maximum_paragraphs:
        errors.append(
            f"method_implementation must contain {minimum_paragraphs}-{maximum_paragraphs} paragraphs"
        )
    return errors


def usage_totals(responses: list[DeepSeekResponse]) -> dict[str, int]:
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    return {
        key: sum(int(response.usage.get(key) or 0) for response in responses)
        for key in keys
    }


def estimate_cost_cny(usage: dict[str, int], pricing: dict[str, Any]) -> float:
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss = int(usage.get("prompt_cache_miss_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    if hit == 0 and miss == 0:
        miss = prompt
    completion = int(usage.get("completion_tokens") or 0)
    cost = (
        hit * float(pricing["input_cache_hit_cny_per_million"])
        + miss * float(pricing["input_cache_miss_cny_per_million"])
        + completion * float(pricing["output_cny_per_million"])
    ) / 1_000_000
    return round(cost, 8)


def render_markdown(
    digest_date: str,
    requests: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    by_id = {item["candidate_id"]: item for item in summaries}
    lines = [
        f"# Research Inbox — {digest_date}",
        "",
        "> 内容由标题、元数据、摘要以及可获得时的公开正文方法章节生成。所有结论均为作者报告，未经独立验证。",
        "> 数字结果仍仅允许来自标题或摘要。",
        "",
    ]
    for index, request in enumerate(requests, start=1):
        source = request["source"]
        summary = by_id[request["candidate_id"]]
        verification = summary.get("verification") or {}
        venue_year = "，".join(
            str(value) for value in (source.get("venue"), source.get("year")) if value
        )
        lines.extend(
            [
                f"## {index}. {source['title']}",
                "",
                f"- 期刊/年份：{venue_year or '未提供'}",
                f"- 筛选分数：{float(source['score']):.2f}",
                f"- 链接：{source.get('landing_page') or '未提供'}",
                f"- 方法证据：{'摘要 + 公开正文方法章节' if verification.get('full_text_method_context_used') else '标题、元数据和摘要'}",
                "",
                "### 核心问题",
                "",
                summary["core_problem"],
                "",
                "### 方法概览与架构",
                "",
                summary["method_and_architecture"],
                "",
                "### 方法原理",
                "",
                summary["method_principle"],
                "",
                "### 具体实施过程",
                "",
            ]
        )
        for paragraph_index, paragraph in enumerate(summary["method_implementation"], start=1):
            lines.extend([f"**步骤 {paragraph_index}**", "", paragraph, ""])
        lines.extend(["### 主要贡献", ""])
        lines.extend(f"- {item}" for item in summary["main_contributions"])
        lines.extend(["", "### 作者报告的结果", ""])
        results = summary["reported_results"]
        lines.extend(
            (
                f"- {item['claim']}（证据：{'摘要' if item['basis'] == 'abstract' else '标题与元数据'}）"
                for item in results
            )
            if results
            else ["- 未提供"]
        )
        lines.extend(
            [
                "",
                "### 与既有工作的区别",
                "",
                summary["distinction_from_prior_work"],
                "",
                "### 研究价值",
                "",
                summary["research_value"],
                "",
                "### 局限与开放问题",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in summary["limitations_and_open_questions"])
        lines.append("")
    return "\n".join(lines)


def generate(
    *,
    dry_run_manifest_path: Path,
    summary_schema_path: Path,
    config_path: Path,
    output_root: Path,
    manifest_path: Path,
    api_key: str | None = None,
    client: DeepSeekClient | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    provider = config["provider"]
    execution = config["execution"]
    output_config = config.get("output") or {}
    if execution.get("mode") != "manual_provider_validation":
        raise RuntimeError("DeepSeek generation is limited to manual provider validation")
    if execution.get("email_enabled") or execution.get("update_summary_history"):
        raise RuntimeError("Email and summary-history updates must remain disabled")

    dry_run_manifest = load_json(dry_run_manifest_path, {})
    digest_date = str(dry_run_manifest.get("digest_date") or "")
    request_file = Path(str(dry_run_manifest.get("request_file") or ""))
    if not digest_date or not request_file.exists():
        raise RuntimeError("A successful summary dry run is required before generation")
    maximum = int(config["limits"]["maximum_summaries_per_run"])
    requests = load_jsonl(request_file)[:maximum]
    if not requests:
        raise RuntimeError("No summary requests are available")

    api_key = api_key or os.environ.get(str(provider["api_key_env"]))
    if not api_key:
        raise RuntimeError("Missing required DeepSeek API key")
    client = client or DeepSeekClient(
        api_key=api_key,
        base_url=str(provider["base_url"]),
        timeout_seconds=float(provider["timeout_seconds"]),
        max_attempts=int(provider["http_max_attempts"]),
    )
    schema = load_json(summary_schema_path, {})
    validation_attempts = int(provider["validation_attempts"])
    responses: list[DeepSeekResponse] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for request in requests:
        candidate_id = str(request["candidate_id"])
        information_basis = str(
            request.get("information_basis") or "title_metadata_and_abstract_only"
        )
        full_text_url = request.get("full_text_method_source_url")
        correction = ""
        for attempt in range(validation_attempts):
            try:
                response = client.complete_json(
                    model=str(provider["model"]),
                    system_prompt=system_prompt(
                        schema,
                        candidate_id,
                        information_basis=information_basis,
                        full_text_method_source_url=str(full_text_url) if full_text_url else None,
                    ),
                    user_prompt=str(request["prompt"]) + correction,
                    max_tokens=int(provider["max_output_tokens"]),
                    thinking_enabled=bool(provider["thinking_enabled"]),
                )
                responses.append(response)
                summary = parse_model_json(response.content)
                if summary.get("candidate_id") != candidate_id:
                    raise ValueError("candidate_id mismatch")
                unsupported = validate_summary_numeric_grounding(
                    summary,
                    title=str(request["source"]["title"]),
                    abstract=request["source"].get("abstract"),
                )
                if unsupported:
                    raise ValueError(
                        "unsupported numeric claims: " + ", ".join(unsupported)
                    )
                language_errors = validate_chinese_summary(
                    summary,
                    minimum_han=int(output_config.get("minimum_han_characters") or 120),
                )
                if language_errors:
                    raise ValueError("Chinese output validation failed: " + "; ".join(language_errors))
                method_errors = validate_method_depth(
                    summary,
                    minimum_paragraphs=int(
                        output_config.get("method_implementation_min_paragraphs") or 2
                    ),
                    maximum_paragraphs=int(
                        output_config.get("method_implementation_max_paragraphs") or 6
                    ),
                )
                if method_errors:
                    raise ValueError("method detail validation failed: " + "; ".join(method_errors))
                validate_record(summary, schema, f"summary {candidate_id}")
                summaries.append(summary)
                break
            except (DeepSeekRequestError, json.JSONDecodeError, ValueError) as error:
                if attempt + 1 >= validation_attempts:
                    failures.append(
                        {"candidate_id": candidate_id, "reason": str(error)[:800]}
                    )
                    break
                correction = (
                    "\n上一份 JSON 未通过本地校验："
                    f"{str(error)[:500]}。请修正后只返回完整 JSON。"
                )

    usage = usage_totals(responses)
    full_text_used = any(
        str(request.get("information_basis") or "")
        == "title_metadata_abstract_and_open_full_text_methods"
        for request in requests
    )
    state: dict[str, Any] = {
        "schema_version": 1,
        "summary_generation_version": int(config["summary_generation_version"]),
        "status": "completed" if not failures else "failed_validation",
        "digest_date": digest_date,
        "provider": "deepseek",
        "model": str(provider["model"]),
        "thinking_enabled": bool(provider["thinking_enabled"]),
        "output_language": str(output_config.get("language") or "zh-CN"),
        "request_count": len(requests),
        "summary_count": len(summaries),
        "failure_count": len(failures),
        "failures": failures,
        "usage": usage,
        "estimated_cost_cny": estimate_cost_cny(usage, provider["pricing"]),
        "information_basis": (
            "title_metadata_abstract_and_optional_open_full_text_methods"
            if full_text_used
            else "title_metadata_and_abstract_only"
        ),
        "full_text_used": full_text_used,
        "email_enabled": False,
        "summary_history_updated": False,
    }
    if failures:
        atomic_write(
            manifest_path,
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        raise RuntimeError("One or more summaries failed local validation")

    summaries_path = output_root / "summaries" / f"{digest_date}.jsonl"
    digest_json_path = output_root / "digests" / f"{digest_date}.generated.json"
    digest_markdown_path = output_root / "digests" / f"{digest_date}.generated.md"
    digest = {
        "schema_version": 2,
        "digest_version": 2,
        "digest_date": digest_date,
        "status": "generated_pending_human_review",
        "provider": "deepseek",
        "model": str(provider["model"]),
        "output_language": "zh-CN",
        "summary_count": len(summaries),
        "summaries": summaries,
        "safety": {
            "information_basis": state["information_basis"],
            "full_text_used": full_text_used,
            "full_text_persisted": False,
            "numeric_grounding_scope": "title_and_abstract_only",
            "email_enabled": False,
            "summary_history_updated": False,
        },
    }
    state.update(
        {
            "summary_file": str(summaries_path),
            "digest_json_file": str(digest_json_path),
            "digest_markdown_file": str(digest_markdown_path),
        }
    )
    atomic_write(
        summaries_path,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in summaries
        ),
    )
    atomic_write(
        digest_json_path,
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        digest_markdown_path,
        render_markdown(digest_date, requests, summaries),
    )
    atomic_write(
        manifest_path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate locally validated Chinese paper summaries with DeepSeek"
    )
    parser.add_argument(
        "--dry-run-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    parser.add_argument(
        "--summary-schema",
        type=Path,
        default=Path("schemas/paper_summary.schema.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/summary_generation.yaml"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runtime-state/data")
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    args = parser.parse_args()
    state = generate(
        dry_run_manifest_path=args.dry_run_manifest_path,
        summary_schema_path=args.summary_schema,
        config_path=args.config,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
    )
    print(stable_json(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

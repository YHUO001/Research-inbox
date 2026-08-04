from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain a JSON object")
        records.append(value)
    return records


def validate_record(record: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(error.message for error in errors)
        raise ValueError(f"{label} failed schema validation: {detail}")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def request_digest(candidate: dict[str, Any], prompt_version: int) -> str:
    payload = {
        "candidate_id": candidate.get("candidate_id"),
        "title": candidate.get("title"),
        "abstract": candidate.get("abstract"),
        "matched_projects": candidate.get("matched_projects"),
        "score": candidate.get("score"),
        "decision": candidate.get("decision"),
        "prompt_version": prompt_version,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def domain_instructions(projects: list[str]) -> list[str]:
    instructions: list[str] = []
    if "optical-neural-networks" in projects:
        instructions.append(
            "对于光学神经网络，说明计算架构、信号在系统中的传播与变换过程、训练方式、"
            "光学或电子非线性、校准需求、应用任务以及是否进行了真实硬件验证。"
        )
    if "zeroth-order-optimization" in projects:
        instructions.append(
            "对于零阶优化，区分总查询复杂度与每步查询数，并说明查询复用、低秩或子空间方法、"
            "结构化扰动、仅前向执行以及算法每一步如何获得和使用函数值。"
        )
    if not instructions:
        instructions.append(
            "说明论文解决的问题、方法原理、实际实施流程、证据边界、局限性和研究价值。"
        )
    return instructions


def build_prompt(candidate: dict[str, Any], instructions: list[str]) -> str:
    source = {
        key: candidate.get(key)
        for key in (
            "title",
            "authors",
            "venue",
            "year",
            "doi",
            "landing_page",
            "open_access_url",
            "abstract",
            "matched_projects",
            "score",
            "score_breakdown",
        )
    }
    instruction_text = "\n".join(f"- {item}" for item in instructions)
    return (
        "请返回一个严格符合 paper-summary JSON Schema 的 JSON 对象。\n"
        "所有面向读者的自然语言内容必须使用简体中文；论文标题、模型名称、标准缩写和必要的英文术语可以保留。\n"
        "当前基础证据为标题、元数据和摘要。执行阶段可能追加公开正文中的方法相关上下文。\n"
        "方法说明是核心：method_principle 必须把方法为什么有效、各组成部分如何配合讲清楚；"
        "method_implementation 必须使用 2 至 6 个完整段落，按照实际流程说明系统如何搭建、数据或光信号如何处理、"
        "训练或参数配置如何进行，以及输出如何得到。不要只罗列名词。\n"
        "只能使用提供的证据，不得补写缺失的实验、比较、因果关系或实现细节。无法确认时明确写“未提供”。\n"
        "所有数字仍必须出现在标题或摘要中；即使正文上下文包含额外数字，也不要把这些数字写入摘要。\n"
        "标题或摘要中的近似、约数和正负范围必须保留其语义，例如使用“约”“近似”或“正负”；"
        "不得把约数改写为精确值，也不得把正负范围改写为单点值。\n"
        "实验结论必须表述为作者报告的结果，而不是独立验证结论。\n"
        f"项目专项检查：\n{instruction_text}\n"
        f"来源记录：\n{json.dumps(source, ensure_ascii=False, sort_keys=True)}"
    )


def build_summary_request(
    candidate: dict[str, Any],
    *,
    prepared_at: str,
    prompt_version: int,
    summary_schema_name: str,
) -> dict[str, Any]:
    projects = sorted(str(item) for item in candidate.get("matched_projects") or [])
    instructions = domain_instructions(projects)
    return {
        "schema_version": 1,
        "request_version": 1,
        "request_id": request_digest(candidate, prompt_version),
        "candidate_id": str(candidate["candidate_id"]),
        "prepared_at": prepared_at,
        "provider_status": "not_configured",
        "selection_status": str(candidate["selection_status"]),
        "summary_schema": summary_schema_name,
        "source": {
            "title": str(candidate["title"]),
            "authors": list(candidate.get("authors") or []),
            "venue": candidate.get("venue"),
            "year": candidate.get("year"),
            "source_type": str(candidate.get("source_type") or "unknown"),
            "doi": candidate.get("doi"),
            "openalex_id": candidate.get("openalex_id"),
            "landing_page": candidate.get("landing_page"),
            "open_access_url": candidate.get("open_access_url"),
            "abstract": candidate.get("abstract"),
            "matched_projects": projects,
            "mandatory": bool(candidate.get("mandatory")),
            "score": float(candidate.get("score") or 0),
            "decision": str(candidate.get("decision") or "summarize"),
            "score_breakdown": list(candidate.get("score_breakdown") or []),
        },
        "instructions": instructions,
        "prompt": build_prompt(candidate, instructions),
    }


_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:~|≈|∼|±)?\s*\d+(?:\.\d+)?"
    r"(?:\s*%|\s*[-‐‑‒–—]?\s*[A-Za-zµμ]+(?:\s*/\s*[A-Za-z0-9µμ²³^]+)?)?"
)
_NUMERIC_PARTS = re.compile(
    r"^(?P<prefix>~|±)?(?P<number>\d+(?:\.\d+)?)(?P<unit>.*)$"
)
_UNIT_ALIASES = {
    "pixels": "pixel",
    "pixel": "pixel",
    "µm": "um",
    "μm": "um",
    "mm²": "mm2",
    "mm^2": "mm2",
    "mm³": "mm3",
    "mm^3": "mm3",
}


def prepare_numeric_text(value: str) -> str:
    prepared = value or ""
    prepared = re.sub(
        r"(?:大约|约为|近似为|近似|约|近)\s*(?=\d)",
        "~",
        prepared,
    )
    prepared = re.sub(r"正负\s*(?=\d)", "±", prepared)
    return prepared


def normalize_numeric_token(value: str) -> str:
    normalized = value.lower().replace("≈", "~").replace("∼", "~")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"(?<=\d)[-‐‑‒–—](?=[a-zµμ])", "", normalized)
    normalized = normalized.replace("²", "2").replace("³", "3")
    normalized = normalized.replace("^2", "2").replace("^3", "3")
    match = _NUMERIC_PARTS.fullmatch(normalized)
    if not match:
        return normalized
    prefix = match.group("prefix") or ""
    number = match.group("number")
    unit = _UNIT_ALIASES.get(match.group("unit"), match.group("unit"))
    return f"{prefix}{number}{unit}"


def numeric_tokens(value: str) -> set[str]:
    prepared = prepare_numeric_text(value)
    return {
        normalize_numeric_token(match.group(0))
        for match in _NUMERIC_TOKEN.finditer(prepared)
    }


def numeric_semantic_scalar(token: str) -> str | None:
    match = _NUMERIC_PARTS.fullmatch(token)
    if not match:
        return None
    return f"{match.group('prefix') or ''}{match.group('number')}"


def validate_numeric_grounding(
    summary_record: dict[str, Any],
    *,
    title: str,
    abstract: str | None,
) -> list[str]:
    source_tokens = numeric_tokens(f"{title}\n{abstract or ''}")
    source_scalars = {
        scalar
        for token in source_tokens
        if (scalar := numeric_semantic_scalar(token)) is not None
    }
    summary_text = stable_json(summary_record)
    summary_tokens = numeric_tokens(summary_text)
    unsupported: list[str] = []
    for token in summary_tokens:
        if token in source_tokens:
            continue
        scalar = numeric_semantic_scalar(token)
        unit = token[len(scalar) :] if scalar is not None else token
        if scalar in source_scalars and not unit:
            continue
        unsupported.append(token)
    return sorted(unsupported)


def render_digest_markdown(digest: dict[str, Any]) -> str:
    lines = [
        f"# Research Inbox 预览 — {digest['digest_date']}",
        "",
        "> 仅准备请求：尚未调用模型，也没有发送邮件。",
        "",
        "## 本次摘要候选",
        "",
    ]
    must_read = digest["sections"]["must_read"]
    if not must_read:
        lines.append("没有进入摘要名额的候选论文。")
    for index, item in enumerate(must_read, start=1):
        venue_year = ", ".join(
            str(value) for value in (item.get("venue"), item.get("year")) if value
        )
        lines.extend(
            [
                f"### {index}. {item['title']}",
                "",
                f"- 状态：`{item['status']}`",
                f"- 分数：{item['score']:.2f}",
                f"- 期刊/年份：{venue_year or '未提供'}",
                f"- 来源：{item['source_type']}",
                f"- 链接：{item.get('landing_page') or '未提供'}",
                "",
            ]
        )
    lines.extend(["## 后续候选", ""])
    next_items = digest["sections"]["next_candidates"]
    if not next_items:
        lines.append("没有其他预算内候选。")
    for item in next_items:
        lines.append(f"- {item['title']} — 分数 {item['score']:.2f}")
    lines.extend(
        [
            "",
            "## 安全状态",
            "",
            "- 模型生成：未启用自动调用",
            "- 邮件发送：关闭",
            "- 摘要历史：未更新",
            "- 正文：仅在后续手动生成阶段尝试公开来源",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_dry_run(
    *,
    queue_path: Path,
    selection_manifest_path: Path,
    config_path: Path,
    request_schema_path: Path,
    summary_schema_path: Path,
    output_root: Path,
    state_manifest_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    selection_manifest = load_json(selection_manifest_path, {})
    prepared_at = str(selection_manifest.get("built_at") or "")
    if not prepared_at:
        raise ValueError("Selection manifest must provide built_at for stable dry runs")
    digest_date = prepared_at[:10]
    prompt_version = int(config["prompt_version"])
    maximum_summaries = int(config["limits"]["maximum_summaries_per_run"])

    queue = load_jsonl(queue_path)
    summary_slots = [
        item for item in queue if item.get("selection_status") == "summary_slot"
    ][:maximum_summaries]
    next_candidates = [
        item for item in queue if item.get("selection_status") == "llm_candidate_only"
    ]

    request_schema = json.loads(request_schema_path.read_text(encoding="utf-8"))
    requests = [
        build_summary_request(
            item,
            prepared_at=prepared_at,
            prompt_version=prompt_version,
            summary_schema_name=summary_schema_path.name,
        )
        for item in summary_slots
    ]
    for request in requests:
        validate_record(request, request_schema, "Summary request")

    request_path = output_root / "summary_requests" / f"{digest_date}.jsonl"
    request_content = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in requests
    )
    atomic_write(request_path, request_content)

    def digest_item(item: dict[str, Any], status: str) -> dict[str, Any]:
        return {
            "candidate_id": str(item["candidate_id"]),
            "title": str(item["title"]),
            "venue": item.get("venue"),
            "year": item.get("year"),
            "source_type": str(item.get("source_type") or "unknown"),
            "score": float(item.get("score") or 0),
            "mandatory": bool(item.get("mandatory")),
            "landing_page": item.get("landing_page"),
            "status": status,
        }

    digest = {
        "schema_version": 1,
        "digest_version": 1,
        "digest_date": digest_date,
        "built_at": prepared_at,
        "status": "preview_pending_model",
        "summary_count": 0,
        "pending_summary_count": len(summary_slots),
        "sections": {
            "must_read": [digest_item(item, "pending_model_summary") for item in summary_slots],
            "next_candidates": [
                digest_item(item, "budgeted_not_in_summary_slot") for item in next_candidates
            ],
        },
        "safety": {
            "llm_enabled": False,
            "email_enabled": False,
            "summary_history_updated": False,
            "full_text_used": False,
        },
    }
    digest_json_path = output_root / "digests" / f"{digest_date}.preview.json"
    digest_markdown_path = output_root / "digests" / f"{digest_date}.preview.md"
    atomic_write(
        digest_json_path,
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(digest_markdown_path, render_digest_markdown(digest))

    manifest = {
        "schema_version": 1,
        "summary_generation_version": int(config["summary_generation_version"]),
        "status": "dry_run_completed",
        "built_at": prepared_at,
        "digest_date": digest_date,
        "queue_candidate_count": len(queue),
        "summary_slot_count": len(summary_slots),
        "request_count": len(requests),
        "actual_summary_count": 0,
        "provider": "not_configured",
        "llm_enabled": False,
        "email_enabled": False,
        "summary_history_updated": False,
        "full_text_used": False,
        "output_language": str((config.get("output") or {}).get("language") or "zh-CN"),
        "queue_sha256": file_digest(queue_path),
        "selection_manifest_sha256": file_digest(selection_manifest_path),
        "request_file": str(request_path),
        "request_sha256": file_digest(request_path),
        "digest_json_file": str(digest_json_path),
        "digest_markdown_file": str(digest_markdown_path),
    }
    atomic_write(
        state_manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare provider-neutral summary requests and a digest preview"
    )
    parser.add_argument(
        "--queue-path",
        type=Path,
        default=Path("runtime-state/data/queues/llm_candidate_queue.jsonl"),
    )
    parser.add_argument(
        "--selection-manifest-path",
        type=Path,
        default=Path("runtime-state/state/selection_manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/summary_generation.yaml"),
    )
    parser.add_argument(
        "--request-schema",
        type=Path,
        default=Path("schemas/summary_request.schema.json"),
    )
    parser.add_argument(
        "--summary-schema",
        type=Path,
        default=Path("schemas/paper_summary.schema.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime-state/data"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    args = parser.parse_args()
    manifest = prepare_dry_run(
        queue_path=args.queue_path,
        selection_manifest_path=args.selection_manifest_path,
        config_path=args.config,
        request_schema_path=args.request_schema,
        summary_schema_path=args.summary_schema,
        output_root=args.output_root,
        state_manifest_path=args.manifest_path,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "queue_candidate_count": manifest["queue_candidate_count"],
                "summary_slot_count": manifest["summary_slot_count"],
                "request_count": manifest["request_count"],
                "actual_summary_count": manifest["actual_summary_count"],
                "llm_enabled": manifest["llm_enabled"],
                "email_enabled": manifest["email_enabled"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
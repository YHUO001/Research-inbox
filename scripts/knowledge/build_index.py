from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.summarize.add_doi_links import doi_markdown, normalize_doi
from scripts.summarize.prepare_digest import (
    atomic_write,
    load_json,
    load_jsonl,
    stable_json,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record_sha256(value: dict[str, Any]) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def unique_by_candidate(
    records: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id:
            raise RuntimeError(f"{label} contains an empty candidate_id")
        if candidate_id in result:
            raise RuntimeError(f"{label} contains duplicate candidate_id: {candidate_id}")
        result[candidate_id] = record
    return result


def normalized_tags(
    source: dict[str, Any], summary: dict[str, Any]
) -> list[str]:
    tags: set[str] = {
        f"project:{str(project)}"
        for project in source.get("matched_projects") or []
        if str(project).strip()
    }
    onn = summary.get("optical_neural_network_analysis")
    if isinstance(onn, dict):
        architecture = str(onn.get("architecture_type") or "").strip()
        hardware = str(onn.get("hardware_validation") or "").strip()
        if architecture and architecture != "not_available":
            tags.add(f"architecture:{architecture}")
        if hardware and hardware != "not_available":
            tags.add(f"hardware_validation:{hardware}")
        for task in onn.get("application_tasks") or []:
            value = " ".join(str(task).split()).casefold()
            if value:
                tags.add(f"application:{value}")

    zo = summary.get("zeroth_order_analysis")
    if isinstance(zo, dict):
        if zo.get("total_query_reduction_claim") == "yes":
            tags.add("zo:total_query_reduction")
        for field, tag in (
            ("query_reuse", "zo:query_reuse"),
            ("low_rank_or_subspace", "zo:low_rank_or_subspace"),
            ("structured_perturbation", "zo:structured_perturbation"),
            ("forward_only", "zo:forward_only"),
        ):
            value = str(zo.get(field) or "").strip()
            if value and value not in {"not_available", "未提供", "不适用"}:
                tags.add(tag)
    return sorted(tags)


def search_text(source: dict[str, Any], summary: dict[str, Any]) -> str:
    values: list[str] = [
        str(source.get("title") or ""),
        " ".join(str(item) for item in source.get("authors") or []),
        str(source.get("venue") or ""),
        " ".join(str(item) for item in source.get("matched_projects") or []),
        str(summary.get("core_problem") or ""),
        str(summary.get("method_and_architecture") or ""),
        str(summary.get("method_principle") or ""),
        str(summary.get("research_value") or ""),
    ]
    values.extend(str(item) for item in summary.get("main_contributions") or [])
    return "\n".join(value.strip() for value in values if value.strip())


def build_record(
    *,
    request: dict[str, Any],
    summary: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    source = request.get("source") or {}
    candidate_id = str(request["candidate_id"])
    doi = normalize_doi(source.get("doi"))
    digest_date = str(generation.get("digest_date") or "")
    completed_at = str(generation.get("completed_at") or "")
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "title": str(source.get("title") or ""),
        "authors": list(source.get("authors") or []),
        "venue": source.get("venue"),
        "year": source.get("year"),
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}" if doi else None,
        "openalex_id": source.get("openalex_id"),
        "landing_page": source.get("landing_page"),
        "source_type": source.get("source_type"),
        "matched_projects": sorted(
            str(item) for item in source.get("matched_projects") or []
        ),
        "mandatory": bool(source.get("mandatory")),
        "score": float(source.get("score") or 0),
        "decision": str(source.get("decision") or "summarize"),
        "digest_date": digest_date,
        "completed_at": completed_at,
        "provider": str(generation.get("provider") or ""),
        "model": str(generation.get("model") or ""),
        "request_id": request.get("request_id"),
        "summary_record_sha256": record_sha256(summary),
        "summary_file": f"data/summaries/{digest_date}.jsonl",
        "digest_markdown_file": f"data/digests/{digest_date}.generated.md",
        "knowledge_tags": normalized_tags(source, summary),
        "search_text": search_text(source, summary),
        "summary": summary,
    }


def add_index(mapping: dict[str, list[str]], key: Any, candidate_id: str) -> None:
    value = str(key or "").strip()
    if value:
        mapping[value].append(candidate_id)


def build_index(records: list[dict[str, Any]], *, updated_at: str) -> dict[str, Any]:
    by_project: dict[str, list[str]] = defaultdict(list)
    by_year: dict[str, list[str]] = defaultdict(list)
    by_venue: dict[str, list[str]] = defaultdict(list)
    by_digest_date: dict[str, list[str]] = defaultdict(list)
    by_tag: dict[str, list[str]] = defaultdict(list)
    by_doi: dict[str, str] = {}
    by_candidate_id: dict[str, dict[str, Any]] = {}

    for record in records:
        candidate_id = str(record["candidate_id"])
        by_candidate_id[candidate_id] = {
            "title": record["title"],
            "doi": record.get("doi"),
            "digest_date": record.get("digest_date"),
            "matched_projects": record.get("matched_projects") or [],
            "summary_record_sha256": record.get("summary_record_sha256"),
        }
        doi = str(record.get("doi") or "")
        if doi:
            existing = by_doi.get(doi)
            if existing and existing != candidate_id:
                raise RuntimeError(f"DOI maps to multiple candidates: {doi}")
            by_doi[doi] = candidate_id
        for project in record.get("matched_projects") or []:
            add_index(by_project, project, candidate_id)
        add_index(by_year, record.get("year"), candidate_id)
        add_index(by_venue, record.get("venue"), candidate_id)
        add_index(by_digest_date, record.get("digest_date"), candidate_id)
        for tag in record.get("knowledge_tags") or []:
            add_index(by_tag, tag, candidate_id)

    def sorted_mapping(value: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            key: sorted(set(candidate_ids))
            for key, candidate_ids in sorted(value.items())
        }

    return {
        "schema_version": 1,
        "updated_at": updated_at,
        "paper_count": len(records),
        "by_candidate_id": dict(sorted(by_candidate_id.items())),
        "by_doi": dict(sorted(by_doi.items())),
        "by_project": sorted_mapping(by_project),
        "by_year": sorted_mapping(by_year),
        "by_venue": sorted_mapping(by_venue),
        "by_digest_date": sorted_mapping(by_digest_date),
        "by_tag": sorted_mapping(by_tag),
    }


def render_markdown(records: list[dict[str, Any]], *, updated_at: str) -> str:
    lines = [
        "# Research Inbox 长期知识库索引",
        "",
        f"- 论文数：`{len(records)}`",
        f"- 更新时间：`{updated_at}`",
        "- 正文原文：`不保存`",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        projects = record.get("matched_projects") or ["unclassified"]
        for project in projects:
            grouped[str(project)].append(record)

    for project, project_records in sorted(grouped.items()):
        lines.extend([f"## {project}", ""])
        ordered = sorted(
            project_records,
            key=lambda item: (
                str(item.get("digest_date") or ""),
                str(item.get("title") or "").casefold(),
            ),
            reverse=True,
        )
        for record in ordered:
            title = str(record.get("title") or "未提供标题")
            doi = record.get("doi")
            target = doi_markdown(doi) if doi else (record.get("landing_page") or "未提供")
            venue_year = "，".join(
                str(value)
                for value in (record.get("venue"), record.get("year"))
                if value
            )
            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"- DOI/链接：{target}",
                    f"- 期刊/年份：{venue_year or '未提供'}",
                    f"- 摘要日期：`{record.get('digest_date') or '未提供'}`",
                    f"- 候选 ID：`{record['candidate_id']}`",
                    f"- 标签：{', '.join(record.get('knowledge_tags') or []) or '未提供'}",
                    "",
                ]
            )
    return "\n".join(lines)


def update_knowledge_base(
    *,
    generation_manifest_path: Path,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict):
        raise RuntimeError("Summary generation manifest must be a JSON object")
    if generation.get("status") != "completed_automatic":
        raise RuntimeError("Knowledge indexing requires completed_automatic status")
    if generation.get("summary_history_updated") is not True:
        raise RuntimeError("Knowledge indexing requires completed summary history")

    request_path = Path(str(generation.get("request_file") or ""))
    summary_path = Path(str(generation.get("summary_file") or ""))
    if not request_path.exists() or not summary_path.exists():
        raise RuntimeError("Summary requests and summaries are required")

    requests = load_jsonl(request_path)
    summaries = load_jsonl(summary_path)
    request_by_id = unique_by_candidate(requests, "Summary requests")
    summary_by_id = unique_by_candidate(summaries, "Summaries")
    if set(request_by_id) != set(summary_by_id):
        raise RuntimeError("Request and summary candidate sets do not match")

    knowledge_root = output_root / "knowledge_base"
    papers_path = knowledge_root / "papers.jsonl"
    index_path = knowledge_root / "index.json"
    markdown_path = knowledge_root / "index.md"
    existing = load_jsonl(papers_path)
    existing_by_id = unique_by_candidate(existing, "Knowledge papers")

    newly_indexed = 0
    for candidate_id in sorted(request_by_id):
        record = build_record(
            request=request_by_id[candidate_id],
            summary=summary_by_id[candidate_id],
            generation=generation,
        )
        previous = existing_by_id.get(candidate_id)
        if previous is not None:
            if previous.get("summary_record_sha256") != record.get("summary_record_sha256"):
                raise RuntimeError(
                    f"Knowledge record changed for completed candidate: {candidate_id}"
                )
            continue
        existing_by_id[candidate_id] = record
        newly_indexed += 1

    records = sorted(
        existing_by_id.values(),
        key=lambda item: (
            str(item.get("completed_at") or ""),
            str(item.get("candidate_id") or ""),
        ),
    )
    updated_at = str(generation.get("completed_at") or "")
    index = build_index(records, updated_at=updated_at)

    atomic_write(
        papers_path,
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
    )
    atomic_write(
        index_path,
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(markdown_path, render_markdown(records, updated_at=updated_at))

    state = {
        "schema_version": 1,
        "status": "completed",
        "updated_at": updated_at,
        "paper_count": len(records),
        "newly_indexed_count": newly_indexed,
        "papers_file": str(papers_path),
        "papers_sha256": file_sha256(papers_path),
        "index_file": str(index_path),
        "index_sha256": file_sha256(index_path),
        "markdown_file": str(markdown_path),
        "markdown_sha256": file_sha256(markdown_path),
        "full_text_persisted": False,
    }
    atomic_write(
        manifest_path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(stable_json(state), flush=True)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update the long-term Research Inbox knowledge-base index"
    )
    parser.add_argument("--generation-manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    update_knowledge_base(
        generation_manifest_path=args.generation_manifest_path,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

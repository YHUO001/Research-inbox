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
            "For optical neural networks, identify free-space, integrated, or hybrid "
            "architecture; training method; optical nonlinearity; calibration needs; "
            "application tasks; and whether physical hardware was demonstrated."
        )
    if "zeroth-order-optimization" in projects:
        instructions.append(
            "For zeroth-order optimization, distinguish total query complexity from "
            "per-step query count and report query reuse, low-rank or subspace methods, "
            "structured perturbations, and forward-only execution when available."
        )
    if not instructions:
        instructions.append(
            "Describe the method, evidence, limitations, and research relevance using "
            "only the supplied title, metadata, and abstract."
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
            "abstract",
            "matched_projects",
            "score",
            "score_breakdown",
        )
    }
    instruction_text = "\n".join(f"- {item}" for item in instructions)
    return (
        "Return one JSON object matching the supplied paper-summary schema.\n"
        "Use only the supplied metadata and abstract. Do not infer missing experiments, "
        "numbers, comparisons, or causal claims. Mark unavailable information explicitly. "
        "Every numerical result must already appear in the supplied source text. "
        "Treat experimental claims as reported by the authors, not independently verified.\n"
        f"Project-specific checks:\n{instruction_text}\n"
        f"Source record:\n{json.dumps(source, ensure_ascii=False, sort_keys=True)}"
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
    r"(?<![A-Za-z0-9])(?:~|≈|±)?\s*\d+(?:\.\d+)?(?:\s*%|\s*[A-Za-z]+(?:/[A-Za-z0-9²^]+)?)?"
)


def normalize_numeric_token(value: str) -> str:
    return re.sub(r"\s+", "", value.lower()).replace("≈", "~")


def numeric_tokens(value: str) -> set[str]:
    return {
        normalize_numeric_token(match.group(0))
        for match in _NUMERIC_TOKEN.finditer(value or "")
    }


def validate_numeric_grounding(
    summary_record: dict[str, Any],
    *,
    title: str,
    abstract: str | None,
) -> list[str]:
    source_tokens = numeric_tokens(f"{title}\n{abstract or ''}")
    summary_text = stable_json(summary_record)
    summary_tokens = numeric_tokens(summary_text)
    return sorted(token for token in summary_tokens if token not in source_tokens)


def render_digest_markdown(digest: dict[str, Any]) -> str:
    lines = [
        f"# Research Inbox Preview — {digest['digest_date']}",
        "",
        "> Dry run only. No model was called and no email was sent.",
        "",
        "## Must read",
        "",
    ]
    must_read = digest["sections"]["must_read"]
    if not must_read:
        lines.append("No summary-slot candidates.")
    for index, item in enumerate(must_read, start=1):
        venue_year = ", ".join(
            str(value) for value in (item.get("venue"), item.get("year")) if value
        )
        lines.extend(
            [
                f"### {index}. {item['title']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Score: {item['score']:.2f}",
                f"- Venue: {venue_year or 'not available'}",
                f"- Source: {item['source_type']}",
                f"- Link: {item.get('landing_page') or 'not available'}",
                "",
            ]
        )
    lines.extend(["## Next candidates", ""])
    next_items = digest["sections"]["next_candidates"]
    if not next_items:
        lines.append("No additional budgeted candidates.")
    for item in next_items:
        lines.append(f"- {item['title']} — score {item['score']:.2f}")
    lines.extend(
        [
            "",
            "## Safety state",
            "",
            "- LLM generation: disabled",
            "- Email delivery: disabled",
            "- Summary history updated: no",
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
            "must_read": [
                digest_item(item, "pending_model_summary") for item in summary_slots
            ],
            "next_candidates": [
                digest_item(item, "budgeted_not_in_summary_slot")
                for item in next_candidates
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

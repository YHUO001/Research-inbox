from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.summarize.evidence_guard import enforce_onn_architecture
from scripts.summarize.generate_summaries import (
    render_markdown,
    validate_summary_numeric_grounding,
)
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


def relative_state_path(path: Path, output_root: Path) -> str:
    try:
        return str(path.relative_to(output_root.parent))
    except ValueError:
        return str(path)


def render_review_markdown(packet: dict[str, Any]) -> str:
    lines = [
        f"# Human Summary Review — {packet['digest_date']}",
        "",
        "> Review the generated summary against the supplied abstract only. The abstract is the current evidence boundary; full text was not used.",
        "",
        "## Review scale",
        "",
        "- 5: accurate and complete for the available abstract",
        "- 4: minor wording or emphasis issue",
        "- 3: usable but needs a substantive correction",
        "- 2: major omissions or technical misclassification",
        "- 1: unreliable",
        "",
    ]
    for index, item in enumerate(packet["papers"], start=1):
        source = item["source"]
        summary = item["summary"]
        checks = item["automated_checks"]
        onn = summary.get("optical_neural_network_analysis") or {}
        lines.extend(
            [
                f"## Paper {index}: {source['title']}",
                "",
                f"- Candidate ID: `{item['candidate_id']}`",
                f"- Venue: {source.get('venue') or 'not_available'}, {source.get('year') or 'not_available'}",
                f"- DOI: {source.get('doi') or 'not_available'}",
                f"- Model: `{packet['model']}`",
                "",
                "### Source abstract",
                "",
                source.get("abstract") or "not_available",
                "",
                "### Generated summary",
                "",
                f"**Core problem:** {summary['core_problem']}",
                "",
                f"**Method and architecture:** {summary['method_and_architecture']}",
                "",
                "**Main contributions:**",
                "",
            ]
        )
        lines.extend(f"- {value}" for value in summary["main_contributions"])
        lines.extend(["", "**Reported results:**", ""])
        results = summary.get("reported_results") or []
        lines.extend(
            [
                f"- {value['claim']} (basis: {value['basis']}; author-reported)"
                for value in results
            ]
            or ["- not_available"]
        )
        lines.extend(
            [
                "",
                f"**Distinction from prior work:** {summary['distinction_from_prior_work']}",
                "",
                f"**Research value:** {summary['research_value']}",
                "",
                "**Limitations and open questions:**",
                "",
            ]
        )
        lines.extend(
            f"- {value}" for value in summary["limitations_and_open_questions"]
        )
        lines.extend(
            [
                "",
                "### ONN technical classification",
                "",
                f"- Architecture type: `{onn.get('architecture_type', 'not_available')}`",
                f"- Training method: {onn.get('training_method', 'not_available')}",
                f"- Optical nonlinearity: {onn.get('optical_nonlinearity', 'not_available')}",
                f"- Calibration: {onn.get('calibration_requirements', 'not_available')}",
                f"- Hardware validation: `{onn.get('hardware_validation', 'not_available')}`",
                "",
                "### Automated checks",
                "",
                f"- JSON Schema valid: `{str(checks['schema_valid']).lower()}`",
                f"- Unsupported numeric claims: `{checks['unsupported_numeric_claims'] or []}`",
                f"- Architecture evidence result: `{checks['architecture_evidence']['resolved_type']}`",
                f"- Architecture consistent: `{str(checks['architecture_consistent']).lower()}`",
                "",
                "Architecture evidence:",
                "",
            ]
        )
        evidence = checks["architecture_evidence"]
        lines.extend(
            f"- Free-space: {value}" for value in evidence["free_space_evidence"]
        )
        lines.extend(
            f"- Integrated: {value}" for value in evidence["integrated_evidence"]
        )
        if not evidence["free_space_evidence"] and not evidence["integrated_evidence"]:
            lines.append("- No explicit architecture evidence in the abstract; `unclear` is required.")
        lines.extend(
            [
                "",
                "### Your evaluation",
                "",
                "| Criterion | Score (1–5) | Notes |",
                "|---|---:|---|",
                "| Factual accuracy |  |  |",
                "| Completeness |  |  |",
                "| Technical classification |  |  |",
                "| Usefulness for research triage |  |  |",
                "",
                "Decision: `approve` / `revise`",
                "",
                "Required corrections:",
                "",
                "---",
                "",
            ]
        )
    lines.extend(
        [
            "## Batch decision",
            "",
            "Only run **Finalize Reviewed Summaries** with `approve_all` after all papers are acceptable. Choose `hold_for_revision` when any paper needs correction.",
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
        raise RuntimeError("Review packets require email and history updates to be disabled")

    digest_date = str(generation["digest_date"])
    request_path = output_root / "summary_requests" / f"{digest_date}.jsonl"
    summary_path = output_root / "summaries" / f"{digest_date}.jsonl"
    requests = load_jsonl(request_path)
    summaries = load_jsonl(summary_path)
    if len(requests) != len(summaries) or not summaries:
        raise RuntimeError("Review packet requires one summary per request")

    schema = load_json(summary_schema_path, {})
    requests_by_id = {str(item["candidate_id"]): item for item in requests}
    canonical_summaries: list[dict[str, Any]] = []
    papers: list[dict[str, Any]] = []
    architecture_repairs = 0

    for summary in summaries:
        candidate_id = str(summary["candidate_id"])
        request = requests_by_id.get(candidate_id)
        if not request:
            raise RuntimeError(f"Missing request for summary {candidate_id}")
        source = request["source"]
        summary, evidence, changed, previous = enforce_onn_architecture(
            summary,
            abstract=source.get("abstract"),
        )
        architecture_repairs += int(changed)
        unsupported = validate_summary_numeric_grounding(
            summary,
            title=str(source["title"]),
            abstract=source.get("abstract"),
        )
        if unsupported:
            raise RuntimeError(
                f"Summary {candidate_id} has unsupported numeric claims: {unsupported}"
            )
        validate_record(summary, schema, f"summary {candidate_id}")
        canonical_summaries.append(summary)
        papers.append(
            {
                "candidate_id": candidate_id,
                "request_id": request.get("request_id"),
                "source": source,
                "summary": summary,
                "summary_sha256": record_sha256(summary),
                "automated_checks": {
                    "schema_valid": True,
                    "unsupported_numeric_claims": [],
                    "information_basis": "title_metadata_and_abstract_only",
                    "architecture_evidence": evidence.as_dict(),
                    "architecture_model_value_before_guard": previous,
                    "architecture_consistent": not changed,
                    "architecture_repaired": changed,
                },
                "review_template": {
                    "factual_accuracy": None,
                    "completeness": None,
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
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in canonical_summaries
        ),
    )

    digest_json_path = output_root / "digests" / f"{digest_date}.generated.json"
    digest_markdown_path = output_root / "digests" / f"{digest_date}.generated.md"
    digest = {
        "schema_version": 1,
        "digest_version": 1,
        "digest_date": digest_date,
        "status": "pending_human_review",
        "provider": generation["provider"],
        "model": generation["model"],
        "summary_count": len(canonical_summaries),
        "summaries": canonical_summaries,
        "safety": {
            "information_basis": "title_metadata_and_abstract_only",
            "full_text_used": False,
            "email_enabled": False,
            "summary_history_updated": False,
            "human_review_required": True,
        },
    }
    atomic_write(
        digest_json_path,
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        digest_markdown_path,
        render_markdown(digest_date, requests, canonical_summaries),
    )

    review_json_path = output_root / "reviews" / f"{digest_date}.review.json"
    review_markdown_path = output_root / "reviews" / f"{digest_date}.review.md"
    packet = {
        "schema_version": 1,
        "review_version": 1,
        "digest_date": digest_date,
        "status": "pending_human_review",
        "provider": generation["provider"],
        "model": generation["model"],
        "information_basis": "title_metadata_and_abstract_only",
        "paper_count": len(papers),
        "papers": papers,
        "batch_review": {
            "decision": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "notes": "",
        },
        "safety": {
            "full_text_used": False,
            "email_enabled": False,
            "summary_history_updated": False,
        },
        "artifacts": {
            "request_file": relative_state_path(request_path, output_root),
            "request_sha256": file_sha256(request_path),
            "summary_file": relative_state_path(summary_path, output_root),
            "summary_sha256": file_sha256(summary_path),
            "digest_json_file": relative_state_path(digest_json_path, output_root),
            "digest_json_sha256": file_sha256(digest_json_path),
        },
    }
    atomic_write(
        review_json_path,
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(review_markdown_path, render_review_markdown(packet))

    review_state = {
        "schema_version": 1,
        "review_version": 1,
        "status": "pending_human_review",
        "digest_date": digest_date,
        "paper_count": len(papers),
        "model": generation["model"],
        "architecture_repairs": architecture_repairs,
        "review_json_file": relative_state_path(review_json_path, output_root),
        "review_markdown_file": relative_state_path(review_markdown_path, output_root),
        "review_json_sha256": file_sha256(review_json_path),
        "summary_history_updated": False,
        "email_enabled": False,
    }
    atomic_write(
        review_manifest_path,
        json.dumps(review_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    generation["review_status"] = "pending_human_review"
    generation["review_manifest_file"] = relative_state_path(
        review_manifest_path, output_root
    )
    generation["post_generation_architecture_repairs"] = architecture_repairs
    generation["summary_file_sha256"] = file_sha256(summary_path)
    atomic_write(
        generation_manifest_path,
        json.dumps(generation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return review_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human summary review packet")
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

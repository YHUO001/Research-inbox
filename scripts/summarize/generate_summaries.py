from __future__ import annotations

import argparse
import json
import os
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
    stable_json,
    validate_numeric_grounding,
    validate_record,
)


def system_prompt(schema: dict[str, Any], candidate_id: str) -> str:
    example = {
        "schema_version": 1,
        "summary_version": 1,
        "candidate_id": candidate_id,
        "core_problem": "not_available",
        "method_and_architecture": "not_available",
        "main_contributions": ["not_available"],
        "reported_results": [],
        "distinction_from_prior_work": "not_available",
        "research_value": "not_available",
        "limitations_and_open_questions": ["not_available"],
        "optical_neural_network_analysis": None,
        "zeroth_order_analysis": None,
        "verification": {
            "information_basis": "title_metadata_and_abstract_only",
            "unsupported_numbers_detected": False,
            "missing_information": [],
        },
    }
    return (
        "Return JSON only. The output must match the supplied JSON Schema exactly, "
        "with no markdown or commentary. Preserve the required candidate_id. Use "
        "not_available when the source does not support a field. Do not invent numbers.\n"
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
        f"Example JSON shape:\n{json.dumps(example, ensure_ascii=False, sort_keys=True)}"
    )


def parse_model_json(content: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


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
    *,
    digest_date: str,
    requests: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    by_id = {item["candidate_id"]: item for item in summaries}
    lines = [
        f"# Research Inbox — {digest_date}",
        "",
        "> Generated from title, metadata, and abstract only. Claims are not independently verified.",
        "",
    ]
    for index, request in enumerate(requests, start=1):
        source = request["source"]
        summary = by_id[request["candidate_id"]]
        venue_year = ", ".join(
            str(value) for value in (source.get("venue"), source.get("year")) if value
        )
        lines.extend(
            [
                f"## {index}. {source['title']}",
                "",
                f"- Venue: {venue_year or 'not available'}",
                f"- Score: {float(source['score']):.2f}",
                f"- Link: {source.get('landing_page') or 'not available'}",
                "",
                "### Core problem",
                "",
                summary["core_problem"],
                "",
                "### Method and architecture",
                "",
                summary["method_and_architecture"],
                "",
                "### Main contributions",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in summary["main_contributions"])
        lines.extend(["", "### Reported results", ""])
        if summary["reported_results"]:
            lines.extend(
                f"- {item['claim']} (`reported_by_authors`, basis: {item['basis']})"
                for item in summary["reported_results"]
            )
        else:
            lines.append("- not_available")
        lines.extend(["", "### Research value", "", summary["research_value"], ""])
        lines.extend(["### Limitations and open questions", ""])
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
    if execution.get("mode") != "manual_provider_validation":
        raise RuntimeError("DeepSeek generation is limited to manual provider validation")
    if execution.get("email_enabled"):
        raise RuntimeError("Email must remain disabled during provider validation")
    if execution.get("update_summary_history"):
        raise RuntimeError("Summary history updates are disabled during provider validation")

    dry_run_manifest = load_json(dry_run_manifest_path, {})
    digest_date = str(dry_run_manifest.get("digest_date") or "")
    request_file = Path(str(dry_run_manifest.get("request_file") or ""))
    if not digest_date or not request_file.exists():
        raise RuntimeError("A successful summary dry run is required before generation")
    requests = load_jsonl(request_file)
    maximum = int(config["limits"]["maximum_summaries_per_run"])
    requests = requests[:maximum]
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
        correction = ""
        completed = False
        for validation_attempt in range(validation_attempts):
            try:
                response = client.complete_json(
                    model=str(provider["model"]),
                    system_prompt=system_prompt(schema, candidate_id),
                    user_prompt=str(request["prompt"]) + correction,
                    max_tokens=int(provider["max_output_tokens"]),
                    thinking_enabled=bool(provider["thinking_enabled"]),
                )
                responses.append(response)
                summary = parse_model_json(response.content)
                if summary.get("candidate_id") != candidate_id:
                    raise ValueError("candidate_id mismatch")
                unsupported = validate_numeric_grounding(
                    summary,
                    title=str(request["source"]["title"]),
                    abstract=request["source"].get("abstract"),
                )
                if unsupported:
                    raise ValueError(
                        "unsupported numeric claims: " + ", ".join(unsupported)
                    )
                validate_record(summary, schema, f"summary {candidate_id}")
                summaries.append(summary)
                completed = True
                break
            except (DeepSeekRequestError, json.JSONDecodeError, ValueError) as error:
                if validation_attempt + 1 >= validation_attempts:
                    failures.append(
                        {"candidate_id": candidate_id, "reason": str(error)[:500]}
                    )
                    break
                correction = (
                    "\nYour previous JSON response failed local validation: "
                    f"{str(error)[:300]}. Return a corrected JSON object only."
                )
        if not completed:
            continue

    usage = usage_totals(responses)
    state = {
        "schema_version": 1,
        "summary_generation_version": int(config["summary_generation_version"]),
        "status": "completed" if not failures else "failed_validation",
        "digest_date": digest_date,
        "provider": "deepseek",
        "model": str(provider["model"]),
        "thinking_enabled": bool(provider["thinking_enabled"]),
        "request_count": len(requests),
        "summary_count": len(summaries),
        "failure_count": len(failures),
        "failures": failures,
        "usage": usage,
        "estimated_cost_cny": estimate_cost_cny(usage, provider["pricing"]),
        "information_basis": "title_metadata_and_abstract_only",
        "full_text_used": False,
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
    summaries_content = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in summaries
    )
    digest = {
        "schema_version": 1,
        "digest_version": 1,
        "digest_date": digest_date,
        "status": "generated_pending_human_review",
        "provider": "deepseek",
        "model": str(provider["model"]),
        "summary_count": len(summaries),
        "summaries": summaries,
        "safety": {
            "information_basis": "title_metadata_and_abstract_only",
            "full_text_used": False,
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
    atomic_write(summaries_path, summaries_content)
    atomic_write(
        digest_json_path,
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        digest_markdown_path,
        render_markdown(
            digest_date=digest_date,
            requests=requests,
            summaries=summaries,
        ),
    )
    atomic_write(
        manifest_path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate locally validated paper summaries with DeepSeek"
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

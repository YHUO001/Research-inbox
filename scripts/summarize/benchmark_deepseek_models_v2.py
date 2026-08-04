from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from scripts.summarize.benchmark_deepseek_models import (
    categorical_comparison,
    file_sha256,
    model_config_file,
    normalized_cache_miss_cost_cny,
    render_report,
    summary_metrics,
)
from scripts.summarize.generate_summaries import (
    estimate_cost_cny,
    generate,
    render_markdown,
)
from scripts.summarize.prepare_digest import atomic_write, load_json, load_jsonl, stable_json


USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


def add_usage(total: dict[str, int], current: dict[str, Any]) -> None:
    for key in USAGE_KEYS:
        total[key] = int(total.get(key) or 0) + int(current.get(key) or 0)


def candidate_failure(candidate_id: str, state: dict[str, Any], error: Exception) -> dict[str, str]:
    reasons = state.get("failures") or []
    if reasons and isinstance(reasons[0], dict):
        reason = str(reasons[0].get("reason") or error)
    else:
        reason = str(error)
    return {"candidate_id": candidate_id, "reason": reason[:500]}


def append_failure_section(report: str, models: dict[str, dict[str, Any]]) -> str:
    lines = [report.rstrip(), "", "## Validation failures", ""]
    any_failure = False
    for label, result in models.items():
        failures = result.get("failures") or []
        if not failures:
            continue
        any_failure = True
        lines.append(f"### {label}")
        lines.append("")
        for failure in failures:
            lines.append(
                f"- `{failure.get('candidate_id')}`: {failure.get('reason') or 'unknown failure'}"
            )
        lines.append("")
    if not any_failure:
        lines.append("No candidate-level validation failures.")
        lines.append("")
    return "\n".join(lines)


def benchmark(
    *,
    dry_run_manifest_path: Path,
    summary_schema_path: Path,
    config_path: Path,
    output_root: Path,
    manifest_path: Path,
    api_key: str | None = None,
    client_factory: Callable[[str], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    benchmark_config = config.get("benchmark") or {}
    model_specs = benchmark_config.get("models") or []
    if len(model_specs) != 2:
        raise RuntimeError("Benchmark requires exactly two model specifications")
    execution = config.get("execution") or {}
    if execution.get("email_enabled") or execution.get("update_summary_history"):
        raise RuntimeError("Benchmark cannot enable email or summary-history updates")

    dry_run_manifest = load_json(dry_run_manifest_path, {})
    digest_date = str(dry_run_manifest.get("digest_date") or "")
    request_file = Path(str(dry_run_manifest.get("request_file") or ""))
    if not digest_date or not request_file.exists():
        raise RuntimeError("A successful summary dry run is required before benchmarking")
    requests = load_jsonl(request_file)[: int(config["limits"]["maximum_summaries_per_run"])]
    if not requests:
        raise RuntimeError("No summary requests are available")

    api_key = api_key or os.environ.get(str(config["provider"]["api_key_env"]))
    if not api_key:
        raise RuntimeError("Missing required DeepSeek API key")

    benchmark_root = output_root / "benchmarks" / "deepseek" / digest_date
    benchmark_root.mkdir(parents=True, exist_ok=True)
    request_snapshot = benchmark_root / "requests.jsonl"
    atomic_write(
        request_snapshot,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in requests
        ),
    )

    model_states: dict[str, dict[str, Any]] = {}
    model_records: dict[str, list[dict[str, Any]]] = {}

    for model_spec in model_specs:
        label = str(model_spec["label"])
        model = str(model_spec["model"])
        model_root = benchmark_root / label
        model_root.mkdir(parents=True, exist_ok=True)
        combined_summaries: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        usage: dict[str, int] = {key: 0 for key in USAGE_KEYS}
        started = clock()
        temporary_config = model_config_file(config, model_spec)

        try:
            for request in requests:
                candidate_id = str(request["candidate_id"])
                candidate_root = model_root / "candidates" / candidate_id
                candidate_root.mkdir(parents=True, exist_ok=True)
                candidate_request = candidate_root / "request.jsonl"
                candidate_dry_manifest = candidate_root / "dry_run_manifest.json"
                candidate_state_manifest = candidate_root / "manifest.json"
                atomic_write(
                    candidate_request,
                    json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n",
                )
                atomic_write(
                    candidate_dry_manifest,
                    json.dumps(
                        {
                            "digest_date": digest_date,
                            "request_file": str(candidate_request),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
                try:
                    state = generate(
                        dry_run_manifest_path=candidate_dry_manifest,
                        summary_schema_path=summary_schema_path,
                        config_path=Path(temporary_config.name),
                        output_root=candidate_root / "output",
                        manifest_path=candidate_state_manifest,
                        api_key=api_key,
                        client=client_factory(model) if client_factory else None,
                    )
                except Exception as error:
                    state = load_json(candidate_state_manifest, {})
                    if not isinstance(state, dict):
                        state = {}
                    failures.append(candidate_failure(candidate_id, state, error))
                add_usage(usage, state.get("usage") or {})
                summary_file = state.get("summary_file")
                if summary_file and Path(str(summary_file)).exists():
                    combined_summaries.extend(load_jsonl(Path(str(summary_file))))
        finally:
            temporary_config.close()
            Path(temporary_config.name).unlink(missing_ok=True)

        elapsed = round(max(0.0, clock() - started), 4)
        completed_ids = {str(item["candidate_id"]) for item in combined_summaries}
        completed_requests = [
            request for request in requests if str(request["candidate_id"]) in completed_ids
        ]
        summaries_path = model_root / "summaries" / f"{digest_date}.jsonl"
        digest_path = model_root / "digests" / f"{digest_date}.generated.md"
        atomic_write(
            summaries_path,
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in combined_summaries
            ),
        )
        atomic_write(
            digest_path,
            render_markdown(digest_date, completed_requests, combined_summaries)
            if combined_summaries
            else (
                f"# Research Inbox — {digest_date}\n\n"
                "> No candidate passed local validation for this model.\n"
            ),
        )
        result = {
            "model": model,
            "status": "completed" if not failures else "partial_failure",
            "request_count": len(requests),
            "summary_count": len(combined_summaries),
            "failure_count": len(failures),
            "failures": failures,
            "usage": usage,
            "estimated_cost_cny": estimate_cost_cny(usage, model_spec["pricing"]),
            "normalized_cache_miss_cost_cny": normalized_cache_miss_cost_cny(
                usage, model_spec["pricing"]
            ),
            "elapsed_seconds": elapsed,
            "quality_metrics": summary_metrics(combined_summaries),
            "summary_file": str(summaries_path),
            "digest_markdown_file": str(digest_path),
        }
        atomic_write(
            model_root / "manifest.json",
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        model_states[label] = result
        model_records[label] = combined_summaries

    completed = all(item.get("status") == "completed" for item in model_states.values())
    manifest = {
        "schema_version": 1,
        "benchmark_version": 2,
        "status": "completed" if completed else "partial_failure",
        "digest_date": digest_date,
        "request_count": len(requests),
        "request_snapshot": str(request_snapshot),
        "request_sha256": file_sha256(request_snapshot),
        "thinking_enabled": False,
        "json_mode": True,
        "same_prompt_for_both_models": True,
        "candidate_level_isolation": True,
        "information_basis": "title_metadata_and_abstract_only",
        "full_text_used": False,
        "email_enabled": False,
        "summary_history_updated": False,
        "models": model_states,
        "candidate_comparisons": categorical_comparison(model_records),
        "actual_total_cost_cny": round(
            sum(float(item.get("estimated_cost_cny") or 0) for item in model_states.values()),
            8,
        ),
        "normalized_total_cache_miss_cost_cny": round(
            sum(
                float(item.get("normalized_cache_miss_cost_cny") or 0)
                for item in model_states.values()
            ),
            8,
        ),
    }
    report_path = benchmark_root / "comparison.md"
    manifest["comparison_report"] = str(report_path)
    atomic_write(report_path, append_failure_section(render_report(manifest), model_states))
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek Flash and Pro with candidate-level isolation"
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
        default=Path("runtime-state/state/deepseek_benchmark_manifest.json"),
    )
    args = parser.parse_args()
    manifest = benchmark(
        dry_run_manifest_path=args.dry_run_manifest_path,
        summary_schema_path=args.summary_schema,
        config_path=args.config,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
    )
    print(stable_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

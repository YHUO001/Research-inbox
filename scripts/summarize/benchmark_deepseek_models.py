from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from scripts.summarize.generate_summaries import generate
from scripts.summarize.prepare_digest import atomic_write, load_json, load_jsonl, stable_json


NARRATIVE_FIELDS = (
    "core_problem",
    "method_and_architecture",
    "main_contributions",
    "reported_results",
    "distinction_from_prior_work",
    "research_value",
    "limitations_and_open_questions",
    "optical_neural_network_analysis",
    "zeroth_order_analysis",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_not_available(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() == "not_available")
    if isinstance(value, list):
        return sum(count_not_available(item) for item in value)
    if isinstance(value, dict):
        return sum(count_not_available(item) for item in value.values())
    return 0


def summary_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    narrative_characters = 0
    reported_results = 0
    contributions = 0
    limitations = 0
    missing_information = 0
    explicit_architecture = 0
    explicit_hardware_validation = 0
    not_available = 0
    for record in records:
        narrative = {key: record.get(key) for key in NARRATIVE_FIELDS}
        narrative_characters += len(stable_json(narrative))
        reported_results += len(record.get("reported_results") or [])
        contributions += len(record.get("main_contributions") or [])
        limitations += len(record.get("limitations_and_open_questions") or [])
        missing_information += len((record.get("verification") or {}).get("missing_information") or [])
        not_available += count_not_available(narrative)
        onn = record.get("optical_neural_network_analysis")
        if isinstance(onn, dict):
            if onn.get("architecture_type") not in {None, "unclear", "not_available"}:
                explicit_architecture += 1
            if onn.get("hardware_validation") not in {None, "unclear", "not_available"}:
                explicit_hardware_validation += 1
    return {
        "summary_count": len(records),
        "narrative_characters": narrative_characters,
        "reported_result_count": reported_results,
        "main_contribution_count": contributions,
        "limitation_count": limitations,
        "missing_information_count": missing_information,
        "not_available_count": not_available,
        "explicit_architecture_count": explicit_architecture,
        "explicit_hardware_validation_count": explicit_hardware_validation,
    }


def normalized_cache_miss_cost_cny(
    usage: dict[str, Any], pricing: dict[str, Any]
) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cost = (
        prompt * float(pricing["input_cache_miss_cny_per_million"])
        + completion * float(pricing["output_cny_per_million"])
    ) / 1_000_000
    return round(cost, 8)


def model_config_file(
    base_config: dict[str, Any], model_spec: dict[str, Any]
) -> tempfile.NamedTemporaryFile:
    model_config = copy.deepcopy(base_config)
    provider = model_config["provider"]
    provider["model"] = str(model_spec["model"])
    provider["thinking_enabled"] = False
    provider["pricing"] = copy.deepcopy(model_spec["pricing"])
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", delete=False
    )
    yaml.safe_dump(model_config, handle, sort_keys=False, allow_unicode=True)
    handle.flush()
    return handle


def load_completed_summaries(state: dict[str, Any]) -> list[dict[str, Any]]:
    path_value = state.get("summary_file")
    if not path_value:
        return []
    return load_jsonl(Path(str(path_value)))


def categorical_comparison(
    model_records: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    by_model = {
        label: {str(item["candidate_id"]): item for item in records}
        for label, records in model_records.items()
    }
    candidate_ids = sorted(
        set().union(*(set(records) for records in by_model.values()))
    )
    comparisons: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        item: dict[str, Any] = {"candidate_id": candidate_id, "models": {}}
        architecture_values: list[str | None] = []
        hardware_values: list[str | None] = []
        for label, records in by_model.items():
            record = records.get(candidate_id)
            onn = record.get("optical_neural_network_analysis") if record else None
            architecture = onn.get("architecture_type") if isinstance(onn, dict) else None
            hardware = onn.get("hardware_validation") if isinstance(onn, dict) else None
            architecture_values.append(architecture)
            hardware_values.append(hardware)
            item["models"][label] = {
                "present": record is not None,
                "architecture_type": architecture,
                "hardware_validation": hardware,
                "reported_result_count": len(record.get("reported_results") or []) if record else 0,
                "not_available_count": count_not_available(record) if record else 0,
            }
        item["architecture_agreement"] = (
            len(set(architecture_values)) == 1 and architecture_values[0] is not None
        )
        item["hardware_validation_agreement"] = (
            len(set(hardware_values)) == 1 and hardware_values[0] is not None
        )
        comparisons.append(item)
    return comparisons


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        f"# DeepSeek Flash vs Pro benchmark — {manifest['digest_date']}",
        "",
        "> Same three requests, same prompts, JSON mode, and thinking disabled. No email was sent and summary history was not updated.",
        "",
        "## Model totals",
        "",
        "| Model | Status | Summaries | Prompt tokens | Output tokens | Actual cost (CNY) | All-miss cost (CNY) | Elapsed (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in manifest["models"].items():
        usage = result.get("usage") or {}
        lines.append(
            "| {model} | {status} | {summaries} | {prompt} | {completion} | {actual:.6f} | {normalized:.6f} | {elapsed:.2f} |".format(
                model=label,
                status=result.get("status"),
                summaries=result.get("summary_count", 0),
                prompt=int(usage.get("prompt_tokens") or 0),
                completion=int(usage.get("completion_tokens") or 0),
                actual=float(result.get("estimated_cost_cny") or 0),
                normalized=float(result.get("normalized_cache_miss_cost_cny") or 0),
                elapsed=float(result.get("elapsed_seconds") or 0),
            )
        )
    lines.extend(
        [
            "",
            "## Deterministic quality indicators",
            "",
            "| Model | Valid summaries | Reported results | Contributions | Limitations | not_available | Explicit architecture | Explicit hardware validation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in manifest["models"].items():
        metrics = result.get("quality_metrics") or {}
        lines.append(
            "| {model} | {summaries} | {results} | {contributions} | {limitations} | {missing} | {architecture} | {hardware} |".format(
                model=label,
                summaries=int(metrics.get("summary_count") or 0),
                results=int(metrics.get("reported_result_count") or 0),
                contributions=int(metrics.get("main_contribution_count") or 0),
                limitations=int(metrics.get("limitation_count") or 0),
                missing=int(metrics.get("not_available_count") or 0),
                architecture=int(metrics.get("explicit_architecture_count") or 0),
                hardware=int(metrics.get("explicit_hardware_validation_count") or 0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These indicators verify structure, grounding constraints, coverage, token use, and categorical consistency. They do not by themselves establish that longer or more detailed prose is scientifically better. Final model selection requires paper-by-paper comparison against the supplied abstracts.",
            "",
        ]
    )
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
    if config["execution"].get("email_enabled") or config["execution"].get("update_summary_history"):
        raise RuntimeError("Benchmark cannot enable email or summary-history updates")

    dry_run_manifest = load_json(dry_run_manifest_path, {})
    digest_date = str(dry_run_manifest.get("digest_date") or "")
    request_file = Path(str(dry_run_manifest.get("request_file") or ""))
    if not digest_date or not request_file.exists():
        raise RuntimeError("A successful summary dry run is required before benchmarking")

    api_key = api_key or os.environ.get(str(config["provider"]["api_key_env"]))
    if not api_key:
        raise RuntimeError("Missing required DeepSeek API key")

    benchmark_root = output_root / "benchmarks" / "deepseek" / digest_date
    benchmark_root.mkdir(parents=True, exist_ok=True)
    request_snapshot = benchmark_root / "requests.jsonl"
    atomic_write(request_snapshot, request_file.read_text(encoding="utf-8"))

    model_states: dict[str, dict[str, Any]] = {}
    model_records: dict[str, list[dict[str, Any]]] = {}
    for model_spec in model_specs:
        label = str(model_spec["label"])
        model = str(model_spec["model"])
        model_root = benchmark_root / label
        model_manifest = model_root / "manifest.json"
        temporary_config = model_config_file(config, model_spec)
        started = clock()
        error_text: str | None = None
        try:
            state = generate(
                dry_run_manifest_path=dry_run_manifest_path,
                summary_schema_path=summary_schema_path,
                config_path=Path(temporary_config.name),
                output_root=model_root,
                manifest_path=model_manifest,
                api_key=api_key,
                client=client_factory(model) if client_factory else None,
            )
        except Exception as error:  # persist both model outcomes before failing the workflow
            error_text = str(error)[:500]
            state = load_json(model_manifest, {})
            if not isinstance(state, dict):
                state = {}
            state.setdefault("status", "benchmark_failed")
        finally:
            temporary_config.close()
            Path(temporary_config.name).unlink(missing_ok=True)
        elapsed = round(max(0.0, clock() - started), 4)
        records = load_completed_summaries(state) if state.get("status") == "completed" else []
        model_records[label] = records
        result = {
            "model": model,
            "status": state.get("status", "benchmark_failed"),
            "summary_count": int(state.get("summary_count") or 0),
            "failure_count": int(state.get("failure_count") or 0),
            "usage": state.get("usage") or {},
            "estimated_cost_cny": float(state.get("estimated_cost_cny") or 0),
            "normalized_cache_miss_cost_cny": normalized_cache_miss_cost_cny(
                state.get("usage") or {}, model_spec["pricing"]
            ),
            "elapsed_seconds": elapsed,
            "quality_metrics": summary_metrics(records),
            "manifest_file": str(model_manifest),
            "summary_file": state.get("summary_file"),
            "digest_markdown_file": state.get("digest_markdown_file"),
            "error": error_text,
        }
        model_states[label] = result

    completed = all(item.get("status") == "completed" for item in model_states.values())
    manifest = {
        "schema_version": 1,
        "benchmark_version": 1,
        "status": "completed" if completed else "partial_failure",
        "digest_date": digest_date,
        "request_count": len(load_jsonl(request_snapshot)),
        "request_snapshot": str(request_snapshot),
        "request_sha256": file_sha256(request_snapshot),
        "thinking_enabled": False,
        "json_mode": True,
        "same_prompt_for_both_models": True,
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
            sum(float(item.get("normalized_cache_miss_cost_cny") or 0) for item in model_states.values()),
            8,
        ),
    }
    report_path = benchmark_root / "comparison.md"
    manifest["comparison_report"] = str(report_path)
    atomic_write(report_path, render_report(manifest))
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if not completed:
        raise RuntimeError("One or more benchmark models failed")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek Flash and Pro on identical paper-summary requests"
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

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.summarize import benchmark_deepseek_models_v2 as benchmark_v2
from scripts.summarize.deepseek_provider import DeepSeekClient, DeepSeekResponse
from scripts.summarize.generate_summaries import generate as production_generate
from scripts.summarize.prepare_digest import atomic_write, load_json, load_jsonl, stable_json


@dataclass
class TransportRepairDiagnostics:
    expected_candidate_id: str
    candidate_id_repair_responses: int = 0
    unit_format_normalization_responses: int = 0


class BenchmarkNormalizingClient:
    """Normalize transport-only fields while preserving scientific validation.

    Candidate IDs are controlled by the caller rather than scientific model output.
    Unicode superscript unit formatting is normalized to the ASCII form used by some
    source abstracts. Approximation symbols and numerical values are not changed.
    """

    def __init__(
        self,
        *,
        base_client: Any,
        diagnostics: TransportRepairDiagnostics,
    ) -> None:
        self.base_client = base_client
        self.diagnostics = diagnostics

    @staticmethod
    def _normalize_units(value: Any) -> tuple[Any, bool]:
        if isinstance(value, str):
            normalized = value.replace("²", "2").replace("³", "3")
            return normalized, normalized != value
        if isinstance(value, list):
            changed = False
            output: list[Any] = []
            for item in value:
                normalized, item_changed = BenchmarkNormalizingClient._normalize_units(item)
                output.append(normalized)
                changed = changed or item_changed
            return output, changed
        if isinstance(value, dict):
            changed = False
            output: dict[str, Any] = {}
            for key, item in value.items():
                normalized, item_changed = BenchmarkNormalizingClient._normalize_units(item)
                output[str(key)] = normalized
                changed = changed or item_changed
            return output, changed
        return value, False

    def complete_json(self, **kwargs: Any) -> DeepSeekResponse:
        response = self.base_client.complete_json(**kwargs)
        try:
            value = json.loads(response.content)
        except json.JSONDecodeError:
            return response
        if not isinstance(value, dict):
            return response

        if value.get("candidate_id") != self.diagnostics.expected_candidate_id:
            self.diagnostics.candidate_id_repair_responses += 1
        value["candidate_id"] = self.diagnostics.expected_candidate_id

        normalized, unit_changed = self._normalize_units(value)
        if unit_changed:
            self.diagnostics.unit_format_normalization_responses += 1

        return DeepSeekResponse(
            content=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            usage=response.usage,
            model=response.model,
        )


def _candidate_id_from_dry_manifest(path: Path) -> str:
    manifest = load_json(path, {})
    request_file = Path(str(manifest.get("request_file") or ""))
    requests = load_jsonl(request_file)
    if len(requests) != 1:
        raise RuntimeError("Fair benchmark normalization requires one candidate request")
    return str(requests[0]["candidate_id"])


def _base_client(config: dict[str, Any], api_key: str, client: Any | None) -> Any:
    if client is not None:
        return client
    provider = config["provider"]
    return DeepSeekClient(
        api_key=api_key,
        base_url=str(provider["base_url"]),
        timeout_seconds=float(provider["timeout_seconds"]),
        max_attempts=int(provider["http_max_attempts"]),
    )


def normalized_generate(
    *,
    dry_run_manifest_path: Path,
    summary_schema_path: Path,
    config_path: Path,
    output_root: Path,
    manifest_path: Path,
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    key = api_key or ""
    if not key:
        raise RuntimeError("Missing required DeepSeek API key")

    expected_candidate_id = _candidate_id_from_dry_manifest(dry_run_manifest_path)
    diagnostics = TransportRepairDiagnostics(expected_candidate_id=expected_candidate_id)
    wrapped_client = BenchmarkNormalizingClient(
        base_client=_base_client(config, key, client),
        diagnostics=diagnostics,
    )

    try:
        state = production_generate(
            dry_run_manifest_path=dry_run_manifest_path,
            summary_schema_path=summary_schema_path,
            config_path=config_path,
            output_root=output_root,
            manifest_path=manifest_path,
            api_key=key,
            client=wrapped_client,
        )
    finally:
        state_value = load_json(manifest_path, {})
        if not isinstance(state_value, dict):
            state_value = {}
        state_value["transport_repairs"] = {
            "candidate_id_repair_responses": diagnostics.candidate_id_repair_responses,
            "unit_format_normalization_responses": diagnostics.unit_format_normalization_responses,
        }
        atomic_write(
            manifest_path,
            json.dumps(state_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    state["transport_repairs"] = {
        "candidate_id_repair_responses": diagnostics.candidate_id_repair_responses,
        "unit_format_normalization_responses": diagnostics.unit_format_normalization_responses,
    }
    return state


def _aggregate_transport_repairs(
    *,
    output_root: Path,
    digest_date: str,
    manifest: dict[str, Any],
) -> None:
    benchmark_root = output_root / "benchmarks" / "deepseek" / digest_date
    for label, model_result in (manifest.get("models") or {}).items():
        candidate_root = benchmark_root / str(label) / "candidates"
        candidate_repairs: list[dict[str, Any]] = []
        totals = {
            "candidate_id_repair_responses": 0,
            "unit_format_normalization_responses": 0,
        }
        if candidate_root.exists():
            for manifest_path in sorted(candidate_root.glob("*/manifest.json")):
                state = load_json(manifest_path, {})
                repairs = state.get("transport_repairs") or {}
                candidate_id = manifest_path.parent.name
                item = {
                    "candidate_id": candidate_id,
                    "candidate_id_repair_responses": int(
                        repairs.get("candidate_id_repair_responses") or 0
                    ),
                    "unit_format_normalization_responses": int(
                        repairs.get("unit_format_normalization_responses") or 0
                    ),
                }
                candidate_repairs.append(item)
                totals["candidate_id_repair_responses"] += item[
                    "candidate_id_repair_responses"
                ]
                totals["unit_format_normalization_responses"] += item[
                    "unit_format_normalization_responses"
                ]
        model_result["transport_repairs"] = totals
        model_result["candidate_transport_repairs"] = candidate_repairs

        model_manifest = benchmark_root / str(label) / "manifest.json"
        if model_manifest.exists():
            atomic_write(
                model_manifest,
                json.dumps(model_result, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )


def benchmark(**kwargs: Any) -> dict[str, Any]:
    original_generate = benchmark_v2.generate
    benchmark_v2.generate = normalized_generate
    try:
        manifest = benchmark_v2.benchmark(**kwargs)
    finally:
        benchmark_v2.generate = original_generate

    output_root = Path(kwargs["output_root"])
    _aggregate_transport_repairs(
        output_root=output_root,
        digest_date=str(manifest["digest_date"]),
        manifest=manifest,
    )
    manifest["benchmark_version"] = 3
    manifest["transport_fields_canonicalized"] = True
    manifest["unit_normalization_scope"] = ["²_to_2", "³_to_3"]

    manifest_path = Path(kwargs["manifest_path"])
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    report_path = Path(str(manifest["comparison_report"]))
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    lines = [report.rstrip(), "", "## Transport normalization", ""]
    for label, result in (manifest.get("models") or {}).items():
        repairs = result.get("transport_repairs") or {}
        lines.append(
            f"- {label}: candidate ID repairs = "
            f"{int(repairs.get('candidate_id_repair_responses') or 0)}, "
            f"unit-format normalizations = "
            f"{int(repairs.get('unit_format_normalization_responses') or 0)}"
        )
    lines.extend(
        [
            "",
            "Candidate IDs are transport metadata injected by the application. Only Unicode squared/cubed unit glyphs are normalized; approximation markers and numerical values remain strict.",
            "",
        ]
    )
    atomic_write(report_path, "\n".join(lines))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fairly benchmark DeepSeek Flash and Pro with transport normalization"
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

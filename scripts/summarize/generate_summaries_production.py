from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.summarize.deepseek_provider import DeepSeekClient, DeepSeekResponse
from scripts.summarize.generate_summaries import generate as strict_generate
from scripts.summarize.prepare_digest import atomic_write, load_json, stable_json


@dataclass
class TransportRepairDiagnostics:
    candidate_id_repair_responses: int = 0
    unit_format_normalization_responses: int = 0


def expected_candidate_id(system_prompt: str) -> str:
    marker = "Example JSON shape:\n"
    if marker not in system_prompt:
        raise RuntimeError("Summary system prompt is missing its example JSON shape")
    example = json.loads(system_prompt.rsplit(marker, 1)[1])
    candidate_id = str(example.get("candidate_id") or "")
    if not candidate_id:
        raise RuntimeError("Summary system prompt is missing candidate_id")
    return candidate_id


def normalize_unit_format(value: Any) -> tuple[Any, bool]:
    """Normalize equivalent squared/cubed unit glyphs only.

    Numerical values and approximation markers are deliberately untouched.
    """
    if isinstance(value, str):
        normalized = value.replace("²", "2").replace("³", "3")
        return normalized, normalized != value
    if isinstance(value, list):
        changed = False
        output: list[Any] = []
        for item in value:
            normalized, item_changed = normalize_unit_format(item)
            output.append(normalized)
            changed = changed or item_changed
        return output, changed
    if isinstance(value, dict):
        changed = False
        output: dict[str, Any] = {}
        for key, item in value.items():
            normalized, item_changed = normalize_unit_format(item)
            output[str(key)] = normalized
            changed = changed or item_changed
        return output, changed
    return value, False


class ProductionNormalizingClient:
    """Canonicalize application-owned transport data before strict validation."""

    def __init__(self, *, base_client: Any, diagnostics: TransportRepairDiagnostics) -> None:
        self.base_client = base_client
        self.diagnostics = diagnostics

    def complete_json(self, **kwargs: Any) -> DeepSeekResponse:
        response = self.base_client.complete_json(**kwargs)
        try:
            value = json.loads(response.content)
        except json.JSONDecodeError:
            return response
        if not isinstance(value, dict):
            return response

        candidate_id = expected_candidate_id(str(kwargs.get("system_prompt") or ""))
        if value.get("candidate_id") != candidate_id:
            self.diagnostics.candidate_id_repair_responses += 1
        value["candidate_id"] = candidate_id

        normalized, unit_changed = normalize_unit_format(value)
        if unit_changed:
            self.diagnostics.unit_format_normalization_responses += 1

        return DeepSeekResponse(
            content=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            usage=response.usage,
            model=response.model,
        )


def generate(
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
    provider = config["provider"]
    key = api_key or os.environ.get(str(provider["api_key_env"]))
    if not key:
        raise RuntimeError("Missing required DeepSeek API key")

    base_client = client or DeepSeekClient(
        api_key=key,
        base_url=str(provider["base_url"]),
        timeout_seconds=float(provider["timeout_seconds"]),
        max_attempts=int(provider["http_max_attempts"]),
    )
    diagnostics = TransportRepairDiagnostics()
    wrapped_client = ProductionNormalizingClient(
        base_client=base_client,
        diagnostics=diagnostics,
    )

    try:
        state = strict_generate(
            dry_run_manifest_path=dry_run_manifest_path,
            summary_schema_path=summary_schema_path,
            config_path=config_path,
            output_root=output_root,
            manifest_path=manifest_path,
            api_key=key,
            client=wrapped_client,
        )
    finally:
        persisted = load_json(manifest_path, {})
        if not isinstance(persisted, dict):
            persisted = {}
        persisted["transport_repairs"] = {
            "candidate_id_repair_responses": diagnostics.candidate_id_repair_responses,
            "unit_format_normalization_responses": diagnostics.unit_format_normalization_responses,
        }
        atomic_write(
            manifest_path,
            json.dumps(persisted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    state["transport_repairs"] = {
        "candidate_id_repair_responses": diagnostics.candidate_id_repair_responses,
        "unit_format_normalization_responses": diagnostics.unit_format_normalization_responses,
    }
    atomic_write(
        manifest_path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate production DeepSeek summaries with transport normalization"
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

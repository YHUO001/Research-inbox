from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scripts.summarize.deepseek_provider import DeepSeekClient, DeepSeekResponse
from scripts.summarize.evidence_guard import enforce_onn_architecture
from scripts.summarize.generate_summaries import generate as strict_generate
from scripts.summarize.prepare_digest import (
    atomic_write,
    load_json,
    load_jsonl,
    stable_json,
)


_TOPS_LONG_FORM = re.compile(
    r"(?<![A-Za-z0-9])((?:~|≈|±)?\s*\d+(?:\.\d+)?)\s+"
    r"trillion\s+operations\s+per\s+second"
    r"(?:\s*\(\s*TOPS\s*\))?",
    re.IGNORECASE,
)


@dataclass
class TransportRepairDiagnostics:
    candidate_id_repair_responses: int = 0
    unit_format_normalization_responses: int = 0
    tops_alias_expansions: int = 0
    architecture_repairs: int = 0
    architecture_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    """Normalize equivalent squared/cubed unit glyphs only."""
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


def tops_grounding_aliases(abstract: str) -> list[str]:
    """Return explicit TOPS aliases already defined by the abstract."""
    aliases: list[str] = []
    seen: set[str] = set()
    for match in _TOPS_LONG_FORM.finditer(abstract or ""):
        number = re.sub(r"\s+", "", match.group(1)).replace("≈", "~")
        alias = f"{number} TOPS"
        key = alias.lower()
        if key not in seen:
            seen.add(key)
            aliases.append(alias)
    return aliases


def request_abstracts(dry_run_manifest_path: Path) -> dict[str, str]:
    manifest = load_json(dry_run_manifest_path, {})
    request_file = Path(str(manifest.get("request_file") or ""))
    records = load_jsonl(request_file)
    return {
        str(record["candidate_id"]): str((record.get("source") or {}).get("abstract") or "")
        for record in records
    }


def grounding_manifest_copy(
    dry_run_manifest_path: Path,
    temporary_root: Path,
) -> tuple[Path, int]:
    """Create a temporary request snapshot with machine-only grounding aliases."""
    dry_manifest = load_json(dry_run_manifest_path, {})
    if not isinstance(dry_manifest, dict):
        raise ValueError("Summary dry-run manifest must be a JSON object")
    request_file = Path(str(dry_manifest.get("request_file") or ""))
    requests = load_jsonl(request_file)
    alias_count = 0

    for request in requests:
        source = request.get("source")
        if not isinstance(source, dict):
            continue
        abstract = str(source.get("abstract") or "")
        aliases = tops_grounding_aliases(abstract)
        if not aliases:
            continue
        source["abstract"] = (
            abstract
            + "\nMachine-only numeric grounding aliases: "
            + "; ".join(aliases)
            + "."
        )
        alias_count += len(aliases)

    temporary_request = temporary_root / "summary_requests.jsonl"
    atomic_write(
        temporary_request,
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in requests
        ),
    )
    dry_manifest["request_file"] = str(temporary_request)
    temporary_manifest = temporary_root / "summary_generation_manifest.json"
    atomic_write(
        temporary_manifest,
        json.dumps(dry_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return temporary_manifest, alias_count


class ProductionNormalizingClient:
    """Canonicalize transport fields and evidence-bound semantic labels."""

    def __init__(
        self,
        *,
        base_client: Any,
        diagnostics: TransportRepairDiagnostics,
        abstracts: dict[str, str],
    ) -> None:
        self.base_client = base_client
        self.diagnostics = diagnostics
        self.abstracts = abstracts

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

        normalized, evidence, changed, previous = enforce_onn_architecture(
            normalized,
            abstract=self.abstracts.get(candidate_id),
        )
        if changed:
            self.diagnostics.architecture_repairs += 1
        evidence_record = evidence.as_dict()
        evidence_record.update(
            {
                "model_value": previous,
                "repaired": changed,
            }
        )
        self.diagnostics.architecture_evidence[candidate_id] = evidence_record

        return DeepSeekResponse(
            content=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            usage=response.usage,
            model=response.model,
        )


def diagnostic_payload(diagnostics: TransportRepairDiagnostics) -> dict[str, Any]:
    return {
        "candidate_id_repair_responses": diagnostics.candidate_id_repair_responses,
        "unit_format_normalization_responses": diagnostics.unit_format_normalization_responses,
        "tops_alias_expansions": diagnostics.tops_alias_expansions,
        "architecture_repairs": diagnostics.architecture_repairs,
        "architecture_evidence": diagnostics.architecture_evidence,
    }


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
        abstracts=request_abstracts(dry_run_manifest_path),
    )

    try:
        with tempfile.TemporaryDirectory(prefix="research-inbox-grounding-") as directory:
            grounding_manifest, alias_count = grounding_manifest_copy(
                dry_run_manifest_path,
                Path(directory),
            )
            diagnostics.tops_alias_expansions = alias_count
            state = strict_generate(
                dry_run_manifest_path=grounding_manifest,
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
        persisted["transport_repairs"] = diagnostic_payload(diagnostics)
        atomic_write(
            manifest_path,
            json.dumps(persisted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    state["transport_repairs"] = diagnostic_payload(diagnostics)
    atomic_write(
        manifest_path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate production DeepSeek summaries with evidence guards"
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

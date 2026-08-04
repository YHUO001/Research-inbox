from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from scripts.summarize.deepseek_provider import DeepSeekClient, DeepSeekResponse
from scripts.summarize.evidence_guard import enforce_onn_architecture
from scripts.summarize.fulltext_methods import MethodContext, collect_method_context
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
    generation_metadata_repair_responses: int = 0
    tops_alias_expansions: int = 0
    architecture_repairs: int = 0
    architecture_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    full_text_method_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)


def expected_example(system_prompt: str) -> dict[str, Any]:
    for marker in ("JSON 形状示例:\n", "Example JSON shape:\n"):
        if marker in system_prompt:
            value = json.loads(system_prompt.rsplit(marker, 1)[1])
            if isinstance(value, dict):
                return value
    raise RuntimeError("Summary system prompt is missing its JSON example")


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


def append_method_context(prompt: str, context: MethodContext) -> str:
    headings = "；".join(context.section_headings) or "未识别到明确章节标题"
    return (
        prompt
        + "\n\n公开正文中的方法相关上下文（仅用于定性解释方法；不得据此新增标题或摘要中没有的数字）：\n"
        + f"- 来源：{context.source_url}\n"
        + f"- 章节：{headings}\n"
        + "- 方法上下文开始：\n"
        + context.text
        + "\n- 方法上下文结束。"
    )


def generation_manifest_copy(
    dry_run_manifest_path: Path,
    temporary_root: Path,
    *,
    config: dict[str, Any],
    method_context_loader: Callable[..., MethodContext] = collect_method_context,
) -> tuple[Path, int, dict[str, dict[str, Any]]]:
    """Create an ephemeral request snapshot with grounding aliases and method context."""
    dry_manifest = load_json(dry_run_manifest_path, {})
    if not isinstance(dry_manifest, dict):
        raise ValueError("Summary dry-run manifest must be a JSON object")
    request_file = Path(str(dry_manifest.get("request_file") or ""))
    requests = load_jsonl(request_file)
    alias_count = 0
    context_records: dict[str, dict[str, Any]] = {}
    execution = config.get("execution") or {}
    full_text_config = config.get("full_text") or {}
    full_text_enabled = bool(execution.get("use_full_text")) and bool(
        full_text_config.get("enabled")
    )

    for request in requests:
        candidate_id = str(request["candidate_id"])
        source = request.get("source")
        if not isinstance(source, dict):
            continue
        abstract = str(source.get("abstract") or "")
        aliases = tops_grounding_aliases(abstract)
        if aliases:
            source["abstract"] = (
                abstract
                + "\nMachine-only numeric grounding aliases: "
                + "; ".join(aliases)
                + "."
            )
            alias_count += len(aliases)

        context = MethodContext(candidate_id, "disabled", None, None, [], "")
        if full_text_enabled:
            context_source = dict(source)
            context_source["candidate_id"] = candidate_id
            context = method_context_loader(context_source, config=full_text_config)
        context_records[candidate_id] = context.audit_record()
        if context.status == "used" and context.text:
            request["prompt"] = append_method_context(str(request["prompt"]), context)
            request["information_basis"] = (
                "title_metadata_abstract_and_open_full_text_methods"
            )
            request["full_text_method_source_url"] = context.source_url
            request["full_text_method_section_headings"] = context.section_headings
        else:
            request["information_basis"] = "title_metadata_and_abstract_only"
            request["full_text_method_source_url"] = None
            request["full_text_method_section_headings"] = []

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
    return temporary_manifest, alias_count, context_records


class ProductionNormalizingClient:
    """Canonicalize transport, evidence metadata, and architecture labels."""

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

        expected = expected_example(str(kwargs.get("system_prompt") or ""))
        candidate_id = str(expected.get("candidate_id") or "")
        repaired_metadata = False
        if value.get("candidate_id") != candidate_id:
            self.diagnostics.candidate_id_repair_responses += 1
        value["candidate_id"] = candidate_id
        for key in ("schema_version", "summary_version", "output_language"):
            if value.get(key) != expected.get(key):
                repaired_metadata = True
            value[key] = expected.get(key)

        expected_verification = expected.get("verification") or {}
        verification = value.get("verification")
        if not isinstance(verification, dict):
            verification = {}
            repaired_metadata = True
        for key in (
            "information_basis",
            "full_text_method_context_used",
            "full_text_method_source_url",
            "unsupported_numbers_detected",
        ):
            if verification.get(key) != expected_verification.get(key):
                repaired_metadata = True
            verification[key] = expected_verification.get(key)
        if not isinstance(verification.get("missing_information"), list):
            verification["missing_information"] = []
            repaired_metadata = True
        value["verification"] = verification
        if repaired_metadata:
            self.diagnostics.generation_metadata_repair_responses += 1

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
        evidence_record.update({"model_value": previous, "repaired": changed})
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
        "generation_metadata_repair_responses": diagnostics.generation_metadata_repair_responses,
        "tops_alias_expansions": diagnostics.tops_alias_expansions,
        "architecture_repairs": diagnostics.architecture_repairs,
        "architecture_evidence": diagnostics.architecture_evidence,
        "full_text_method_contexts": diagnostics.full_text_method_contexts,
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
    method_context_loader: Callable[..., MethodContext] = collect_method_context,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    provider = config["provider"]
    execution = config.get("execution") or {}
    full_text_config = config.get("full_text") or {}
    if execution.get("use_full_text") and not full_text_config.get("open_access_only"):
        raise RuntimeError("Full-text method context must remain open-access-only")
    if full_text_config.get("persist_extracted_text"):
        raise RuntimeError("Extracted full text must not be persisted")
    if full_text_config.get("numeric_grounding_scope") != "title_and_abstract_only":
        raise RuntimeError("Numeric grounding must remain limited to title and abstract")

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
            generation_manifest, alias_count, contexts = generation_manifest_copy(
                dry_run_manifest_path,
                Path(directory),
                config=config,
                method_context_loader=method_context_loader,
            )
            diagnostics.tops_alias_expansions = alias_count
            diagnostics.full_text_method_contexts = contexts
            state = strict_generate(
                dry_run_manifest_path=generation_manifest,
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
        persisted["full_text_method_contexts"] = diagnostics.full_text_method_contexts
        persisted["full_text_used"] = any(
            item.get("status") == "used"
            for item in diagnostics.full_text_method_contexts.values()
        )
        persisted["full_text_persisted"] = False
        persisted["numeric_grounding_scope"] = "title_and_abstract_only"
        atomic_write(
            manifest_path,
            json.dumps(persisted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    state["transport_repairs"] = diagnostic_payload(diagnostics)
    state["full_text_method_contexts"] = diagnostics.full_text_method_contexts
    state["full_text_used"] = any(
        item.get("status") == "used"
        for item in diagnostics.full_text_method_contexts.values()
    )
    state["full_text_persisted"] = False
    state["numeric_grounding_scope"] = "title_and_abstract_only"
    atomic_write(
        manifest_path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate production Chinese DeepSeek summaries with evidence guards"
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

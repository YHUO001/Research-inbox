from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from scripts.summarize.prepare_digest import atomic_write, load_json


def validate_automatic_config(config: dict[str, Any]) -> None:
    execution = config.get("execution") or {}
    automation = config.get("automation") or {}
    review = config.get("review") or {}
    if automation.get("enabled") is not True:
        raise RuntimeError("Automatic summary orchestration must be enabled")
    if automation.get("mode") != "automatic_daily_batch":
        raise RuntimeError("Automatic generation requires automatic_daily_batch")
    if automation.get("update_summary_history_after_validation") is not True:
        raise RuntimeError("Automatic generation must complete history after validation")
    if automation.get("all_or_nothing_batch") is not True:
        raise RuntimeError("Automatic generation requires all-or-nothing batches")
    if automation.get("review_required") or review.get("required"):
        raise RuntimeError("Human review must be disabled during automatic generation")
    if execution.get("email_enabled"):
        raise RuntimeError("The generator must not send email directly")
    if automation.get("delivery_mode") != "separate_daily_digest_workflow":
        raise RuntimeError("Email delivery must remain in the daily digest workflow")


def compatibility_config(config: dict[str, Any]) -> dict[str, Any]:
    compatible = deepcopy(config)
    execution = compatible.setdefault("execution", {})
    execution["mode"] = "manual_provider_validation"
    execution["llm_enabled"] = False
    execution["update_summary_history"] = False
    return compatible


def generate_automatic(
    *,
    dry_run_manifest_path: Path,
    summary_schema_path: Path,
    config_path: Path,
    prepared_root: Path,
    output_root: Path,
    manifest_path: Path,
) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Summary generation config must be a YAML object")
    validate_automatic_config(config)

    prepared_root.mkdir(parents=True, exist_ok=True)
    compatibility_path = prepared_root / "automatic_generation_compatibility.yaml"
    atomic_write(
        compatibility_path,
        yaml.safe_dump(
            compatibility_config(config),
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    command = [
        sys.executable,
        "-m",
        "scripts.summarize.staged_summary_pipeline_safe",
        "generate",
        "--dry-run-manifest-path",
        str(dry_run_manifest_path),
        "--summary-schema",
        str(summary_schema_path),
        "--config",
        str(compatibility_path),
        "--prepared-root",
        str(prepared_root),
        "--output-root",
        str(output_root),
        "--manifest-path",
        str(manifest_path),
    ]
    completed = subprocess.run(command, check=False)

    manifest = load_json(manifest_path, {})
    if isinstance(manifest, dict) and manifest:
        manifest["execution_mode"] = "automatic_daily_batch"
        manifest["review_required"] = False
        manifest["automatic_history_pending"] = manifest.get("status") == "completed"
        manifest["knowledge_base_pending"] = manifest.get("status") == "completed"
        manifest["daily_digest_pending"] = manifest.get("status") == "completed"
        atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate validated summaries under the automatic production policy"
    )
    parser.add_argument("--dry-run-manifest-path", type=Path, required=True)
    parser.add_argument("--summary-schema", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return generate_automatic(
        dry_run_manifest_path=args.dry_run_manifest_path,
        summary_schema_path=args.summary_schema,
        config_path=args.config,
        prepared_root=args.prepared_root,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())

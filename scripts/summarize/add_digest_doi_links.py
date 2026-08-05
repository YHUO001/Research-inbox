from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.summarize.add_doi_links import add_links_to_digest
from scripts.summarize.prepare_digest import atomic_write, load_json, load_jsonl, stable_json


def apply_digest_doi_links(
    *, generation_manifest_path: Path, output_root: Path
) -> dict[str, Any]:
    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict):
        raise RuntimeError("Summary generation manifest must be a JSON object")
    if generation.get("status") != "completed":
        raise RuntimeError("DOI links require a completed generation manifest")
    digest_date = str(generation.get("digest_date") or "")
    if not digest_date:
        raise RuntimeError("Summary generation manifest is missing digest_date")

    request_path = output_root / "summary_requests" / f"{digest_date}.jsonl"
    digest_path = output_root / "digests" / f"{digest_date}.generated.md"
    if not request_path.exists() or not digest_path.exists():
        raise RuntimeError("Summary requests and generated digest Markdown are required")

    requests = load_jsonl(request_path)
    digest, replacement_count = add_links_to_digest(
        digest_path.read_text(encoding="utf-8"), requests
    )
    if replacement_count != len(requests):
        raise RuntimeError(
            f"Expected {len(requests)} DOI metadata insertions, got {replacement_count}"
        )
    atomic_write(digest_path, digest)

    state = {
        "status": "completed",
        "digest_date": digest_date,
        "request_count": len(requests),
        "digest_doi_links": replacement_count,
        "doi_source": "request_metadata",
        "model_generated_doi": False,
        "review_artifact_required": False,
    }
    print(stable_json(state), flush=True)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add deterministic DOI links to an automatically generated digest"
    )
    parser.add_argument("--generation-manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    apply_digest_doi_links(
        generation_manifest_path=args.generation_manifest_path,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

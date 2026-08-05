from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from scripts.summarize.prepare_digest import atomic_write, load_json, load_jsonl, stable_json


DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
DOI_RESOLVER_HOSTS = {"doi.org", "dx.doi.org", "www.doi.org"}


def normalize_doi(value: Any) -> str | None:
    """Return a canonical DOI identifier without a resolver prefix."""

    text = unquote(str(value or "").strip())
    if not text:
        return None
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if text.startswith(("http://", "https://")):
        parsed = urlsplit(text)
        if parsed.netloc.lower() not in DOI_RESOLVER_HOSTS:
            return None
        text = parsed.path.lstrip("/").strip()
    if not DOI_PATTERN.fullmatch(text):
        return None
    return text


def doi_markdown(value: Any) -> str:
    doi = normalize_doi(value)
    if not doi:
        return "未提供"
    target = "https://doi.org/" + quote(doi, safe="/")
    return f"[{doi}]({target})"


def _replace_once(text: str, old: str, new: str) -> tuple[str, bool]:
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def add_links_to_digest(markdown: str, requests: list[dict[str, Any]]) -> tuple[str, int]:
    updated = markdown
    replacements = 0
    for request in requests:
        source = request.get("source") or {}
        landing = source.get("landing_page") or "未提供"
        old = f"- 链接：{landing}"
        doi_line = f"- DOI：{doi_markdown(source.get('doi'))}"
        original_line = f"- 原始链接：{landing}"
        updated, changed = _replace_once(updated, old, f"{doi_line}\n{original_line}")
        replacements += int(changed)
    return updated, replacements


def add_links_to_review(markdown: str, requests: list[dict[str, Any]]) -> tuple[str, int]:
    updated = markdown
    replacements = 0
    for request in requests:
        source = request.get("source") or {}
        old = f"- DOI：{source.get('doi') or '未提供'}"
        new = f"- DOI：{doi_markdown(source.get('doi'))}"
        updated, changed = _replace_once(updated, old, new)
        replacements += int(changed)
    return updated, replacements


def apply_doi_links(*, generation_manifest_path: Path, output_root: Path) -> dict[str, Any]:
    generation = load_json(generation_manifest_path, {})
    if not isinstance(generation, dict):
        raise RuntimeError("Summary generation manifest must be a JSON object")
    digest_date = str(generation.get("digest_date") or "")
    if not digest_date:
        raise RuntimeError("Summary generation manifest is missing digest_date")

    request_path = output_root / "summary_requests" / f"{digest_date}.jsonl"
    digest_path = output_root / "digests" / f"{digest_date}.generated.md"
    review_path = output_root / "reviews" / f"{digest_date}.review.md"
    if not request_path.exists():
        raise RuntimeError(f"Summary request file is missing: {request_path}")
    if not digest_path.exists() or not review_path.exists():
        raise RuntimeError("Generated digest and review Markdown are required before adding DOI links")

    requests = load_jsonl(request_path)
    digest, digest_count = add_links_to_digest(
        digest_path.read_text(encoding="utf-8"), requests
    )
    review, review_count = add_links_to_review(
        review_path.read_text(encoding="utf-8"), requests
    )
    atomic_write(digest_path, digest)
    atomic_write(review_path, review)

    state = {
        "status": "completed",
        "digest_date": digest_date,
        "request_count": len(requests),
        "digest_doi_links": digest_count,
        "review_doi_links": review_count,
        "doi_source": "request_metadata",
        "model_generated_doi": False,
    }
    print(stable_json(state), flush=True)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add deterministic DOI links to generated summary and review Markdown"
    )
    parser.add_argument(
        "--generation-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime-state/data"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    apply_doi_links(
        generation_manifest_path=args.generation_manifest_path,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

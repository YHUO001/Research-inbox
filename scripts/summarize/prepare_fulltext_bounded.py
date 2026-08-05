from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from scripts.summarize.fulltext_methods import MethodContext
from scripts.summarize.prepare_digest import stable_json
from scripts.summarize.springer_openaccess import (
    api_audit_url,
    collect_official_or_open_context,
    normalize_doi,
)
from scripts.summarize.staged_summary_pipeline import prepare_stage


def _loader_worker(
    source: dict[str, Any],
    config: dict[str, Any],
    loader: Callable[..., MethodContext],
    output: Any,
) -> None:
    try:
        output.put(("ok", loader(source, config=config)))
    except BaseException as error:  # The parent converts optional retrieval failures to fallback.
        output.put(("error", type(error).__name__, str(error)[:300]))


def _timeout_source_url(source: dict[str, Any], config: dict[str, Any]) -> str | None:
    doi = normalize_doi(source.get("doi"))
    if not doi:
        return None
    endpoint = str(
        config.get("springer_openaccess_endpoint")
        or "https://api.springernature.com/openaccess/jats"
    ).rstrip("?")
    return api_audit_url(endpoint, doi)


def bounded_collect_method_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    loader: Callable[..., MethodContext] = collect_official_or_open_context,
) -> MethodContext:
    """Run one optional full-text lookup in a killable child process.

    Socket inactivity timeouts and Python signals do not reliably interrupt a
    blocked TLS read. GitHub-hosted runners are Linux, so a forked child can be
    terminated at a strict wall-clock deadline. Timeout remains a non-fatal
    evidence fallback: generation continues with title, metadata, and abstract.
    """

    candidate_id = str(source.get("candidate_id") or source.get("id") or "unknown")
    timeout_seconds = float(config.get("candidate_timeout_seconds") or 30)
    if timeout_seconds <= 0:
        return loader(source, config=config)

    available = multiprocessing.get_all_start_methods()
    if "fork" not in available:
        return loader(source, config=config)

    context = multiprocessing.get_context("fork")
    output = context.Queue(maxsize=1)
    process = context.Process(
        target=_loader_worker,
        args=(source, config, loader, output),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(5)
        output.close()
        return MethodContext(
            candidate_id=candidate_id,
            status="timed_out",
            source_url=_timeout_source_url(source, config),
            media_type="application/xml+jats",
            section_headings=[],
            text="",
            error=(
                f"full-text child process exceeded {timeout_seconds:g} seconds; "
                "fell back to abstract"
            ),
        )

    try:
        result = output.get(timeout=1)
    except Empty:
        return MethodContext(
            candidate_id=candidate_id,
            status="not_available",
            source_url=_timeout_source_url(source, config),
            media_type="application/xml+jats",
            section_headings=[],
            text="",
            error="full-text child process exited without a result; fell back to abstract",
        )
    finally:
        output.close()

    if result[0] == "ok":
        return result[1]
    return MethodContext(
        candidate_id=candidate_id,
        status="not_available",
        source_url=_timeout_source_url(source, config),
        media_type="application/xml+jats",
        section_headings=[],
        text="",
        error=f"full-text child error {result[1]}; fell back to abstract",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare optional method context through official OA APIs with hard fallback"
    )
    parser.add_argument(
        "--dry-run-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_generation_manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/summary_generation.yaml"),
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = prepare_stage(
        dry_run_manifest_path=args.dry_run_manifest_path,
        config_path=args.config,
        prepared_root=args.prepared_root,
        method_context_loader=bounded_collect_method_context,
    )
    print(stable_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

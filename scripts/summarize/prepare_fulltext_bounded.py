from __future__ import annotations

import argparse
import signal
from pathlib import Path
from types import FrameType
from typing import Any, Callable

from scripts.summarize.fulltext_methods import (
    MethodContext,
    candidate_urls,
    collect_method_context,
)
from scripts.summarize.prepare_digest import stable_json
from scripts.summarize.staged_summary_pipeline import prepare_stage


class FullTextCandidateTimeout(TimeoutError):
    """Raised when one paper exceeds its full-text preparation budget."""


def _raise_candidate_timeout(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise FullTextCandidateTimeout("full-text candidate hard timeout")


def bounded_collect_method_context(
    source: dict[str, Any],
    *,
    config: dict[str, Any],
    loader: Callable[..., MethodContext] = collect_method_context,
) -> MethodContext:
    """Run one optional full-text lookup with a process-level hard deadline.

    GitHub-hosted runners use Linux, where SIGALRM interrupts a blocking
    urllib read even when the remote server keeps the socket alive by
    slowly sending data. A timeout is a non-fatal evidence fallback: the
    model still runs using title, metadata, and abstract.
    """

    candidate_id = str(source.get("candidate_id") or source.get("id") or "unknown")
    timeout_seconds = float(config.get("candidate_timeout_seconds") or 45)
    urls = candidate_urls(source, int(config.get("candidate_url_limit") or 3))

    if timeout_seconds <= 0 or not hasattr(signal, "setitimer"):
        return loader(source, config=config)

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _raise_candidate_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return loader(source, config=config)
    except FullTextCandidateTimeout:
        return MethodContext(
            candidate_id=candidate_id,
            status="timed_out",
            source_url=urls[0] if urls else None,
            media_type=None,
            section_headings=[],
            text="",
            error=f"candidate hard timeout after {timeout_seconds:g} seconds; fell back to abstract",
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare open full-text method context with a hard per-paper timeout"
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

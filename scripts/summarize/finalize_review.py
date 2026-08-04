from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.summarize.prepare_digest import atomic_write, load_json, stable_json


ALLOWED_DECISIONS = {"approve_all", "hold_for_revision"}
CONFIRMATION_PHRASE = "REVIEWED"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_state_path(state_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else state_root / path


def verify_artifacts(packet: dict[str, Any], state_root: Path) -> None:
    artifacts = packet.get("artifacts") or {}
    for name in ("request", "summary", "digest_json"):
        path = resolve_state_path(state_root, str(artifacts.get(f"{name}_file") or ""))
        expected = str(artifacts.get(f"{name}_sha256") or "")
        if not path.exists() or not expected:
            raise RuntimeError(f"Missing review artifact: {name}")
        if file_sha256(path) != expected:
            raise RuntimeError(f"Review artifact changed after packet creation: {name}")


def approved_digest_markdown(content: str, *, reviewer: str, reviewed_at: str) -> str:
    marker = "\n## Human review\n"
    if marker in content:
        content = content.split(marker, 1)[0].rstrip()
    return (
        content.rstrip()
        + marker
        + f"\n- Decision: `approved`\n- Reviewer: `{reviewer}`\n"
        + f"- Reviewed at: `{reviewed_at}`\n- Email sent: `false`\n"
    )


def finalize_review(
    *,
    state_root: Path,
    review_manifest_path: Path,
    history_path: Path,
    decision: str,
    confirmation: str,
    reviewer: str,
    reviewed_at: str,
    notes: str = "",
) -> dict[str, Any]:
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported review decision: {decision}")
    if confirmation != CONFIRMATION_PHRASE:
        raise RuntimeError(f"Confirmation must be exactly {CONFIRMATION_PHRASE}")
    if not reviewer.strip() or not reviewed_at.strip():
        raise ValueError("Reviewer and reviewed_at are required")

    review_state = load_json(review_manifest_path, {})
    if not isinstance(review_state, dict):
        raise ValueError("Review manifest must be a JSON object")
    review_json_path = resolve_state_path(
        state_root, str(review_state.get("review_json_file") or "")
    )
    packet = load_json(review_json_path, {})
    if not isinstance(packet, dict):
        raise ValueError("Review packet must be a JSON object")
    if packet.get("status") == "approved" and decision == "approve_all":
        return review_state
    if packet.get("status") not in {"pending_human_review", "revision_requested"}:
        raise RuntimeError("Review packet is not awaiting a human decision")

    verify_artifacts(packet, state_root)
    papers = packet.get("papers") or []
    if not papers or len(papers) != int(packet.get("paper_count") or 0):
        raise RuntimeError("Review packet paper count is invalid")
    for paper in papers:
        checks = paper.get("automated_checks") or {}
        if not checks.get("schema_valid") or checks.get("unsupported_numeric_claims"):
            raise RuntimeError("Automated checks must pass before human finalization")
        if not checks.get("architecture_consistent"):
            raise RuntimeError("Architecture evidence must be canonicalized before review")

    packet["batch_review"] = {
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "notes": notes,
    }
    digest_json_path = resolve_state_path(
        state_root, str((packet.get("artifacts") or {})["digest_json_file"])
    )
    digest = load_json(digest_json_path, {})
    if not isinstance(digest, dict):
        raise ValueError("Generated digest must be a JSON object")

    if decision == "hold_for_revision":
        packet["status"] = "revision_requested"
        packet["safety"]["summary_history_updated"] = False
        digest["status"] = "revision_requested"
        digest["review"] = packet["batch_review"]
        review_state.update(
            {
                "status": "revision_requested",
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "summary_history_updated": False,
                "email_enabled": False,
            }
        )
    else:
        history = load_json(
            history_path,
            {
                "schema_version": 1,
                "completed_candidate_ids": {},
                "failed_candidate_ids": {},
            },
        )
        if not isinstance(history, dict):
            raise ValueError("Summary history must be a JSON object")
        completed = history.setdefault("completed_candidate_ids", {})
        if not isinstance(completed, dict):
            raise ValueError("completed_candidate_ids must be an object")
        history.setdefault("failed_candidate_ids", {})

        for paper in papers:
            candidate_id = str(paper["candidate_id"])
            completed[candidate_id] = {
                "completed_at": reviewed_at,
                "digest_date": packet["digest_date"],
                "model": packet["model"],
                "provider": packet["provider"],
                "request_id": paper.get("request_id"),
                "summary_sha256": paper["summary_sha256"],
                "reviewer": reviewer,
                "review_decision": "approved",
            }
        atomic_write(
            history_path,
            json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

        packet["status"] = "approved"
        packet["safety"]["summary_history_updated"] = True
        digest["status"] = "approved_human_review"
        digest["review"] = packet["batch_review"]
        digest["safety"]["summary_history_updated"] = True
        review_state.update(
            {
                "status": "approved",
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "completed_candidate_count": len(papers),
                "summary_history_updated": True,
                "email_enabled": False,
            }
        )

        digest_markdown_path = (
            state_root
            / "data"
            / "digests"
            / f"{packet['digest_date']}.generated.md"
        )
        atomic_write(
            digest_markdown_path,
            approved_digest_markdown(
                digest_markdown_path.read_text(encoding="utf-8"),
                reviewer=reviewer,
                reviewed_at=reviewed_at,
            ),
        )

    atomic_write(
        digest_json_path,
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        review_json_path,
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    review_state["review_json_sha256"] = file_sha256(review_json_path)
    atomic_write(
        review_manifest_path,
        json.dumps(review_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return review_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a human summary review")
    parser.add_argument("--state-root", type=Path, default=Path("runtime-state"))
    parser.add_argument(
        "--review-manifest-path",
        type=Path,
        default=Path("runtime-state/state/summary_review_manifest.json"),
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("runtime-state/state/summary_history.json"),
    )
    parser.add_argument("--decision", choices=sorted(ALLOWED_DECISIONS), required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    state = finalize_review(
        state_root=args.state_root,
        review_manifest_path=args.review_manifest_path,
        history_path=args.history_path,
        decision=args.decision,
        confirmation=args.confirmation,
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
        notes=args.notes,
    )
    print(stable_json(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

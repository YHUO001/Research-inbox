from __future__ import annotations

from pathlib import Path

from scripts.delivery import send_daily_digest as base
from scripts.summarize.prepare_digest import load_json


_ORIGINAL_EMPTY_DIGEST = base.empty_digest_body


def evidence_aware_empty_digest_body(state_root: Path, digest_date: str) -> str:
    generation = load_json(state_root / "state" / "summary_generation_manifest.json", {})
    if isinstance(generation, dict):
        skipped = int(generation.get("skipped_no_abstract_count") or 0)
        request_count = int(generation.get("request_count") or 0)
        status = str(generation.get("status") or "")
        if skipped > 0 and request_count == 0 and status == "no_eligible_evidence":
            selection = load_json(state_root / "state" / "selection_manifest.json", {})
            selected = int((selection or {}).get("summary_slot_count") or 0)
            return "\n".join(
                [
                    f"# 每日研究汇总 {digest_date}",
                    "",
                    "今天有论文进入摘要名额，但没有论文具备最低可用证据，因此未生成正式摘要。",
                    "",
                    "## 证据安全处理",
                    "",
                    f"- 原始摘要名额：`{selected}`",
                    f"- 跳过论文数：`{skipped}`",
                    "- 跳过条件：公开全文连续获取三次仍失败，并且没有可用摘要或摘要片段。",
                    "- DeepSeek：未调用",
                    "- summary_history：未更新",
                    "- 后续行为：候选保留，后续运行可在元数据或摘要补全后重试。",
                ]
            )
    return _ORIGINAL_EMPTY_DIGEST(state_root, digest_date)


def main() -> int:
    base.empty_digest_body = evidence_aware_empty_digest_body
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

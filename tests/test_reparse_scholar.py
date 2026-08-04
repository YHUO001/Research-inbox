from scripts.maintenance.reparse_scholar import replace_scholar_records


def record(candidate_id: str, message_id: str, fingerprint: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "content_fingerprint": fingerprint,
        "source": {"message_id": message_id},
    }


def test_replace_target_messages_and_keep_other_sources() -> None:
    existing = [
        record("old-a", "message-a", "old-fingerprint"),
        record("other", "other-message", "other-fingerprint"),
    ]
    repaired = [
        record("new-a", "message-a", "new-fingerprint"),
        record("duplicate", "message-b", "other-fingerprint"),
    ]
    rebuilt, duplicates = replace_scholar_records(
        existing,
        target_message_ids={"message-a", "message-b"},
        repaired=repaired,
    )
    assert [item["candidate_id"] for item in rebuilt] == ["other", "new-a"]
    assert duplicates == 1

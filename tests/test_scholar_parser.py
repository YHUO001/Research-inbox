from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from scripts.ingest.scholar_parser import SourceContext, parse_alert_body

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "scholar_optical_nn_body.txt"
SCHEMA = ROOT / "schemas" / "alert_candidate.schema.json"


def source_context() -> SourceContext:
    return SourceContext(
        message_id="test-message-001",
        thread_id="test-thread-001",
        received_at="2026-07-26T11:38:46Z",
        sender="scholaralerts-noreply@google.com",
        subject='"optical neural network" - 新的结果',
        spf="pass",
        dkim="pass",
    )


def validate(records: list[dict]) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for record in records:
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        assert not errors, "\n".join(error.message for error in errors)


def test_parse_gmail_normalized_body() -> None:
    records = parse_alert_body(
        FIXTURE.read_text(encoding="utf-8"),
        source_context(),
        content_type="text",
        extracted_at="2026-08-03T09:00:00Z",
    )

    assert len(records) == 2
    validate(records)

    first = records[0]
    assert first["position_in_message"] == 0
    assert first["source"]["alert_name"] == "optical neural network"
    assert first["title"] == (
        "Distillation-guided optical neural networks with "
        "reinforcement learning-assisted calibration"
    )
    assert first["venue"]["normalized"] == "Optica"
    assert first["year"] == 2026
    assert first["authors"][0]["name"] == "K Di"
    assert len(first["authors"]) == 8
    assert first["links"]["primary_url"] == (
        "https://opg.optica.org/optica/fulltext.cfm?uri=optica-13-8-1423"
    )
    assert first["parse_status"]["state"] == "complete"

    second = records[1]
    assert second["venue"]["normalized"] == "Photonics for Quantum"
    assert second["year"] == 2026
    assert second["identifiers"]["doi"]["value"] == "10.1117/12.3110867"
    assert second["identifiers"]["doi"]["source"] == "url"


def test_parse_minimal_html_body() -> None:
    html = """
    <html><body>
      <div class="result">
        <a href="https://scholar.google.com/scholar_url?url=https%3A%2F%2Fexample.org%2Fpaper-one">
          Query-efficient zeroth-order training for language models
        </a>
        <div>A Author, B Author - NeurIPS, 2026</div>
        <div>We reduce the number of function evaluations using query reuse.</div>
        <a href="https://scholar.google.com/citations?update_op=email_library_add">保存</a>
      </div>
      <div class="result">
        <a href="https://scholar.google.com/scholar_url?url=https%3A%2F%2Fexample.org%2Fpaper-two">
          Zeroth-order optimization of photonic neural networks
        </a>
        <div>C Author - Optica, 2026</div>
        <div>A black-box method for physical optical hardware.</div>
      </div>
    </body></html>
    """

    records = parse_alert_body(
        html,
        source_context(),
        content_type="html",
        extracted_at="2026-08-03T09:00:00Z",
    )

    assert len(records) == 2
    validate(records)
    assert records[0]["venue"]["normalized"] == "NeurIPS"
    assert records[1]["venue"]["normalized"] == "Optica"
    assert records[1]["links"]["primary_url"] == "https://example.org/paper-two"

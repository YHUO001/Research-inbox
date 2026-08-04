from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator, FormatChecker

CLASSIFIER_VERSION = 2
ROUTES = (
    "mandatory_summary_queue",
    "standard_scoring_queue",
    "metadata_enrichment_queue",
    "manual_review_queue",
    "archive",
)


@dataclass(frozen=True)
class Match:
    field: str
    pattern: str
    matched_text: str

    def as_evidence(self) -> dict[str, str]:
        return {
            "field": self.field,
            "pattern": self.pattern,
            "matched_text": self.matched_text,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", value or "").lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()


def candidate_fields(candidate: dict[str, Any]) -> dict[str, str]:
    venue_obj = candidate.get("venue") or {}
    authors = candidate.get("authors") or []
    return {
        "title": str(candidate.get("title") or ""),
        "snippet": str(candidate.get("snippet") or ""),
        "venue": str(venue_obj.get("normalized") or venue_obj.get("raw") or ""),
        "metadata_line": str(candidate.get("raw_metadata_line") or ""),
        "authors": ", ".join(
            str(author.get("name") or "")
            for author in authors
            if isinstance(author, dict)
        ),
    }


def first_matches(
    fields: dict[str, str],
    patterns: Iterable[str],
    *,
    inspect: Iterable[str] = ("title", "snippet", "venue", "metadata_line"),
) -> list[Match]:
    matches: list[Match] = []
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for field in inspect:
            text = fields.get(field, "")
            hit = compiled.search(text)
            if hit:
                matches.append(Match(field, pattern, hit.group(0)))
                break
    return matches


def normalized_venue(value: str | None) -> str:
    text = normalize_text(value)
    text = text.replace("…", " ").replace("...", " ")
    text = text.replace("&", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def venue_policy(candidate: dict[str, Any], venues: dict[str, Any]) -> dict[str, Any]:
    raw = str((candidate.get("venue") or {}).get("normalized") or "")
    value = normalized_venue(raw)
    source_type = str((candidate.get("source") or {}).get("source_type") or "")
    allowed_sources = set(venues.get("scope", {}).get("applies_to_sources", []))

    if source_type not in allowed_sources:
        return {
            "matched_tier": None,
            "matched_venue": None,
            "policy_action": "unresolved_venue",
        }

    for tier, group in venues.get("venue_groups", {}).items():
        for item in group.get("venues", []):
            canonical = str(item.get("name") or "")
            alternatives = [canonical, *item.get("aliases", [])]
            for alternative in alternatives:
                target = normalized_venue(str(alternative))
                exact = bool(value and target and value == target)
                truncated = bool(
                    value
                    and target
                    and len(value) >= 7
                    and (
                        "…" in raw
                        or "..." in raw
                        or raw.rstrip().endswith("&")
                    )
                    and target.startswith(value)
                )
                if exact or truncated:
                    return {
                        "matched_tier": tier,
                        "matched_venue": canonical,
                        "policy_action": str(group.get("policy", {}).get("action")),
                    }

    return {
        "matched_tier": "tier_3_and_unlisted" if raw else None,
        "matched_venue": raw or None,
        "policy_action": (
            "summarize_only_when_exceptionally_relevant"
            if raw
            else "unresolved_venue"
        ),
    }


def priority_features(
    fields: dict[str, str],
    feature_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    features: list[dict[str, Any]] = []
    warnings: list[str] = []
    explicit_query_evidence = first_matches(
        fields,
        feature_config.get("query_efficiency", {}).get("patterns", []),
    )
    per_step_evidence = first_matches(
        fields,
        feature_config.get("per_step_query_count", {}).get("patterns", []),
    )
    if per_step_evidence and not explicit_query_evidence:
        warnings.append("per_step_query_count_not_total_query_reduction")

    for name, item in feature_config.items():
        if name == "per_step_query_count":
            continue
        evidence = first_matches(fields, item.get("patterns", []))
        if not evidence:
            continue
        features.append(
            {
                "feature": name,
                "weight": float(item.get("weight", 0)),
                "evidence": [match.as_evidence() for match in evidence],
            }
        )
    return features, warnings


def classify_zo(
    fields: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool, list[str]]:
    negative = first_matches(fields, config.get("negative_patterns", []))
    method_title = first_matches(
        fields,
        config.get("method_patterns", []),
        inspect=("title",),
    )
    method_body = first_matches(
        fields,
        config.get("method_patterns", []),
        inspect=("snippet", "metadata_line"),
    )
    alias_title = first_matches(
        fields,
        config.get("method_aliases", []),
        inspect=("title",),
    )
    alias_body = first_matches(
        fields,
        config.get("method_aliases", []),
        inspect=("snippet", "metadata_line"),
    )
    optimization_context = first_matches(
        fields,
        config.get("optimization_context_patterns", []),
    )
    method_evidence = [*method_title, *method_body]
    alias_evidence = [*alias_title, *alias_body]

    if not method_evidence and not (alias_evidence and optimization_context):
        return None, False, []

    ambiguous = first_matches(fields, config.get("ambiguous_aliases", []))
    if ambiguous and not method_evidence:
        confidence = "uncertain"
        matched_rules = ["ambiguous_method_alias"]
    elif method_title or alias_title:
        confidence = "confirmed"
        matched_rules = ["zo_method_in_title"]
    else:
        confidence = "probable"
        matched_rules = ["zo_method_in_context"]

    optical_context = first_matches(
        fields,
        config.get("optical_context_patterns", []),
    )
    llm_context = first_matches(
        fields,
        config.get("llm_context_patterns", []),
    )
    if optical_context:
        matched_rules.append("optical_zo")
    if llm_context:
        matched_rules.append("llm_zo")

    features, warnings = priority_features(
        fields,
        config.get("priority_features", {}),
    )
    return (
        {
            "project_id": "zeroth-order-optimization",
            "confidence": confidence,
            "matched_rules": sorted(set(matched_rules)),
            "positive_evidence": [
                match.as_evidence()
                for match in [
                    *method_evidence,
                    *alias_evidence,
                    *optical_context,
                    *llm_context,
                ]
            ],
            "negative_evidence": [match.as_evidence() for match in negative],
            "priority_features": features,
        },
        bool(optical_context),
        warnings,
    )


def classify_optical(
    fields: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    negative = first_matches(fields, config.get("negative_patterns", []))
    precise_title = first_matches(
        fields,
        config.get("high_precision_patterns", []),
        inspect=("title",),
    )
    precise_body = first_matches(
        fields,
        config.get("high_precision_patterns", []),
        inspect=("snippet", "metadata_line"),
    )
    hardware = first_matches(
        fields,
        config.get("optical_hardware_patterns", []),
    )
    neural = first_matches(
        fields,
        config.get("neural_context_patterns", []),
    )

    if precise_title:
        confidence = "confirmed"
        rules = ["high_precision_optical_title"]
    elif precise_body or (hardware and neural):
        confidence = "probable"
        rules = [
            "high_precision_optical_context"
            if precise_body
            else "optical_hardware_and_neural_context"
        ]
    else:
        return None

    return {
        "project_id": "optical-neural-networks",
        "confidence": confidence,
        "matched_rules": rules,
        "positive_evidence": [
            match.as_evidence()
            for match in [*precise_title, *precise_body, *hardware, *neural]
        ],
        "negative_evidence": [match.as_evidence() for match in negative],
        "priority_features": [],
    }


def route_candidate(
    candidate: dict[str, Any],
    projects: list[dict[str, Any]],
    venue: dict[str, Any],
    *,
    optical_zo: bool,
) -> dict[str, Any]:
    parse_state = str((candidate.get("parse_status") or {}).get("state") or "partial")
    venue_missing = not (candidate.get("venue") or {}).get("normalized")

    if parse_state in {"manual_review", "rejected"}:
        return {
            "route": "manual_review_queue",
            "priority": "normal",
            "mandatory": False,
            "reasons": ["parser_manual_review"],
            "requires_semantic_scoring": False,
            "requires_manual_review": True,
            "overflow_action": None,
        }

    if venue.get("matched_tier") == "tier_1_must_summarize":
        return {
            "route": "mandatory_summary_queue",
            "priority": "highest",
            "mandatory": True,
            "reasons": ["tier_1_alert_venue"],
            "requires_semantic_scoring": False,
            "requires_manual_review": False,
            "overflow_action": "carry_forward",
        }

    if optical_zo:
        return {
            "route": "mandatory_summary_queue",
            "priority": "highest",
            "mandatory": True,
            "reasons": ["optical_zo_project_override"],
            "requires_semantic_scoring": False,
            "requires_manual_review": False,
            "overflow_action": "carry_forward",
        }

    if not projects:
        return {
            "route": "archive",
            "priority": "low",
            "mandatory": False,
            "reasons": ["no_project_match"],
            "requires_semantic_scoring": False,
            "requires_manual_review": False,
            "overflow_action": None,
        }

    if any(project["confidence"] == "uncertain" for project in projects):
        return {
            "route": "manual_review_queue",
            "priority": "normal",
            "mandatory": False,
            "reasons": ["ambiguous_method_alias"],
            "requires_semantic_scoring": False,
            "requires_manual_review": True,
            "overflow_action": None,
        }

    if parse_state == "partial" or venue_missing:
        reasons = ["missing_metadata"]
        if parse_state == "partial":
            reasons.append("parser_partial")
        if venue_missing:
            reasons.append("unresolved_venue")
        return {
            "route": "metadata_enrichment_queue",
            "priority": "high",
            "mandatory": False,
            "reasons": sorted(set(reasons)),
            "requires_semantic_scoring": False,
            "requires_manual_review": False,
            "overflow_action": None,
        }

    reasons = ["project_match"]
    if venue.get("matched_tier") == "tier_2_relevance_gated":
        reasons.append("tier_2_alert_venue")
    if any(project.get("priority_features") for project in projects):
        reasons.append("priority_feature_match")
    return {
        "route": "standard_scoring_queue",
        "priority": (
            "high"
            if any(project["confidence"] == "confirmed" for project in projects)
            else "normal"
        ),
        "mandatory": False,
        "reasons": sorted(set(reasons)),
        "requires_semantic_scoring": True,
        "requires_manual_review": False,
        "overflow_action": None,
    }


def classify_candidate(
    candidate: dict[str, Any],
    *,
    recognition_config: dict[str, Any],
    venues_config: dict[str, Any],
    classified_at: str | None = None,
) -> dict[str, Any]:
    fields = candidate_fields(candidate)
    projects: list[dict[str, Any]] = []
    warnings: list[str] = []

    zo, optical_zo, zo_warnings = classify_zo(
        fields,
        recognition_config["projects"]["zeroth-order-optimization"],
    )
    if zo:
        projects.append(zo)
    warnings.extend(zo_warnings)

    optical = classify_optical(
        fields,
        recognition_config["projects"]["optical-neural-networks"],
    )
    if optical:
        projects.append(optical)

    venue = venue_policy(candidate, venues_config)
    routing = route_candidate(
        candidate,
        projects,
        venue,
        optical_zo=optical_zo,
    )
    return {
        "schema_version": 1,
        "candidate_id": str(candidate["candidate_id"]),
        "classifier_version": CLASSIFIER_VERSION,
        "matched_projects": projects,
        "venue_policy": venue,
        "routing": routing,
        "classifier_warnings": sorted(set(warnings)),
        "classified_at": classified_at or str(candidate.get("extracted_at") or utc_now()),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def validate_result(result: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(result), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(error.message for error in errors)
        raise ValueError(f"Recognition result failed schema validation: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one parsed paper candidate")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rules", type=Path, default=Path("config/recognition_rules.yaml"))
    parser.add_argument("--venues", type=Path, default=Path("config/venues.yaml"))
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/recognition_result.schema.json"),
    )
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = classify_candidate(
        candidate,
        recognition_config=load_yaml(args.rules),
        venues_config=load_yaml(args.venues),
    )
    validate_result(result, json.loads(args.schema.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

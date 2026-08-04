from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FREE_SPACE_PATTERNS = (
    re.compile(r"\bfree[- ]space\b", re.IGNORECASE),
    re.compile(r"\bspatial light modulator\b|\bSLM\b", re.IGNORECASE),
    re.compile(r"\bdiffractive optical (?:neural )?network\b", re.IGNORECASE),
    re.compile(r"\b4f optical system\b", re.IGNORECASE),
)
_INTEGRATED_PATTERNS = (
    re.compile(r"\bon[- ]chip\b", re.IGNORECASE),
    re.compile(r"\bchip[- ]scale\b", re.IGNORECASE),
    re.compile(r"\bmonolithic chip\b", re.IGNORECASE),
    re.compile(r"\bphotonic integrated circuit\b", re.IGNORECASE),
    re.compile(r"\bintegrated photonic (?:circuit|chip|platform|processor|system)\b", re.IGNORECASE),
    re.compile(r"\bnano[- ]?printing\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class ArchitectureEvidence:
    resolved_type: str
    free_space_evidence: tuple[str, ...]
    integrated_evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolved_type": self.resolved_type,
            "free_space_evidence": list(self.free_space_evidence),
            "integrated_evidence": list(self.integrated_evidence),
        }


def _evidence_sentences(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[str, ...]:
    matches: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        normalized = " ".join(sentence.split())
        if not normalized:
            continue
        if any(pattern.search(normalized) for pattern in patterns):
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                matches.append(normalized)
    return tuple(matches)


def resolve_onn_architecture(abstract: str | None) -> ArchitectureEvidence:
    """Resolve architecture only from explicit abstract evidence.

    Generic mentions such as an application involving an integrated microresonator do
    not establish that the computing architecture itself is integrated. The patterns
    are intentionally narrow and auditable.
    """
    text = str(abstract or "")
    free_space = _evidence_sentences(text, _FREE_SPACE_PATTERNS)
    integrated = _evidence_sentences(text, _INTEGRATED_PATTERNS)
    if free_space and integrated:
        resolved = "hybrid"
    elif free_space:
        resolved = "free_space"
    elif integrated:
        resolved = "integrated"
    else:
        resolved = "unclear"
    return ArchitectureEvidence(
        resolved_type=resolved,
        free_space_evidence=free_space,
        integrated_evidence=integrated,
    )


def enforce_onn_architecture(
    summary: dict[str, Any], *, abstract: str | None
) -> tuple[dict[str, Any], ArchitectureEvidence, bool, str | None]:
    """Replace an ONN architecture label with the deterministic evidence result."""
    evidence = resolve_onn_architecture(abstract)
    analysis = summary.get("optical_neural_network_analysis")
    if not isinstance(analysis, dict):
        return summary, evidence, False, None

    previous = str(analysis.get("architecture_type") or "")
    changed = previous != evidence.resolved_type
    analysis["architecture_type"] = evidence.resolved_type
    return summary, evidence, changed, previous

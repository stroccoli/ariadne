from __future__ import annotations

import re
from typing import Sequence

from evals.benchmark_models import (
    ActionQualityResult,
    ConceptCriterion,
    IncidentSample,
    RootCauseQualityResult,
)


UNCERTAINTY_TERMS = (
    "uncertain",
    "unclear",
    "ambiguous",
    "mixed evidence",
    "conflicting",
    "insufficient evidence",
    "incomplete evidence",
    "not enough evidence",
    "cannot determine",
    "unable to determine",
)

_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def criterion(label: str, *keywords: str) -> ConceptCriterion:
    return ConceptCriterion(label=label, keywords=tuple(_normalize_for_match(keyword) for keyword in keywords))


def evaluate_root_cause_quality(sample: IncidentSample, root_cause: str) -> RootCauseQualityResult:
    normalized_root_cause = _normalize_for_match(root_cause)
    matched = tuple(
        criterion_definition.label
        for criterion_definition in sample.root_cause_rubric.required_concepts
        if _criterion_is_matched(normalized_root_cause, criterion_definition)
    )
    missed = tuple(
        criterion_definition.label
        for criterion_definition in sample.root_cause_rubric.required_concepts
        if criterion_definition.label not in matched
    )
    forbidden_hits = tuple(
        term
        for term in sample.root_cause_rubric.forbidden_terms
        if _normalize_for_match(term) in normalized_root_cause
    )
    uncertainty_satisfied = (
        True
        if not sample.root_cause_rubric.require_uncertainty
        else any(_normalize_for_match(term) in normalized_root_cause for term in UNCERTAINTY_TERMS)
    )

    concept_total = len(sample.root_cause_rubric.required_concepts)
    coverage_score = len(matched) / concept_total if concept_total else 1.0
    uncertainty_score = 1.0 if uncertainty_satisfied else 0.0
    forbidden_penalty = 0.25 if forbidden_hits else 0.0
    score = max(0.0, min(1.0, (0.85 * coverage_score) + (0.15 * uncertainty_score) - forbidden_penalty))

    return RootCauseQualityResult(
        score=round(score, 2),
        matched_concepts=matched,
        missed_concepts=missed,
        forbidden_hits=forbidden_hits,
        uncertainty_satisfied=uncertainty_satisfied,
    )


def evaluate_action_quality(sample: IncidentSample, actions: Sequence[str]) -> ActionQualityResult:
    normalized_actions = tuple(_normalize_for_match(action) for action in actions if _normalize_for_match(action))
    combined_text = " ".join(normalized_actions)
    matched = tuple(
        criterion_definition.label
        for criterion_definition in sample.action_rubric.required_concepts
        if _criterion_is_matched(combined_text, criterion_definition)
    )
    missed = tuple(
        criterion_definition.label
        for criterion_definition in sample.action_rubric.required_concepts
        if criterion_definition.label not in matched
    )
    discouraged_hits = tuple(
        phrase
        for phrase in sample.action_rubric.discouraged_phrases
        if _normalize_for_match(phrase) in combined_text
    )
    action_count = len(normalized_actions)
    minimum_actions_met = action_count >= sample.action_rubric.minimum_actions

    concept_total = len(sample.action_rubric.required_concepts)
    coverage_score = len(matched) / concept_total if concept_total else 1.0
    count_score = min(action_count / sample.action_rubric.minimum_actions, 1.0) if sample.action_rubric.minimum_actions else 1.0
    discouraged_penalty = 0.15 * min(1.0, len(discouraged_hits) / max(action_count, 1))
    score = max(0.0, min(1.0, (0.75 * coverage_score) + (0.25 * count_score) - discouraged_penalty))

    return ActionQualityResult(
        score=round(score, 2),
        matched_concepts=matched,
        missed_concepts=missed,
        discouraged_hits=discouraged_hits,
        action_count=action_count,
        minimum_actions_met=minimum_actions_met,
    )


def sample_rubric_to_dict(sample: IncidentSample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "description": sample.description,
        "expected_incident_type": sample.expected_incident_type,
        "expectation_note": sample.expectation_note,
        "logs": sample.logs,
        "root_cause_rubric": {
            "required_concepts": [_criterion_to_dict(item) for item in sample.root_cause_rubric.required_concepts],
            "forbidden_terms": list(sample.root_cause_rubric.forbidden_terms),
            "require_uncertainty": sample.root_cause_rubric.require_uncertainty,
        },
        "action_rubric": {
            "required_concepts": [_criterion_to_dict(item) for item in sample.action_rubric.required_concepts],
            "minimum_actions": sample.action_rubric.minimum_actions,
            "discouraged_phrases": list(sample.action_rubric.discouraged_phrases),
        },
    }


def _criterion_is_matched(text: str, criterion_definition: ConceptCriterion) -> bool:
    return any(
        normalized_keyword and normalized_keyword in text
        for normalized_keyword in (_normalize_for_match(keyword) for keyword in criterion_definition.keywords)
    )


def _criterion_to_dict(criterion_definition: ConceptCriterion) -> dict:
    return {
        "label": criterion_definition.label,
        "keywords": list(criterion_definition.keywords),
    }


def _normalize_for_match(value: str) -> str:
    normalized = _NON_ALPHANUMERIC_PATTERN.sub(" ", str(value).strip().lower())
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()
"""Rubric-based scoring for Ariadne incident analysis evaluations.

Provides deterministic, keyword-matching scores for root cause quality and
action quality against the IncidentSample rubrics defined in sample_library.py.

Scoring approach
----------------
Root cause quality (0.0–1.0):
    85 % concept coverage   – fraction of required_concepts matched
    15 % uncertainty credit – full credit if rubric requires uncertainty and
                              the response mentions hedging language
    -25 % per forbidden term found in the response

Action quality (0.0–1.0):
    75 % concept coverage   – fraction of required_concepts matched
    25 % count score        – min(actual_count / minimum_actions, 1.0)
    -15 % per discouraged phrase found in the actions text
"""
from __future__ import annotations

from evals.sample_library import ActionRubric, RootCauseRubric

_UNCERTAINTY_KEYWORDS = ("may", "might", "could", "possibly", "potential", "likely", "unclear", "uncertain")


def _text_matches_criterion_keywords(text_lower: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text_lower for kw in keywords)


def score_root_cause(response: str, rubric: RootCauseRubric) -> float:
    """Return a 0.0–1.0 score for how well *response* satisfies *rubric*.

    Args:
        response: The model's root_cause string.
        rubric:   The RootCauseRubric for this sample.
    """
    text = response.lower()

    # Concept coverage (0.0–1.0)
    matched = sum(
        1
        for criterion in rubric.required_concepts
        if _text_matches_criterion_keywords(text, criterion.keywords)
    )
    total = len(rubric.required_concepts) or 1
    coverage = matched / total

    # Uncertainty satisfaction (0.0–1.0)
    uncertainty_score = 0.0
    if rubric.require_uncertainty:
        uncertainty_score = 1.0 if any(kw in text for kw in _UNCERTAINTY_KEYWORDS) else 0.0

    # Forbidden term penalty
    forbidden_penalty = sum(
        0.25 for term in rubric.forbidden_terms if term.lower() in text
    )

    raw = 0.85 * coverage + 0.15 * uncertainty_score - forbidden_penalty
    return max(0.0, min(1.0, raw))


def score_action(actions: list[str], rubric: ActionRubric) -> float:
    """Return a 0.0–1.0 score for how well *actions* satisfies *rubric*.

    Args:
        actions: The model's recommended_actions list.
        rubric:  The ActionRubric for this sample.
    """
    combined = " ".join(actions).lower()

    # Concept coverage (0.0–1.0)
    matched = sum(
        1
        for criterion in rubric.required_concepts
        if _text_matches_criterion_keywords(combined, criterion.keywords)
    )
    total = len(rubric.required_concepts) or 1
    coverage = matched / total

    # Count score (0.0–1.0)
    count_score = min(1.0, len(actions) / max(rubric.minimum_actions, 1))

    # Discouraged phrase penalty
    discouraged_penalty = sum(
        0.15 for phrase in rubric.discouraged_phrases if phrase.lower() in combined
    )

    raw = 0.75 * coverage + 0.25 * count_score - discouraged_penalty
    return max(0.0, min(1.0, raw))

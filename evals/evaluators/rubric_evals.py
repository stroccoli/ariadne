"""Rubric-based LangSmith evaluators for Ariadne incident analysis.

These evaluators are fully deterministic (no LLM calls) — they use
keyword matching via the rubrics defined in sample_library.py.

Evaluators:
    eval_root_cause_quality — concept coverage of the root cause response
    eval_action_quality     — concept coverage of recommended actions
    eval_final_score        — composite: 0.3*type_accuracy + 0.4*rc + 0.3*action
"""
from __future__ import annotations

from langsmith.schemas import Example, Run

from evals.rubric_scoring import score_action, score_root_cause
from evals.sample_library import get_sample_by_id


def _get_sample_from_example(example: Example):
    """Resolve the IncidentSample from example metadata, or return None."""
    sample_id = (example.metadata or {}).get("sample_id")
    if not sample_id:
        return None
    try:
        return get_sample_by_id(sample_id)
    except ValueError:
        return None


def eval_root_cause_quality(run: Run, example: Example) -> dict:
    """Score: how well does the root cause cover the rubric required concepts?"""
    sample = _get_sample_from_example(example)
    if sample is None:
        return {"key": "root_cause_quality", "score": None}
    response = (run.outputs or {}).get("response", "")
    return {"key": "root_cause_quality", "score": score_root_cause(response, sample.root_cause_rubric)}


def eval_action_quality(run: Run, example: Example) -> dict:
    """Score: how well do recommended actions cover the rubric required concepts?"""
    sample = _get_sample_from_example(example)
    if sample is None:
        return {"key": "action_quality", "score": None}
    actions = (run.outputs or {}).get("recommended_actions") or []
    return {"key": "action_quality", "score": score_action(actions, sample.action_rubric)}


def eval_final_score(run: Run, example: Example) -> dict:
    """Composite score: 0.3 * type_accuracy + 0.4 * root_cause_quality + 0.3 * action_quality."""
    outputs = run.outputs or {}
    meta = example.metadata or {}

    type_match = float(
        outputs.get("incident_type", "") == meta.get("expected_incident_type", "")
    )

    sample = _get_sample_from_example(example)
    if sample is None:
        return {"key": "final_score", "score": type_match}

    rc_score = score_root_cause(outputs.get("response", ""), sample.root_cause_rubric)
    act_score = score_action(outputs.get("recommended_actions") or [], sample.action_rubric)
    final = 0.3 * type_match + 0.4 * rc_score + 0.3 * act_score
    return {"key": "final_score", "score": round(final, 4)}

from __future__ import annotations

from dataclasses import dataclass

from ariadne.core.models import IncidentReportOutput


DEFAULT_DISCOURAGED_ACTION_PHRASES = (
    "check logs",
    "investigate further",
    "monitor the system",
    "look into the issue",
    "debug the issue",
    "follow up",
)


@dataclass(frozen=True)
class ConceptCriterion:
    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RootCauseRubric:
    required_concepts: tuple[ConceptCriterion, ...]
    forbidden_terms: tuple[str, ...] = ()
    require_uncertainty: bool = False


@dataclass(frozen=True)
class ActionRubric:
    required_concepts: tuple[ConceptCriterion, ...]
    minimum_actions: int = 1
    discouraged_phrases: tuple[str, ...] = DEFAULT_DISCOURAGED_ACTION_PHRASES


@dataclass(frozen=True)
class IncidentSample:
    sample_id: str
    description: str
    logs: str
    expected_incident_type: str
    expectation_note: str
    root_cause_rubric: RootCauseRubric
    action_rubric: ActionRubric


@dataclass(frozen=True)
class RootCauseQualityResult:
    score: float
    matched_concepts: tuple[str, ...]
    missed_concepts: tuple[str, ...]
    forbidden_hits: tuple[str, ...]
    uncertainty_satisfied: bool


@dataclass(frozen=True)
class ActionQualityResult:
    score: float
    matched_concepts: tuple[str, ...]
    missed_concepts: tuple[str, ...]
    discouraged_hits: tuple[str, ...]
    action_count: int
    minimum_actions_met: bool


@dataclass(frozen=True)
class ModeEvaluation:
    mode: str
    report: IncidentReportOutput
    incident_type_match: bool
    root_cause_quality: RootCauseQualityResult
    action_quality: ActionQualityResult
    duration_seconds: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class SampleEvaluation:
    sample: IncidentSample
    evaluations: tuple[ModeEvaluation, ...]


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    total_samples: int
    incident_type_matches: int
    accuracy: float
    average_confidence: float
    average_actions: float
    average_root_cause_quality: float
    average_action_quality: float
    total_duration_seconds: float
    average_duration_seconds: float
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


@dataclass(frozen=True)
class ABTestRun:
    comparisons: tuple[SampleEvaluation, ...]
    summaries: tuple[ModeSummary, ...]
    started_at: str
    finished_at: str
    total_duration_seconds: float
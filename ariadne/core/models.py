"""Typed output models for the incident analysis pipeline.

Design choice:
- Validate LLM outputs at module boundaries so the rest of the pipeline can work with normalized data.

Tradeoff:
- Strict schemas improve reliability, but require explicit fallback behavior when model output is malformed.

Production caveat:
- Schema validation does not guarantee factual correctness; it only constrains output shape and value ranges.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

from ariadne.core.utils.output import coerce_confidence

IncidentType = Literal[
    "timeout",
    "dependency_failure",
    "database_issue",
    "memory_issue",
    "unknown",
]

PromptMode = Literal["detailed", "compact"]

ALLOWED_INCIDENT_TYPES = frozenset(get_args(IncidentType))
ALLOWED_PROMPT_MODES = frozenset(get_args(PromptMode))


class ClassificationOutput(BaseModel):
    incident_type: IncidentType
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("incident_type", mode="before")
    @classmethod
    def normalize_incident_type(cls, value: object) -> str:
        return str(value or "unknown").strip().lower() or "unknown"

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> float:
        return coerce_confidence(value)


class AnalysisOutput(BaseModel):
    root_cause: str
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("root_cause", mode="before")
    @classmethod
    def normalize_root_cause(cls, value: object) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("root_cause must not be empty")
        return normalized

    @field_validator("recommended_actions", mode="before")
    @classmethod
    def normalize_actions(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []

        return [str(action).strip() for action in value if str(action).strip()]

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> float:
        return coerce_confidence(value)


class IncidentReportOutput(BaseModel):
    incident_type: IncidentType
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_actions: list[str] = Field(default_factory=list)

    @field_validator("root_cause", mode="before")
    @classmethod
    def normalize_root_cause(cls, value: object) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("root_cause must not be empty")
        return normalized

    @field_validator("recommended_actions", mode="before")
    @classmethod
    def normalize_actions(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []

        return [str(action).strip() for action in value if str(action).strip()]

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> float:
        return coerce_confidence(value)


def build_output(
    classification: ClassificationOutput,
    analysis: AnalysisOutput,
) -> IncidentReportOutput:
    """Assemble the final IncidentReportOutput from classification and analysis results."""
    overall_confidence = min(classification.confidence, analysis.confidence)
    return IncidentReportOutput(
        incident_type=classification.incident_type,
        root_cause=analysis.root_cause,
        confidence=round(overall_confidence, 2),
        recommended_actions=analysis.recommended_actions,
    )

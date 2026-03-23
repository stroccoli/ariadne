from __future__ import annotations

from pydantic import BaseModel, Field

from ariadne.core.models import IncidentType


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AnalysisMetadata(BaseModel):
    retrieval_attempts: int = 0
    llm_calls: int = 0
    node_timings: dict[str, float] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class AnalyzeResponse(BaseModel):
    incident_type: IncidentType
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_actions: list[str] = Field(default_factory=list)
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)


class ErrorResponse(BaseModel):
    detail: str

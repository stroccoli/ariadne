"""Centralized state for the multi-agent incident analysis pipeline.

Design choice:
- A single state object flows through all agents, giving the orchestrator full visibility into
  what each agent produced and what decisions were made.

Tradeoff:
- Centralised state couples agents to one schema, but makes the execution trace explicit and
  debuggable — a good trade for a system this size.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ariadne.core.models import AnalysisOutput, IncidentReportOutput, IncidentType


class IncidentState(BaseModel):
    # --- inputs (set once, read by agents) ---
    logs: str
    mode: str = "detailed"

    # --- written by ClassifierAgent ---
    incident_type: Optional[IncidentType] = None
    classification_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # --- written by RAGAgent ---
    context: list[str] = Field(default_factory=list)
    retrieved_doc_titles: list[str] = Field(default_factory=list)

    # --- written by AnalyzerAgent ---
    analysis: Optional[AnalysisOutput] = None

    # --- written by Orchestrator ---
    final_output: Optional[IncidentReportOutput] = None

    # --- bookkeeping ---
    retrieval_attempts: int = 0

    # --- observability metrics ---
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_calls: int = 0
    node_timings: dict[str, float] = Field(default_factory=dict)

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ariadne.api.models.request import AnalyzeRequest
from ariadne.api.models.response import (
    AnalysisMetadata,
    AnalyzeResponse,
    TokenUsage,
)
from ariadne.core.graph import run_graph

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the incident analysis pipeline and return structured results."""
    try:
        state = run_graph(logs=request.logs, mode=request.mode)
    except Exception:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Analysis pipeline failed")

    if state.final_output is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no output")

    return AnalyzeResponse(
        incident_type=state.final_output.incident_type,
        root_cause=state.final_output.root_cause,
        confidence=state.final_output.confidence,
        recommended_actions=state.final_output.recommended_actions,
        metadata=AnalysisMetadata(
            retrieval_attempts=state.retrieval_attempts,
            llm_calls=state.total_llm_calls,
            node_timings=state.node_timings,
            usage=TokenUsage(
                prompt_tokens=state.total_prompt_tokens,
                completion_tokens=state.total_completion_tokens,
                total_tokens=state.total_prompt_tokens + state.total_completion_tokens,
            ),
        ),
    )

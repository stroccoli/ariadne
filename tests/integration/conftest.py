from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ariadne.api.main import app
from ariadne.core.config import get_embedding_client, get_vector_store
from ariadne.core.models import AnalysisOutput, IncidentReportOutput
from ariadne.core.state import IncidentState


def _make_incident_state(
    *,
    logs: str = "ERROR: connection refused on db-primary",
    mode: str = "detailed",
    incident_type: str = "database_issue",
    classification_confidence: float = 0.92,
    root_cause: str = "Primary database connection pool exhausted",
    analysis_confidence: float = 0.88,
    recommended_actions: list[str] | None = None,
    retrieval_attempts: int = 1,
    total_llm_calls: int = 3,
    total_prompt_tokens: int = 1200,
    total_completion_tokens: int = 350,
    node_timings: dict[str, float] | None = None,
    final_output: IncidentReportOutput | None = ...,
) -> IncidentState:
    """Build a realistic IncidentState for test scenarios."""
    if recommended_actions is None:
        recommended_actions = ["Restart the primary database", "Scale connection pool"]
    if node_timings is None:
        node_timings = {"classify": 0.15, "retrieve_1": 0.42, "analyze": 1.03, "build_output": 0.002, "total": 1.61}

    analysis = AnalysisOutput(
        root_cause=root_cause,
        recommended_actions=recommended_actions,
        confidence=analysis_confidence,
    )

    overall_confidence = round(min(classification_confidence, analysis_confidence), 2)

    if final_output is ...:
        final_output = IncidentReportOutput(
            incident_type=incident_type,
            root_cause=root_cause,
            confidence=overall_confidence,
            recommended_actions=recommended_actions,
        )

    return IncidentState(
        logs=logs,
        mode=mode,
        incident_type=incident_type,
        classification_confidence=classification_confidence,
        context=["Known issue: DB connection pool under high load"],
        analysis=analysis,
        final_output=final_output,
        retrieval_attempts=retrieval_attempts,
        total_llm_calls=total_llm_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        node_timings=node_timings,
    )


@pytest.fixture()
def client():
    """Provide a FastAPI TestClient with lru_cache cleanup."""
    with TestClient(app) as c:
        yield c
    get_vector_store.cache_clear()
    get_embedding_client.cache_clear()


@pytest.fixture()
def mock_run_graph():
    """Patch run_graph at the route level, returning a valid IncidentState by default."""
    with patch("ariadne.api.routes.analyze.run_graph") as mock:
        mock.return_value = _make_incident_state()
        yield mock

"""Pipeline target function for LangSmith evaluation.

``run_pipeline`` is passed as the target to ``langsmith.evaluate()``.  It
invokes the Ariadne LangGraph and returns the RAGAS-compatible output dict
that all downstream evaluators consume.
"""
from __future__ import annotations

from ariadne.core.graph import run_graph


def run_pipeline(inputs: dict) -> dict:
    """Invoke the Ariadne graph and return RAGAS-compatible output fields.

    Each call is automatically traced in LangSmith when tracing is enabled.
    """
    logs: str = inputs["logs"]
    mode: str = inputs.get("mode", "detailed")

    state = run_graph(logs, mode)

    final_output = state.final_output
    root_cause: str = final_output.root_cause if final_output else ""
    recommended_actions: list[str] = list(final_output.recommended_actions) if final_output else []
    incident_type: str = (
        final_output.incident_type
        if final_output and final_output.incident_type
        else "unknown"
    )
    confidence: float | None = final_output.confidence if final_output else None

    return {
        "user_input": logs,
        "retrieved_contexts": list(state.context or []),
        "response": root_cause,
        "recommended_actions": recommended_actions,
        "incident_type": incident_type,
        "confidence": confidence,
        "prompt_tokens": state.total_prompt_tokens,
        "completion_tokens": state.total_completion_tokens,
    }

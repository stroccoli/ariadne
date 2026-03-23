from __future__ import annotations

import logging
import uuid
from time import perf_counter

from langgraph.graph import END, START, StateGraph

from ariadne.core.agents.analyzer import run_analyzer
from ariadne.core.agents.classifier import run_classifier
from ariadne.core.agents.rag import run_retrieval
from ariadne.core.config import get_langsmith_project, is_langsmith_enabled
from ariadne.core.models import ClassificationOutput, build_output
from ariadne.core.state import IncidentState

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIEVAL_ATTEMPTS = 2


def classify_node(state: IncidentState) -> dict:
    t0 = perf_counter()
    updated = run_classifier(state)
    elapsed = round(perf_counter() - t0, 3)
    logger.info(
        "[classify] incident_type=%s confidence=%s duration=%.3fs",
        updated.incident_type, updated.classification_confidence, elapsed,
    )
    return {
        "incident_type": updated.incident_type,
        "classification_confidence": updated.classification_confidence,
        "total_prompt_tokens": updated.total_prompt_tokens,
        "total_completion_tokens": updated.total_completion_tokens,
        "total_llm_calls": updated.total_llm_calls,
        "node_timings": {**state.node_timings, "classify": elapsed},
    }


def retrieve_node(state: IncidentState) -> dict:
    t0 = perf_counter()
    updated = run_retrieval(state)
    elapsed = round(perf_counter() - t0, 3)
    logger.info(
        "[retrieve] attempt=%d docs=%d duration=%.3fs",
        updated.retrieval_attempts, len(updated.context), elapsed,
    )
    timing_key = f"retrieve_{updated.retrieval_attempts}"
    return {
        "context": updated.context,
        "retrieval_attempts": updated.retrieval_attempts,
        "node_timings": {**state.node_timings, timing_key: elapsed},
    }


def analyze_node(state: IncidentState) -> dict:
    t0 = perf_counter()
    updated = run_analyzer(state)
    elapsed = round(perf_counter() - t0, 3)
    confidence = updated.analysis.confidence if updated.analysis else None
    logger.info("[analyze] confidence=%s duration=%.3fs", confidence, elapsed)
    return {
        "analysis": updated.analysis,
        "total_prompt_tokens": updated.total_prompt_tokens,
        "total_completion_tokens": updated.total_completion_tokens,
        "total_llm_calls": updated.total_llm_calls,
        "node_timings": {**state.node_timings, "analyze": elapsed},
    }


def build_output_node(state: IncidentState) -> dict:
    t0 = perf_counter()
    if state.analysis is None or state.incident_type is None:
        logger.warning("[build_output] insufficient state — skipping")
        return {}
    classification = ClassificationOutput(
        incident_type=state.incident_type,
        confidence=state.classification_confidence or 0.0,
    )
    final = build_output(classification, state.analysis)
    elapsed = round(perf_counter() - t0, 3)
    logger.info("[build_output] confidence=%s duration=%.3fs", final.confidence, elapsed)
    return {
        "final_output": final,
        "node_timings": {**state.node_timings, "build_output": elapsed},
    }


def should_retry(state: IncidentState) -> str:
    confidence = state.analysis.confidence if state.analysis else 0.0
    if confidence < CONFIDENCE_THRESHOLD and state.retrieval_attempts < MAX_RETRIEVAL_ATTEMPTS:
        logger.info("[route] retry — confidence=%.2f attempts=%d", confidence, state.retrieval_attempts)
        return "retry"
    logger.info("[route] done — confidence=%.2f attempts=%d", confidence, state.retrieval_attempts)
    return "done"


def build_graph() -> StateGraph:
    graph = StateGraph(IncidentState)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("build_output", build_output_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_conditional_edges("analyze", should_retry, {"retry": "retrieve", "done": "build_output"})
    graph.add_edge("build_output", END)

    return graph.compile()


def get_graph_diagram() -> str:
    return build_graph().get_graph().draw_mermaid()


def run_graph(logs: str, mode: str = "detailed") -> IncidentState:
    run_id = uuid.uuid4().hex[:8]
    logger.info("[graph] start run_id=%s", run_id)
    t0 = perf_counter()

    state = IncidentState(logs=logs, mode=mode)

    langsmith_config: dict = {}
    if is_langsmith_enabled():
        langsmith_config = {
            "run_name": "incident_analysis",
            "metadata": {
                "run_id": run_id,
                "mode": mode,
                "project": get_langsmith_project(),
            },
            "tags": ["ariadne", mode],
        }

    final_state = build_graph().invoke(state, config=langsmith_config if langsmith_config else None)

    elapsed = round(perf_counter() - t0, 3)
    logger.info("[graph] done run_id=%s total_duration=%.3fs", run_id, elapsed)

    if isinstance(final_state, dict):
        result = IncidentState(**final_state)
    else:
        result = final_state

    # Store total latency in node_timings for unified summary access
    result.node_timings["total"] = elapsed
    return result

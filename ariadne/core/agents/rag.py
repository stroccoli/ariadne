"""Retrieval agent backed by a configurable vector store.

Design choice:
- Keep retrieval separate from the analyzer so indexing, embeddings, and storage can evolve independently.

Tradeoff:
- Returning a formatted text block is easy to plug into prompts, but it discards document structure and metadata.

Production caveat:
- Mature RAG systems usually add metadata filters, chunking, citations, freshness control, and query rewriting.
"""

from __future__ import annotations

import logging

from ariadne.core.config import get_vector_store
from ariadne.core.retrieval.vector_stores import VectorStoreUnavailableError
from ariadne.core.state import IncidentState


logger = logging.getLogger(__name__)


def retrieve_context(logs: str) -> str:
    logger.info("Running retrieval for incident logs")

    try:
        documents = get_vector_store().search(logs)
    except Exception as error:
        logger.error("Retrieval failed: %s", error)
        return "No additional context available beyond the current logs."

    if not documents:
        logger.info("Retrieval returned no supporting documents")
        return "No additional context available beyond the current logs."

    logger.debug("Retrieved %d documents for analyzer context", len(documents))
    return "\n\n".join(
        f"[Context {index}] {document}"
        for index, document in enumerate(documents, start=1)
    )


def _build_search_query(state: IncidentState) -> str:
    """Enrich the search query on retries with classification and analysis hints."""
    query = state.logs
    if state.retrieval_attempts > 0 and (state.incident_type or state.analysis):
        parts = [query]
        if state.incident_type:
            parts.append(f"Incident type: {state.incident_type}")
        if state.analysis:
            parts.append(f"Root cause hypothesis: {state.analysis.root_cause}")
        query = "\n".join(parts)
    return query


def run_retrieval(state: IncidentState) -> IncidentState:
    """Run retrieval on the given state and return the updated state."""
    query = _build_search_query(state)
    attempt = state.retrieval_attempts + 1
    logger.info("[retrieve] running retrieval attempt=%d", attempt)
    if state.retrieval_attempts > 0:
        logger.debug("[retrieve] enriched query with incident_type=%s", state.incident_type)

    try:
        documents = get_vector_store().search(query)
    except VectorStoreUnavailableError as error:
        logger.warning(
            "[retrieve] Qdrant unreachable, skipping RAG for this request "
            "(pipeline continues without retrieval context): %s",
            error,
        )
        documents = []
    except Exception as error:
        logger.error("[retrieve] unexpected retrieval error: %s", error, exc_info=True)
        documents = []

    logger.debug("[retrieve] returned %d document(s)", len(documents))

    state.context = documents
    state.retrieved_doc_titles = [doc.split("\n")[0] for doc in documents]
    state.retrieval_attempts += 1
    return state

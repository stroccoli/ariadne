from __future__ import annotations

import logging

from ariadne.core.retrieval.vector_stores.base import VectorStore


logger = logging.getLogger(__name__)


class NoOpVectorStore(VectorStore):
    def index(self, documents: list[str]) -> None:
        logger.warning("Skipping indexing because VECTOR_STORE=none")

    def search(self, query: str) -> list[str]:
        logger.warning("Skipping retrieval because VECTOR_STORE=none")
        return []

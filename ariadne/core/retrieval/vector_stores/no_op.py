from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ariadne.core.retrieval.vector_stores.base import VectorStore

if TYPE_CHECKING:
    from ariadne.core.retrieval.document import IngestionDocument


logger = logging.getLogger(__name__)


class NoOpVectorStore(VectorStore):
    def index(self, documents: list[str]) -> None:
        logger.warning("Skipping indexing because VECTOR_STORE=none")

    def index_documents(self, docs: list[IngestionDocument], *, embedding_batch_size: int = 32) -> None:
        logger.warning("Skipping indexing because VECTOR_STORE=none")

    def search(self, query: str) -> list[str]:
        logger.warning("Skipping retrieval because VECTOR_STORE=none")
        return []

    def search_filtered(
        self,
        query: str,
        *,
        source: str | None = None,
        service: str | None = None,
        severity: str | None = None,
    ) -> list[str]:
        logger.warning("Skipping filtered retrieval because VECTOR_STORE=none")
        return []

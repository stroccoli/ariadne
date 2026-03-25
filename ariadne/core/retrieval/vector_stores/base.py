from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ariadne.core.retrieval.document import IngestionDocument


class VectorStoreUnavailableError(Exception):
    """Raised when the vector store backend is not reachable (network, timeout, etc.)."""


class VectorStore(ABC):
    @abstractmethod
    def index(self, documents: list[str]) -> None:
        """Index plain text documents."""

    @abstractmethod
    def index_documents(self, docs: list[IngestionDocument], *, embedding_batch_size: int = 32) -> None:
        """Index structured IngestionDocument objects with full metadata."""

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """Return the most relevant documents for a query string."""

    @abstractmethod
    def search_filtered(
        self,
        query: str,
        *,
        source: str | None = None,
        service: str | None = None,
        severity: str | None = None,
    ) -> list[str]:
        """Search with optional metadata filters."""

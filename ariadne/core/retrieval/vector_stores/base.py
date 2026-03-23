from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStoreUnavailableError(Exception):
    """Raised when the vector store backend is not reachable (network, timeout, etc.)."""


class VectorStore(ABC):
    @abstractmethod
    def index(self, documents: list[str]) -> None:
        """Index documents so they are available for later similarity search."""

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """Return the most relevant documents for a query string."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Return a single embedding vector for one input string."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

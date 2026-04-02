from __future__ import annotations

import logging

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False

from ariadne.core.integrations.embeddings.base import EmbeddingClient


logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package is required for EMBEDDING_PROVIDER=openai. "
                "Install with: pip install 'ariadne[openai]'"
            )
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.debug("Generating %d embeddings with OpenAI model '%s'", len(texts), self.model)
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]

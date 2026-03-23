from __future__ import annotations

import logging

from openai import OpenAI

from ariadne.core.integrations.embeddings.base import EmbeddingClient


logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.debug("Generating %d embeddings with OpenAI model '%s'", len(texts), self.model)
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]

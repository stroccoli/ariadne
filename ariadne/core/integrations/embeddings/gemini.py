from __future__ import annotations

import logging

from google import genai
from google.genai import types

from ariadne.core.integrations.embeddings.base import EmbeddingClient


logger = logging.getLogger(__name__)


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "models/text-embedding-004") -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.debug(
            "Generating %d embeddings with Gemini model '%s'", len(texts), self.model
        )
        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        # response.embeddings is a list of ContentEmbedding; each has a .values field.
        return [list(embedding.values) for embedding in response.embeddings]

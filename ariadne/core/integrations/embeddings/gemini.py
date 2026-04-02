from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types

from ariadne.core.integrations.embeddings.base import EmbeddingClient


logger = logging.getLogger(__name__)

# Gemini free tier: 100 embed requests/min. Use 70/min to stay safely under.
_GEMINI_MIN_REQUEST_INTERVAL = 60.0 / 70  # ~0.857s between requests


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001", dimensions: int = 768) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = genai.Client(api_key=api_key)
        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = _GEMINI_MIN_REQUEST_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.debug(
            "Generating %d embeddings with Gemini model '%s' (dim=%d)", len(texts), self.model, self.dimensions
        )
        self._throttle()
        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimensions,
            ),
        )
        # response.embeddings is a list of ContentEmbedding; each has a .values field.
        return [list(embedding.values) for embedding in response.embeddings]


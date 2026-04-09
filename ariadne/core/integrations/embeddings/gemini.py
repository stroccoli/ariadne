from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types

from ariadne.core.integrations.embeddings.base import EmbeddingClient


logger = logging.getLogger(__name__)

# Gemini free tier: 100 embed requests/min, where each *text* counts as one
# request (not each API call).  Use 80 texts/min to stay safely under the cap.
_GEMINI_RPM_LIMIT = 80  # texts per minute budget


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001", dimensions: int = 768) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = genai.Client(api_key=api_key)
        self._last_request_time: float = 0.0

    def _throttle(self, n_texts: int) -> None:
        """Sleep long enough so that n_texts inputs don't exceed _GEMINI_RPM_LIMIT."""
        required_interval = 60.0 * n_texts / _GEMINI_RPM_LIMIT
        elapsed = time.monotonic() - self._last_request_time
        wait = required_interval - elapsed
        if wait > 0:
            logger.debug("Gemini throttle: sleeping %.1fs for %d texts", wait, n_texts)
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.debug(
            "Generating %d embeddings with Gemini model '%s' (dim=%d)", len(texts), self.model, self.dimensions
        )
        self._throttle(len(texts))
        max_retries = 4
        for attempt in range(max_retries):
            try:
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
            except Exception as exc:
                is_last = attempt == max_retries - 1
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    backoff = 60.0 * (2 ** attempt)
                    if is_last:
                        raise
                    logger.warning(
                        "Gemini 429 rate limit on attempt %d/%d. Backing off %.0fs.",
                        attempt + 1, max_retries, backoff,
                    )
                    time.sleep(backoff)
                    self._last_request_time = time.monotonic()
                else:
                    raise


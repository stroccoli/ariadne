from __future__ import annotations

import logging

import requests

from ariadne.core.integrations.embeddings.base import EmbeddingClient


logger = logging.getLogger(__name__)


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def embed_text(self, text: str) -> list[float]:
        logger.debug("Generating Ollama embedding with model '%s'", self.model)
        response = self.session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Ollama embedding response did not include an embedding vector")
        return [float(value) for value in embedding]

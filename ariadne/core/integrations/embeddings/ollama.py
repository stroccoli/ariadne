from __future__ import annotations

import logging

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

from ariadne.core.integrations.embeddings.base import EmbeddingClient


logger = logging.getLogger(__name__)

_EMBED_BATCH_ENDPOINT = "/api/embed"
_EMBED_SINGLE_ENDPOINT = "/api/embeddings"
DEFAULT_BATCH_SIZE = 32


class OllamaEmbeddingClient(EmbeddingClient):

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        batch_size: int = DEFAULT_BATCH_SIZE,
        keep_alive: str | None = None,
        dimensions: int = 768,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.dimensions = dimensions
        # keep_alive controls how long Ollama holds the embedding model in memory.
        # Defaults to Ollama server setting (typically 5m) when None.
        self.keep_alive = keep_alive
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError(
                "requests package is required for EMBEDDING_PROVIDER=ollama. "
                "Install with: pip install 'ariadne[ollama]'"
            )
        self.session = requests.Session()

    def embed_text(self, text: str) -> list[float]:
        logger.debug("Generating Ollama embedding with model '%s'", self.model)
        payload: dict = {"model": self.model, "prompt": text}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        response = self.session.post(
            f"{self.base_url}{_EMBED_SINGLE_ENDPOINT}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Ollama embedding response did not include an embedding vector")
        return [float(value) for value in embedding]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict = {"model": self.model, "input": texts}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        response = self.session.post(
            f"{self.base_url}{_EMBED_BATCH_ENDPOINT}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama batch embed returned {len(embeddings) if isinstance(embeddings, list) else 'no'} "
                f"embeddings for {len(texts)} texts"
            )
        return [[float(v) for v in emb] for emb in embeddings]

    def embed_texts_batched(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        effective_batch_size = batch_size or self.batch_size
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size

        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i : i + effective_batch_size]
            batch_num = i // effective_batch_size + 1
            logger.debug(
                "Embedding batch %d/%d (%d texts)",
                batch_num,
                total_batches,
                len(batch),
            )
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)

        logger.info(
            "Embedded %d texts in %d batches (model=%s)",
            len(texts),
            total_batches,
            self.model,
        )
        return all_embeddings

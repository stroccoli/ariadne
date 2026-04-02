"""Embedding cache wrappers: Redis-backed and in-process LRU."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from ariadne.core.integrations.embeddings.base import EmbeddingClient

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)


class CachedEmbeddingClient(EmbeddingClient):
    """Transparent cache layer that wraps another :class:`EmbeddingClient`.

    Cache key format: ``embed:{model_name}:{sha256(text)}``

    On Redis failure the wrapper falls through to the inner client so the
    pipeline is never blocked by cache unavailability.
    """

    def __init__(
        self,
        inner: EmbeddingClient,
        redis_client: redis.Redis,
        model_name: str,
        ttl_seconds: int = 86_400,
    ) -> None:
        self.inner = inner
        self.redis = redis_client
        self.model_name = model_name
        self.ttl_seconds = ttl_seconds

        # Counters — reset between batches to allow per-run stats.
        self.total_hits = 0
        self.total_misses = 0

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embed:{self.model_name}:{text_hash}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [self._cache_key(t) for t in texts]

        # --- Try bulk read from cache ---
        cached_values: list[bytes | None] = []
        try:
            cached_values = self.redis.mget(keys)
        except Exception as exc:
            logger.warning("Redis mget failed, falling back to inner client: %s", exc)
            cached_values = [None] * len(texts)

        # Separate hits from misses
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []

        for i, raw in enumerate(cached_values):
            if raw is not None:
                try:
                    results[i] = json.loads(raw)
                    continue
                except (json.JSONDecodeError, TypeError):
                    pass  # treat as miss
            miss_indices.append(i)

        hits = len(texts) - len(miss_indices)
        self.total_hits += hits
        self.total_misses += len(miss_indices)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            computed = self.inner.embed_texts(miss_texts)

            # Store computed embeddings back into Redis
            pipe_items: dict[str, str] = {}
            for idx, vec in zip(miss_indices, computed):
                results[idx] = vec
                pipe_items[keys[idx]] = json.dumps(vec)

            if pipe_items:
                try:
                    pipe = self.redis.pipeline(transaction=False)
                    for k, v in pipe_items.items():
                        pipe.setex(k, self.ttl_seconds, v)
                    pipe.execute()
                except Exception as exc:
                    logger.warning("Redis pipeline set failed: %s", exc)

        logger.debug(
            "Embedding cache: %d hits, %d misses (batch of %d)",
            hits,
            len(miss_indices),
            len(texts),
        )

        return results  # type: ignore[return-value]

    def embed_texts_batched(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Batch with cache-awareness — delegates to :meth:`embed_texts`."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self.embed_texts(batch))
        return all_embeddings

    def reset_stats(self) -> None:
        self.total_hits = 0
        self.total_misses = 0


class InMemoryEmbeddingCache(EmbeddingClient):
    """In-process LRU cache for embeddings — no external dependencies.

    Survives across requests within the same process lifetime. For
    single-machine deployments this provides embedding deduplication
    without requiring Redis.

    Cache key format: ``embed:{model_name}:{sha256(text)}``
    """

    def __init__(
        self,
        inner: EmbeddingClient,
        model_name: str,
        max_size: int = 2048,
    ) -> None:
        self.inner = inner
        self.model_name = model_name
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.total_hits = 0
        self.total_misses = 0

    def _cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embed:{self.model_name}:{text_hash}"

    def _get(self, key: str) -> list[float] | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _put(self, key: str, value: list[float]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # evict least-recently-used
            self._cache[key] = value

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self._get(self._cache_key(text))
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)

        hits = len(texts) - len(miss_indices)
        self.total_hits += hits
        self.total_misses += len(miss_indices)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            computed = self.inner.embed_texts(miss_texts)
            for idx, vec in zip(miss_indices, computed):
                results[idx] = vec
                self._put(self._cache_key(texts[idx]), vec)

        logger.debug(
            "Embedding cache: %d hits, %d misses (batch of %d)",
            hits,
            len(miss_indices),
            len(texts),
        )

        return results  # type: ignore[return-value]

    def embed_texts_batched(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self.embed_texts(batch))
        return all_embeddings

    def reset_stats(self) -> None:
        self.total_hits = 0
        self.total_misses = 0

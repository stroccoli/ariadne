"""Tests for CachedEmbeddingClient."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from ariadne.core.integrations.embeddings.cached import CachedEmbeddingClient


class _FakeInnerClient:
    """Minimal EmbeddingClient that tracks calls."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors = vectors or {}
        self.call_count = 0

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._vectors.get(t, [0.0, 0.0]) for t in texts]

    def embed_texts_batched(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self.embed_texts(texts)


def _make_redis_mock() -> MagicMock:
    """Return a mock Redis client that behaves like an empty cache."""
    mock = MagicMock()
    mock.mget.return_value = [None] * 100  # default: all misses
    mock.pipeline.return_value = MagicMock()
    mock.pipeline.return_value.__enter__ = MagicMock(return_value=mock.pipeline.return_value)
    mock.pipeline.return_value.__exit__ = MagicMock(return_value=False)
    return mock


class CachedEmbeddingClientTests(unittest.TestCase):
    def test_cache_miss_delegates_to_inner(self) -> None:
        inner = _FakeInnerClient({"hello": [1.0, 2.0]})
        redis_mock = _make_redis_mock()
        redis_mock.mget.return_value = [None]

        client = CachedEmbeddingClient(inner, redis_mock, "test-model", ttl_seconds=3600)
        result = client.embed_texts(["hello"])

        self.assertEqual(result, [[1.0, 2.0]])
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(client.total_misses, 1)
        self.assertEqual(client.total_hits, 0)

    def test_cache_hit_skips_inner(self) -> None:
        inner = _FakeInnerClient()
        redis_mock = _make_redis_mock()
        cached_vector = [3.0, 4.0]
        redis_mock.mget.return_value = [json.dumps(cached_vector).encode()]

        client = CachedEmbeddingClient(inner, redis_mock, "test-model", ttl_seconds=3600)
        result = client.embed_texts(["cached-text"])

        self.assertEqual(result, [[3.0, 4.0]])
        self.assertEqual(inner.call_count, 0)
        self.assertEqual(client.total_hits, 1)
        self.assertEqual(client.total_misses, 0)

    def test_mixed_hits_and_misses(self) -> None:
        inner = _FakeInnerClient({"miss-text": [5.0, 6.0]})
        redis_mock = _make_redis_mock()
        cached_vector = [1.0, 2.0]
        # First text cached, second is a miss
        redis_mock.mget.return_value = [
            json.dumps(cached_vector).encode(),
            None,
        ]

        client = CachedEmbeddingClient(inner, redis_mock, "test-model", ttl_seconds=3600)
        result = client.embed_texts(["hit-text", "miss-text"])

        self.assertEqual(result, [[1.0, 2.0], [5.0, 6.0]])
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(client.total_hits, 1)
        self.assertEqual(client.total_misses, 1)

    def test_redis_failure_falls_through(self) -> None:
        """When Redis raises, the wrapper falls through to the inner client."""
        inner = _FakeInnerClient({"fallback": [7.0, 8.0]})
        redis_mock = _make_redis_mock()
        redis_mock.mget.side_effect = ConnectionError("Redis down")

        client = CachedEmbeddingClient(inner, redis_mock, "test-model", ttl_seconds=3600)
        result = client.embed_texts(["fallback"])

        self.assertEqual(result, [[7.0, 8.0]])
        self.assertEqual(inner.call_count, 1)

    def test_redis_set_failure_does_not_crash(self) -> None:
        """Pipeline set failure should be non-fatal."""
        inner = _FakeInnerClient({"text": [1.0]})
        redis_mock = _make_redis_mock()
        redis_mock.mget.return_value = [None]
        pipe_mock = redis_mock.pipeline.return_value
        pipe_mock.execute.side_effect = ConnectionError("Redis write failed")

        client = CachedEmbeddingClient(inner, redis_mock, "test-model", ttl_seconds=3600)
        result = client.embed_texts(["text"])

        # Should still return the computed result
        self.assertEqual(result, [[1.0]])

    def test_empty_texts(self) -> None:
        inner = _FakeInnerClient()
        redis_mock = _make_redis_mock()

        client = CachedEmbeddingClient(inner, redis_mock, "test-model")
        result = client.embed_texts([])

        self.assertEqual(result, [])
        self.assertEqual(inner.call_count, 0)

    def test_cache_key_includes_model(self) -> None:
        inner = _FakeInnerClient()
        redis_mock = _make_redis_mock()

        client_a = CachedEmbeddingClient(inner, redis_mock, "model-a")
        client_b = CachedEmbeddingClient(inner, redis_mock, "model-b")

        key_a = client_a._cache_key("same text")
        key_b = client_b._cache_key("same text")

        self.assertNotEqual(key_a, key_b)
        self.assertIn("model-a", key_a)
        self.assertIn("model-b", key_b)

    def test_reset_stats(self) -> None:
        inner = _FakeInnerClient({"x": [1.0]})
        redis_mock = _make_redis_mock()
        redis_mock.mget.return_value = [None]

        client = CachedEmbeddingClient(inner, redis_mock, "test-model")
        client.embed_texts(["x"])
        self.assertEqual(client.total_misses, 1)

        client.reset_stats()
        self.assertEqual(client.total_hits, 0)
        self.assertEqual(client.total_misses, 0)

    def test_embed_texts_batched(self) -> None:
        inner = _FakeInnerClient({"a": [1.0], "b": [2.0], "c": [3.0]})
        redis_mock = _make_redis_mock()
        # Return the right number of Nones for each mget call
        redis_mock.mget.side_effect = lambda keys: [None] * len(keys)

        client = CachedEmbeddingClient(inner, redis_mock, "test-model")
        result = client.embed_texts_batched(["a", "b", "c"], batch_size=2)

        self.assertEqual(len(result), 3)
        # inner.embed_texts called at least twice (batch_size=2, 3 texts)
        self.assertGreaterEqual(inner.call_count, 2)


if __name__ == "__main__":
    unittest.main()

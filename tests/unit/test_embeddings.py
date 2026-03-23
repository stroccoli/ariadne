from __future__ import annotations

import unittest

from ariadne.core.integrations.embeddings import LocalHashEmbeddingClient


class LocalEmbeddingTests(unittest.TestCase):
    def test_hash_embeddings_are_deterministic(self) -> None:
        client = LocalHashEmbeddingClient(dimensions=16)

        first = client.embed_text("database timeout retry")
        second = client.embed_text("database timeout retry")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_hash_embeddings_normalize_non_empty_text(self) -> None:
        client = LocalHashEmbeddingClient(dimensions=32)

        vector = client.embed_text("postgres pool exhausted after retries")

        magnitude = sum(value * value for value in vector) ** 0.5
        self.assertAlmostEqual(magnitude, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
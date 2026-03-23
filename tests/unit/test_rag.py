from __future__ import annotations

import unittest
from unittest.mock import patch

from ariadne.core.agents.rag import retrieve_context
from ariadne.core.retrieval.vector_stores.qdrant_store import _keyword_overlap_score


class RagRetrievalTests(unittest.TestCase):
    def test_retrieve_context_formats_multiple_documents(self) -> None:
        fake_store = type("FakeStore", (), {"search": lambda self, query: ["doc one", "doc two"]})()

        with patch("ariadne.core.agents.rag.get_vector_store", return_value=fake_store):
            result = retrieve_context("timeout logs")

        self.assertIn("[Context 1] doc one", result)
        self.assertIn("[Context 2] doc two", result)

    def test_retrieve_context_returns_fallback_without_matches(self) -> None:
        fake_store = type("FakeStore", (), {"search": lambda self, query: []})()

        with patch("ariadne.core.agents.rag.get_vector_store", return_value=fake_store):
            result = retrieve_context("unseen logs")

        self.assertEqual(result, "No additional context available beyond the current logs.")

    def test_keyword_overlap_prefers_exact_incident_terms(self) -> None:
        specific_score = _keyword_overlap_score(
            "postgres connection pool exhausted after retries",
            "Database pool exhaustion pattern: connection pool exhausted often follows retry storms in postgres",
        )
        generic_score = _keyword_overlap_score(
            "postgres connection pool exhausted after retries",
            "Dependency outage with connection refusal can indicate a generic downstream issue",
        )

        self.assertGreater(specific_score, generic_score)


if __name__ == "__main__":
    unittest.main()

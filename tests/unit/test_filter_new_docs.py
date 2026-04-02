"""Tests for content-aware filter_new_docs() in QdrantVectorStore."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ariadne.core.retrieval.document import IngestionDocument, compute_content_hash
from ariadne.core.retrieval.vector_stores.qdrant_store import QdrantVectorStore


def _make_doc(doc_id: str, content: str) -> IngestionDocument:
    return IngestionDocument(
        id=doc_id,
        content=content,
        content_hash=compute_content_hash(content),
    )


class FilterNewDocsTests(unittest.TestCase):
    """Verify that filter_new_docs uses content_hash comparison."""

    def _build_store(self) -> QdrantVectorStore:
        mock_embedding = MagicMock()
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.embedding_client = mock_embedding
        store.collection_name = "test_collection"
        store.client = MagicMock()
        store.client.collection_exists.return_value = True
        return store

    def test_new_doc_is_included(self) -> None:
        """A doc whose point ID doesn't exist in Qdrant should be returned."""
        store = self._build_store()
        doc = _make_doc("new-1", "brand new content")

        # retrieve returns empty — nothing exists
        store.client.retrieve.return_value = []

        result = store.filter_new_docs([doc])
        self.assertEqual(result, [doc])

    def test_unchanged_doc_is_skipped(self) -> None:
        """A doc with matching content_hash should be skipped."""
        store = self._build_store()
        doc = _make_doc("existing-1", "same content")
        point_id = store._doc_point_id(doc)

        store.client.retrieve.return_value = [
            SimpleNamespace(id=point_id, payload={"content_hash": doc.content_hash}),
        ]

        result = store.filter_new_docs([doc])
        self.assertEqual(result, [])

    def test_changed_content_is_included(self) -> None:
        """A doc whose content changed (different hash) should be re-indexed."""
        store = self._build_store()
        doc = _make_doc("existing-2", "updated content v2")
        point_id = store._doc_point_id(doc)

        # Stored hash differs from doc.content_hash
        store.client.retrieve.return_value = [
            SimpleNamespace(id=point_id, payload={"content_hash": "old_hash_value"}),
        ]

        result = store.filter_new_docs([doc])
        self.assertEqual(result, [doc])

    def test_missing_hash_triggers_reindex(self) -> None:
        """Points without content_hash (pre-upgrade) should be re-indexed."""
        store = self._build_store()
        doc = _make_doc("legacy-1", "legacy doc no hash")
        point_id = store._doc_point_id(doc)

        # Existing point but payload has no content_hash
        store.client.retrieve.return_value = [
            SimpleNamespace(id=point_id, payload={}),
        ]

        result = store.filter_new_docs([doc])
        self.assertEqual(result, [doc])

    def test_empty_docs_returns_empty(self) -> None:
        store = self._build_store()
        result = store.filter_new_docs([])
        self.assertEqual(result, [])

    def test_collection_not_exists_returns_all(self) -> None:
        store = self._build_store()
        store.client.collection_exists.return_value = False

        docs = [_make_doc("a", "aaa"), _make_doc("b", "bbb")]
        result = store.filter_new_docs(docs)
        self.assertEqual(len(result), 2)

    def test_mixed_new_and_unchanged(self) -> None:
        """Batch with some new and some unchanged docs."""
        store = self._build_store()
        new_doc = _make_doc("new-x", "new content")
        old_doc = _make_doc("old-x", "old content")
        old_point_id = store._doc_point_id(old_doc)

        store.client.retrieve.return_value = [
            SimpleNamespace(id=old_point_id, payload={"content_hash": old_doc.content_hash}),
        ]

        result = store.filter_new_docs([new_doc, old_doc])
        self.assertEqual(result, [new_doc])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import logging
import re
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from ariadne.core.integrations.embeddings.base import EmbeddingClient
from ariadne.core.retrieval.document import IngestionDocument
from ariadne.core.retrieval.text_utils import keyword_overlap_score
from ariadne.core.retrieval.vector_stores.base import VectorStore, VectorStoreUnavailableError


logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 200

INDEXED_PAYLOAD_FIELDS = [
    ("source", PayloadSchemaType.KEYWORD),
    ("severity", PayloadSchemaType.KEYWORD),
    ("service", PayloadSchemaType.KEYWORD),
]


def _keyword_overlap_score(query: str, document: str) -> float:
    return keyword_overlap_score(query, document)


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "incident_knowledge",
        search_limit: int = 3,
        candidate_limit: int = 8,
        dense_weight: float = 0.65,
        keyword_weight: float = 0.35,
        timeout: int = 10,
        api_key: str | None = None,
        upsert_batch_size: int = UPSERT_BATCH_SIZE,
    ) -> None:
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self.search_limit = search_limit
        self.candidate_limit = max(candidate_limit, search_limit)
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        self.upsert_batch_size = upsert_batch_size
        # check_compatibility=False suppresses the client-side version check, which can
        # fail against Qdrant Cloud when the cluster version is ahead of the local client.
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout, check_compatibility=False)


    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            collection_info = self.client.get_collection(self.collection_name)
            configured_size = collection_info.config.params.vectors.size
            if configured_size != vector_size:
                raise ValueError(
                    "Qdrant collection vector size mismatch: "
                    f"expected {configured_size}, got {vector_size}. "
                    "Delete or recreate the collection before reindexing with a new embedding model."
                )
            return

        logger.info(
            "Creating Qdrant collection '%s' with vector size %d",
            self.collection_name,
            vector_size,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=200,
            ),
        )

        for field_name, field_schema in INDEXED_PAYLOAD_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            logger.debug("Created payload index on field '%s'", field_name)

    def index(self, documents: list[str]) -> None:
        """Backward-compatible: index plain strings without metadata."""
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        vectors = self.embedding_client.embed_texts(documents)
        self._ensure_collection(len(vectors[0]))

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{document}")),
                vector=vector,
                payload={"document": document},
            )
            for document, vector in zip(documents, vectors)
        ]
        logger.info("Indexed %d documents into Qdrant collection '%s'", len(documents), self.collection_name)

    def index_documents(
        self,
        docs: list[IngestionDocument],
        *,
        embedding_batch_size: int = 32,
    ) -> None:
        if not docs:
            logger.warning("No IngestionDocuments provided for indexing")
            return

        texts = [doc.to_embedding_text() for doc in docs]
        vectors = self.embedding_client.embed_texts_batched(texts, batch_size=embedding_batch_size)
        self._ensure_collection(len(vectors[0]))

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{doc.id}")),
                vector=vector,
                payload=doc.to_payload(),
            )
            for doc, vector in zip(docs, vectors)
        ]

        self._upsert_batch(points)
        logger.info(
            "Indexed %d IngestionDocuments into Qdrant collection '%s'",
            len(docs),
            self.collection_name,
        )

    def _upsert_batch(self, points: list[PointStruct]) -> None:
        total = len(points)
        n_batches = (total + self.upsert_batch_size - 1) // self.upsert_batch_size

        for i in range(0, total, self.upsert_batch_size):
            batch = points[i : i + self.upsert_batch_size]
            batch_num = i // self.upsert_batch_size + 1
            logger.debug("Upserting batch %d/%d (%d points)", batch_num, n_batches, len(batch))
            self._upsert_with_retry(batch)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _upsert_with_retry(self, batch: list[PointStruct]) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=batch,
            wait=True,
        )
        # Fase 12: validate that Qdrant count increased as expected after upsert
        try:
            result = self.client.count(collection_name=self.collection_name, exact=True)
            actual = result.count
            if actual == 0:
                logger.warning(
                    "Post-upsert count check: collection '%s' reports 0 points — possible silent failure",
                    self.collection_name,
                )
        except Exception as exc:
            logger.warning("Post-upsert count check failed: %s", exc)

    def _doc_point_id(self, doc: IngestionDocument) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{doc.id}"))

    def filter_new_docs(self, docs: list[IngestionDocument]) -> list[IngestionDocument]:
        """Return only docs not already present in the collection."""
        if not docs:
            return []

        if not self.client.collection_exists(self.collection_name):
            return docs

        id_to_doc = {self._doc_point_id(doc): doc for doc in docs}
        all_ids = list(id_to_doc.keys())

        existing_ids: set[str] = set()
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            try:
                results = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=batch_ids,
                    with_payload=False,
                    with_vectors=False,
                )
                existing_ids.update(str(r.id) for r in results)
            except Exception as exc:
                logger.warning("Failed to check existing docs (batch %d): %s", i // batch_size, exc)

        new_docs = [doc for pid, doc in id_to_doc.items() if pid not in existing_ids]
        return new_docs

    # ------------------------------------------------------------------
    # Fase 4: collection introspection helpers
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> dict:
        """Return basic stats about the Qdrant collection.

        Returns a dict with keys ``count`` (number of indexed vectors) and
        ``vector_size`` (embedding dimensionality).  Returns zeros if the
        collection does not yet exist.
        """
        if not self.client.collection_exists(self.collection_name):
            return {"count": 0, "vector_size": 0}
        try:
            info = self.client.get_collection(self.collection_name)
            count = info.vectors_count or 0
            size = info.config.params.vectors.size
            return {"count": count, "vector_size": size}
        except Exception as exc:
            logger.warning("get_collection_stats failed: %s", exc)
            return {"count": 0, "vector_size": 0}

    def sample_vectors(self, limit: int = 200) -> list[list[float]]:
        """Return up to *limit* raw vectors from the collection via scroll.

        Used for vector health metrics (norm distribution).  Returns an empty
        list when the collection is absent or has no vectors.
        """
        if not self.client.collection_exists(self.collection_name):
            return []
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_vectors=True,
                with_payload=False,
            )
            vectors: list[list[float]] = []
            for record in records:
                if record.vector is not None:
                    v = record.vector
                    vectors.append(v if isinstance(v, list) else list(v))
            return vectors
        except Exception as exc:
            logger.warning("sample_vectors failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Fase 8: embedding model version guard
    # ------------------------------------------------------------------

    def validate_embedding_model(self, model_name: str) -> str | None:
        """Check whether the collection was built with the same embedding model.

        Scrolls one point and reads ``payload["embedding_model"]``.

        Returns:
            - ``None`` if the collection is empty (no mismatch possible).
            - ``"unknown"`` if the field is absent (pre-versioned corpus).
            - The stored model name if it **differs** from *model_name*.
            - Raises nothing — callers decide what to do with the result.
        """
        if not self.client.collection_exists(self.collection_name):
            return None
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_vectors=False,
                with_payload=True,
            )
            if not records:
                return None
            stored_model = (records[0].payload or {}).get("embedding_model")
            if stored_model is None:
                return "unknown"
            if stored_model != model_name:
                return str(stored_model)
            return None
        except Exception as exc:
            logger.warning("validate_embedding_model failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Fase 11: soft rollback helper
    # ------------------------------------------------------------------

    def rollback_run(self, point_ids: list[str]) -> None:
        """Delete a specific set of points by ID (manual rollback after partial failure).

        Does nothing if *point_ids* is empty or the collection does not exist.
        """
        if not point_ids or not self.client.collection_exists(self.collection_name):
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=point_ids),
                wait=True,
            )
            logger.info("Rollback: deleted %d points from '%s'", len(point_ids), self.collection_name)
        except Exception as exc:
            logger.error("rollback_run failed: %s", exc)

    def _rank_search_results(
        self, query: str, results: list
    ) -> list[tuple[float, str, dict]]:
        """Rank raw Qdrant results by hybrid (dense + keyword) score.

        Returns a list of (hybrid_score, document_text, payload) sorted descending.
        """
        ranked: list[tuple[float, str, dict]] = []
        for result in results:
            document = str(result.payload.get("document", "")).strip()
            if not document:
                continue
            dense_score = float(result.score)
            keyword_score = _keyword_overlap_score(query, document)
            hybrid_score = (dense_score * self.dense_weight) + (keyword_score * self.keyword_weight)
            logger.debug(
                "Retrieved document dense=%.4f keyword=%.4f hybrid=%.4f: %s",
                dense_score,
                keyword_score,
                hybrid_score,
                document[:80],
            )
            ranked.append((hybrid_score, document, result.payload))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _query_qdrant(self, query_vector: list[float]) -> list:
        """Run a Qdrant query_points call and return raw result points."""
        try:
            self._ensure_collection(len(query_vector))
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.candidate_limit,
                with_payload=True,
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            raise VectorStoreUnavailableError(str(exc)) from exc
        return response.points if hasattr(response, "points") else response

    def search(self, query: str) -> list[str]:
        query_vector = self.embedding_client.embed_text(query)
        logger.debug(
            "Query embedding (%d dims, first 8 values): %s",
            len(query_vector),
            [round(value, 4) for value in query_vector[:8]],
        )
        results = self._query_qdrant(query_vector)
        ranked = self._rank_search_results(query, results)
        return [doc for _, doc, _ in ranked[: self.search_limit]]

    def search_with_metadata(self, query: str) -> list[dict]:
        """Like search() but returns structured dicts with id, text, title, and tags.

        Used by the evaluation pipeline to enable tag-based recall scoring.
        """
        query_vector = self.embedding_client.embed_text(query)
        results = self._query_qdrant(query_vector)
        ranked = self._rank_search_results(query, results)
        return [
            {
                "id": payload.get("id", ""),
                "text": doc,
                "title": payload.get("title", ""),
                "tags": payload.get("tags", []),
                "parent_id": payload.get("parent_id", ""),
            }
            for _, doc, payload in ranked[: self.search_limit]
        ]

    def search_filtered(
        self,
        query: str,
        *,
        source: str | None = None,
        service: str | None = None,
        severity: str | list[str] | None = None,
    ) -> list[str]:
        query_vector = self.embedding_client.embed_text(query)

        must_conditions = []
        if source:
            must_conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
        if service:
            must_conditions.append(FieldCondition(key="service", match=MatchValue(value=service)))
        if severity:
            if isinstance(severity, str):
                must_conditions.append(FieldCondition(key="severity", match=MatchValue(value=severity)))
            else:
                must_conditions.append(FieldCondition(key="severity", match=MatchAny(any=severity)))

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        try:
            self._ensure_collection(len(query_vector))
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=self.candidate_limit,
                with_payload=True,
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            raise VectorStoreUnavailableError(str(exc)) from exc

        results = response.points if hasattr(response, "points") else response

        ranked_results: list[tuple[float, str]] = []
        for result in results:
            document = str(result.payload.get("document", "")).strip()
            if not document:
                continue
            dense_score = float(result.score)
            keyword_score = _keyword_overlap_score(query, document)
            hybrid_score = (dense_score * self.dense_weight) + (keyword_score * self.keyword_weight)
            ranked_results.append((hybrid_score, document))

        ranked_results.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked_results[: self.search_limit]]

from __future__ import annotations

import logging
import re
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ariadne.core.integrations.embeddings.base import EmbeddingClient
from ariadne.core.retrieval.vector_stores.base import VectorStore, VectorStoreUnavailableError


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9_.:/-]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "after",
        "and",
        "at",
        "because",
        "before",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "while",
        "with",
    }
)


def _tokenize_search_text(text: str) -> set[str]:
    tokens = {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }
    return tokens


def _keyword_overlap_score(query: str, document: str) -> float:
    query_tokens = _tokenize_search_text(query)
    if not query_tokens:
        return 0.0

    document_tokens = _tokenize_search_text(document)
    if not document_tokens:
        return 0.0

    overlap = query_tokens & document_tokens
    return len(overlap) / len(query_tokens)


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
    ) -> None:
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self.search_limit = search_limit
        self.candidate_limit = max(candidate_limit, search_limit)
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
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
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def index(self, documents: list[str]) -> None:
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

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info("Indexed %d documents into Qdrant collection '%s'", len(documents), self.collection_name)

    def search(self, query: str) -> list[str]:
        query_vector = self.embedding_client.embed_text(query)
        logger.debug(
            "Query embedding (%d dims, first 8 values): %s",
            len(query_vector),
            [round(value, 4) for value in query_vector[:8]],
        )

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

        results = response.points if hasattr(response, "points") else response

        ranked_results: list[tuple[float, str]] = []
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
                document,
            )

            ranked_results.append((hybrid_score, document))

        ranked_results.sort(key=lambda item: item[0], reverse=True)
        documents = [document for _, document in ranked_results[: self.search_limit]]

        return documents

"""Configuration helpers for the incident analyzer.

Design choice:
- Centralize provider selection so agents never need to know about environment variables.

Tradeoff:
- This keeps configuration simple but assumes a single global provider per run.

Production caveat:
- Multi-tenant or per-request routing would need explicit config objects, not module-level env reads.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import os

from dotenv import load_dotenv

from ariadne.core.integrations.embeddings import (
    EmbeddingClient,
    GeminiEmbeddingClient,
    LocalHashEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAIEmbeddingClient,
)
from ariadne.core.integrations.llm import GeminiClient, LLMClient, OllamaClient, OpenAIClient
from ariadne.core.retrieval.vector_stores import NoOpVectorStore, QdrantVectorStore, VectorStore

load_dotenv()

logger = logging.getLogger(__name__)


def _get_env_value(*names: str, default: str = "") -> str:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            continue

        value = raw_value.strip()
        if value:
            return value

    return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_langsmith_api_key() -> str:
    return _get_env_value("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")


def get_langsmith_project() -> str:
    return _get_env_value("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT", default="ariadne")


def get_langsmith_endpoint() -> str:
    return _get_env_value(
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_ENDPOINT",
        default="https://api.smith.langchain.com",
    )


def get_langsmith_workspace_id() -> str:
    return _get_env_value("LANGCHAIN_WORKSPACE_ID", "LANGSMITH_WORKSPACE_ID")


def is_langsmith_enabled() -> bool:
    """Return True when LangSmith tracing environment is configured."""
    return (
        (_env_flag("LANGCHAIN_TRACING_V2") or _env_flag("LANGSMITH_TRACING"))
        and bool(get_langsmith_api_key())
    )


def get_provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def get_embedding_provider_name() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()


def get_vector_store_name() -> str:
    return os.getenv("VECTOR_STORE", "qdrant").strip().lower()


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    provider = get_provider_name()
    logger.info("Initializing LLM client for provider '%s'", provider)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIClient(api_key=api_key, model=model)

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaClient(model=model, base_url=base_url)

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiClient(api_key=api_key, model=model)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    provider = get_embedding_provider_name()
    logger.info("Initializing embedding client for provider '%s'", provider)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddingClient(api_key=api_key, model=model)

    if provider == "ollama":
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        timeout = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "60"))
        return OllamaEmbeddingClient(model=model, base_url=base_url, timeout=timeout)

    if provider == "local_hash":
        dimensions = int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "256"))
        return LocalHashEmbeddingClient(dimensions=dimensions)

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        return GeminiEmbeddingClient(api_key=api_key, model=model)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    provider = get_vector_store_name()
    logger.info("Initializing vector store for provider '%s'", provider)

    if provider == "qdrant":
        search_limit = int(os.getenv("QDRANT_SEARCH_LIMIT", "3"))
        candidate_limit = int(os.getenv("QDRANT_CANDIDATE_LIMIT", "8"))
        dense_weight = float(os.getenv("QDRANT_DENSE_WEIGHT", "0.65"))
        keyword_weight = float(os.getenv("QDRANT_KEYWORD_WEIGHT", "0.35"))
        timeout = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "10"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        return QdrantVectorStore(
            embedding_client=get_embedding_client(),
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION", "incident_knowledge"),
            search_limit=search_limit,
            candidate_limit=candidate_limit,
            dense_weight=dense_weight,
            keyword_weight=keyword_weight,
            timeout=timeout,
            api_key=qdrant_api_key,
        )

    if provider == "none":
        return NoOpVectorStore()

    if _env_flag("ALLOW_UNCONFIGURED_VECTOR_STORE_FALLBACK"):
        logger.warning("Unknown VECTOR_STORE '%s'; falling back to no-op retrieval", provider)
        return NoOpVectorStore()

    raise ValueError(f"Unsupported VECTOR_STORE: {provider}")

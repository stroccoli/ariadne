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
    CachedEmbeddingClient,
    EmbeddingClient,
    GeminiEmbeddingClient,
    InMemoryEmbeddingCache,
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
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE") or None
        num_ctx_raw = os.getenv("OLLAMA_NUM_CTX")
        num_ctx = int(num_ctx_raw) if num_ctx_raw else None
        return OllamaClient(model=model, base_url=base_url, keep_alive=keep_alive, num_ctx=num_ctx)

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

    client: EmbeddingClient
    model_name: str = provider  # fallback label for cache keys

    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        client = OpenAIEmbeddingClient(api_key=api_key, model=model_name, dimensions=embedding_dimensions)

    elif provider == "ollama":
        model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        timeout = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "60"))
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE") or None
        client = OllamaEmbeddingClient(model=model_name, base_url=base_url, timeout=timeout, keep_alive=keep_alive, dimensions=embedding_dimensions)

    elif provider == "local_hash":
        dimensions = int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "256"))
        client = LocalHashEmbeddingClient(dimensions=dimensions)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        client = GeminiEmbeddingClient(api_key=api_key, model=model_name, dimensions=embedding_dimensions)

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")

    return _maybe_wrap_with_cache(client, model_name)


def _maybe_wrap_with_cache(client: EmbeddingClient, model_name: str) -> EmbeddingClient:
    """Wrap *client* with an embedding cache.

    Uses Redis when ``REDIS_URL`` is configured (e.g. local dev with docker-compose).
    Falls back to an in-process LRU cache otherwise — the default for production
    where no external cache service is available.
    """
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        ttl = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))
        try:
            import redis

            rc = redis.Redis.from_url(redis_url, decode_responses=False, socket_connect_timeout=3)
            rc.ping()
            logger.info("Redis embedding cache enabled (ttl=%ds, model=%s)", ttl, model_name)
            return CachedEmbeddingClient(inner=client, redis_client=rc, model_name=model_name, ttl_seconds=ttl)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); falling back to in-process cache", exc)

    max_size = int(os.getenv("EMBEDDING_CACHE_SIZE", "2048"))
    logger.info("In-process LRU embedding cache enabled (max_size=%d, model=%s)", max_size, model_name)
    return InMemoryEmbeddingCache(inner=client, model_name=model_name, max_size=max_size)


def get_qdrant_collection_name() -> str:
    """Build the Qdrant collection name, suffixed by the embedding provider.

    Convention: ``{base}_{embedding_provider}``
    e.g. ``incident_knowledge_openai``.
    """
    base = os.getenv("QDRANT_COLLECTION", "incident_knowledge")
    embedding_provider = get_embedding_provider_name()
    return f"{base}_{embedding_provider}"


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    provider = get_vector_store_name()
    logger.info("Initializing vector store for provider '%s'", provider)

    if provider == "qdrant":
        collection_name = get_qdrant_collection_name()
        search_limit = int(os.getenv("QDRANT_SEARCH_LIMIT", "3"))
        candidate_limit = int(os.getenv("QDRANT_CANDIDATE_LIMIT", "8"))
        dense_weight = float(os.getenv("QDRANT_DENSE_WEIGHT", "0.65"))
        keyword_weight = float(os.getenv("QDRANT_KEYWORD_WEIGHT", "0.35"))
        timeout = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "10"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        logger.info(
            "Using Qdrant collection '%s' (embedding_provider='%s')",
            collection_name,
            get_embedding_provider_name(),
        )
        return QdrantVectorStore(
            embedding_client=get_embedding_client(),
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=collection_name,
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


def reset_provider_caches() -> None:
    """Clear cached provider singletons.

    Call this when switching ``LLM_PROVIDER`` / ``EMBEDDING_PROVIDER`` env vars
    within the same process (e.g. multi-provider ingestion scripts).
    """
    get_llm_client.cache_clear()
    get_embedding_client.cache_clear()
    get_vector_store.cache_clear()

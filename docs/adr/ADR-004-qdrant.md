# ADR-004: Qdrant Cloud over pgvector, FAISS, or Pinecone

## Status: Accepted

## Date: 2026-03-24

---

## Context

Ariadne's RAG layer retrieves incident pattern documents from a vector store and injects the top-3 results into the analyzer prompt. The knowledge base is populated by `scripts/ingest_pipeline.py`, which collects data from GitHub issues and public post-mortems, preprocesses, chunks, and indexes them as structured `IngestionDocument` objects.

Local development already uses Qdrant via Docker Compose (`docker-compose.yml`, ports 6333/6334). The `QdrantVectorStore` implementation uses hybrid search: `QDRANT_DENSE_WEIGHT=0.65`, `QDRANT_KEYWORD_WEIGHT=0.35`, `QDRANT_CANDIDATE_LIMIT=8`, `QDRANT_SEARCH_LIMIT=3`.

The question is: what managed vector store to use in the Fly.io production deployment, where no self-hosted infrastructure can run alongside the API container?

---

## Decision

Use **Qdrant Cloud free tier** for the production vector store.

The `QdrantVectorStore` implementation already accepts `url` and optional `api_key` via `QDRANT_URL` and `QDRANT_API_KEY` env vars. Switching from local to cloud is a two-line `.env` change:

```bash
QDRANT_URL=https://<cluster-id>.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>
```

The `qdrant-client` library handles both local and cloud connections with the same API. **No code changes required.**

Qdrant Cloud free tier (Q1 2026): 1 cluster, 1 GB storage, no expiry.

For 22 documents at 768 dimensions the free tier is ~200× over-provisioned.

---

## Alternatives considered

### pgvector on Supabase free tier

Supabase provides managed PostgreSQL with the `pgvector` extension. Free tier: 500 MB database.

**Why not chosen:**
- The existing `QdrantVectorStore` would need to be replaced or a `pgvector` adapter written — a meaningful refactor.
- Supabase's free tier **pauses the database after 7 days of inactivity**, requiring a manual wake-up or keep-alive script. For a portfolio project that may not be visited for weeks, this is a reliability problem.
- Hybrid BM25 + dense search in pgvector requires additional extensions (`pg_trgm`, `paradedb`) and more complex queries. Qdrant's hybrid search is built-in and configured via query parameters.

**Could be revisited if:** the project adds relational data (incident history, user management) and the vector store moves alongside an existing Supabase database.

### FAISS in-memory

FAISS is high-performance approximate nearest-neighbour search, in-process and in-memory.

**Why not chosen:**
- On Fly.io with `min_machines_running = 0`, the machine stops when idle. Every resume would need to re-load and re-index from disk, adding 1–3s to the first request.
- FAISS does not support hybrid BM25 + dense search natively.
- No FAISS implementation exists in the codebase — adding one is more work than pointing `QDRANT_URL` at a cloud cluster.

**Could be revisited if:** the machine runs always-on with pre-loaded state, and hybrid search is no longer a requirement.

### Pinecone free tier

Pinecone is a managed vector database. Free tier: 1 index, 100K vectors, no expiry.

**Why not chosen:**
- Requires a custom `pinecone-client` integration — extra dependency and code.
- Pinecone serverless pods have higher latency on the first query after a period of inactivity (similar to Supabase's pause behaviour).
- Qdrant Cloud's free tier is more generous (1 GB vs Pinecone's ~100K 768-dim vectors), and the local-to-cloud transition is zero-code since `QdrantVectorStore` handles both.

---

## Consequences

**Positive:**
- Local development and production use the same `QdrantVectorStore` class — no separate code path.
- Hybrid search is already configured and tested locally before being promoted to cloud.
- The ingestion pipeline (`scripts/ingest_pipeline.py`) works identically against both endpoints.
- Qdrant Cloud persists state; no re-indexing on API restart.

**Negative:**
- Production depends on an external managed service that could change pricing or availability.
- Re-indexing requires running `python scripts/ingest_pipeline.py` with `QDRANT_URL` and `QDRANT_API_KEY` set — no background indexing endpoint.
- The 1 GB free tier, while ample now, would be exhausted at ~40,000 documents indexed at 768 dimensions.

---

## When would we revisit this?

- If the knowledge base grows beyond ~10,000 documents and approaches the 1 GB free tier limit.
- If hybrid search is no longer a priority and a simpler alternative (pgvector, in-memory FAISS) becomes more attractive.
- If Qdrant Cloud free tier terms change.
- If the project migrates to a known cloud provider where a native managed vector store reduces external dependencies.

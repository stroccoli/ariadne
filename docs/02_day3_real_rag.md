# Day 3: Real RAG

## What changed

Day 3 replaces the heuristic retrieval stub with a real vector search pipeline:

1. Logs are embedded with a configurable embedding provider.
2. Documents are indexed into Qdrant.
3. Qdrant performs cosine similarity search.
4. The top matches are injected into the analyzer prompt as supporting context.

## Why use Ollama embeddings locally

The earlier fallback local embedder was deterministic and useful for debugging, but it was not a semantic model.

- It hashed tokens into a vector space.
- It preserved some lexical overlap.
- It did not learn semantic relationships between phrases.

Using `nomic-embed-text:latest` through Ollama fixes the main quality limitation of that approach.

- It runs locally.
- It is lightweight enough for this project scope.
- It produces semantically meaningful embeddings.
- It keeps the embedding layer swappable because the application still depends on an `EmbeddingClient` abstraction.

## Why the refactor uses folders

The refactor groups code by responsibility using conventional names:

- `integrations/`: outbound providers such as OpenAI and Ollama
- `retrieval/`: vector-store logic and knowledge loading
- `agents/`: task-level pipeline agents such as classification, retrieval, and analysis

This keeps abstract base classes and concrete implementations separate.

That matters because:

- files become smaller and easier to scan
- provider-specific code stops leaking into unrelated modules
- swapping implementations becomes a configuration problem instead of a code surgery problem

## Remaining limitations

### No metadata filters or hybrid keyword search

The current store returns nearest neighbors based only on dense vector similarity.

- If you later attach metadata like service name, environment, or incident type, you still cannot filter on it.
- If a query contains critical literal tokens like `SQLSTATE 40001` or `OOMKilled`, there is no BM25 or exact-match layer to guarantee those tokens influence ranking strongly.

### No reranker

The first-pass retrieval ranking is accepted as final.

That is fine for a small demo corpus, but weaker once the corpus grows and similar documents cluster together.

### Plain-text context instead of structured citations

The analyzer receives text blocks, not a structured list of source objects.

So the final answer cannot cleanly say which document was used, when it was authored, or whether it is stale.

### No background indexing or freshness workflow

Knowledge updates are manual.

For production you would want:

- change detection for knowledge sources
- automatic re-embedding and reindexing
- collection migrations when embedding dimensions change
- monitoring for stale or failed indexing jobs

# Ariadne — AI Incident Analyzer

Multi-agent incident analysis pipeline built with LangGraph + FastAPI. Takes raw
incident logs, classifies the failure mode, retrieves relevant troubleshooting
context via RAG, and produces a structured root-cause report with confidence
scores and recommended actions.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
<!-- ![CI](https://github.com/YOUR_USER/ariadne/actions/workflows/deploy.yml/badge.svg) -->

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USER/ariadne.git
cd ariadne

# 2. Copy environment template and fill in your API keys
cp .env.example .env
# Edit .env — at minimum set LLM_PROVIDER and its corresponding API key

# 3. Create a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 4. Start Qdrant (needed for RAG retrieval)
docker compose up -d

# 5. Index the knowledge base
python scripts/index_data.py

# 6. Run the API
uvicorn ariadne.api.main:app --reload --port 8000
```

### Test it

```bash
# Health check
curl http://localhost:8000/health

# Analyze an incident
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"logs": "ERROR: connection timeout after 30s to payments-api.internal:8443"}'
```

Interactive API docs are available at `http://localhost:8000/docs`.

## Architecture

The pipeline runs four LangGraph nodes in sequence with a conditional retry loop:

```
START → classify → retrieve → analyze ──→ build_output → END
                      ↑          │
                      └──────────┘
                   (retry if confidence < 0.7,
                    max 2 retrieval attempts)
```

<!-- TODO: Day 5 — full architecture diagram with Mermaid -->

### Key components

| Layer | Path | Purpose |
|-------|------|---------|
| API | `ariadne/api/` | FastAPI wrapper — `POST /analyze`, `GET /health`, `GET /ready` |
| Pipeline | `ariadne/core/graph.py` | LangGraph orchestration and retry logic |
| Agents | `ariadne/core/agents/` | Classifier, RAG retriever, Analyzer |
| LLM providers | `ariadne/core/integrations/llm/` | OpenAI, Ollama, Gemini — swappable |
| Embeddings | `ariadne/core/integrations/embeddings/` | OpenAI, Ollama, Gemini, local hash |
| Vector store | `ariadne/core/retrieval/` | Qdrant (hybrid search) or NoOp fallback |
| Evals | `evals/` | Offline A/B prompt benchmarks |

## Environment Variables

See [.env.example](.env.example) for the full list with defaults and documentation.

Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | Yes | `openai`, `ollama`, or `gemini` |
| `OPENAI_API_KEY` | If provider=openai | OpenAI API key |
| `GEMINI_API_KEY` | If provider=gemini | Google Gemini API key |
| `EMBEDDING_PROVIDER` | Yes | `openai`, `ollama`, `gemini`, or `local_hash` |
| `VECTOR_STORE` | Yes | `qdrant` or `none` |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins (defaults to localhost) |

## Running Evaluations

The `evals/` directory contains an offline A/B testing framework that benchmarks
`detailed` vs `compact` prompt modes across 40+ incident samples with rubric-based
scoring.

```bash
# Run a full A/B test
python evals/run_ab_test.py

# Run provider benchmark (e.g. 10 samples)
python evals/run_provider_test.py -n 10

# List available samples
python evals/run_ab_test.py --list-samples

# Compare two saved benchmark runs
python evals/compare_results.py
```

Results are saved to `evals/results/` (gitignored — these are local artifacts).

The framework measures: incident type accuracy, root-cause quality (rubric-scored),
action quality, latency, and token usage.

## API Reference

### `POST /analyze`

Analyze incident logs and return a structured report.

**Request:**
```json
{
  "logs": "ERROR: connection timeout after 30s to payments-api.internal:8443",
  "mode": "detailed"
}
```

**Response:**
```json
{
  "incident_type": "timeout",
  "root_cause": "The payments API is not responding within the expected timeframe...",
  "confidence": 0.85,
  "recommended_actions": [
    "Check payments-api health and recent deployments",
    "Review network connectivity to payments-api.internal:8443"
  ],
  "metadata": {
    "retrieval_attempts": 1,
    "llm_calls": 2,
    "node_timings": { "classify": 1.2, "retrieve_1": 0.1, "analyze": 2.5 },
    "usage": { "prompt_tokens": 530, "completion_tokens": 85, "total_tokens": 615 }
  }
}
```

### `GET /health`
Returns `{"status": "ok"}` — liveness check.

### `GET /ready`
Returns `{"status": "ready"}` — readiness check.

## Deployment

The backend deploys to [Fly.io](https://fly.io) via GitHub Actions on push to `main`.

Infrastructure files live in `infra/`:
- `infra/Dockerfile` — production container image
- `infra/fly.toml` — Fly.io machine configuration

See `.github/workflows/deploy.yml` for the full CI/CD pipeline (test → deploy → smoke test).

## Project Structure

```
ariadne/
  api/                        FastAPI application
    main.py                   App setup, CORS, lifespan
    routes/                   Endpoint handlers
    models/                   Request/response schemas
  core/                       Pipeline logic
    config.py                 Env-driven factory for LLM, embedding, vector store
    models.py                 Typed output schemas (Pydantic)
    state.py                  Shared IncidentState
    graph.py                  LangGraph orchestration and retry logic
    main.py                   CLI entry point
    agents/                   Classifier, RAG, Analyzer agents
    integrations/             LLM and embedding provider implementations
    retrieval/                Vector store abstraction and dataset loading
data/                         Knowledge base (incident_knowledge.json)
evals/                        Offline A/B benchmark framework
infra/                        Dockerfile, fly.toml
scripts/                      Data indexing utilities
tests/                        Unit and integration tests
```

## TODO

- [ ] Architecture diagram (Day 5)
- [ ] Frontend UI + Vercel deployment (Day 3B)
- [ ] Contributing guide
- [ ] Qdrant Cloud integration for production RAG
- [ ] Custom domain setup
- [ ] Rate limiting and authentication
# ariadne

# Ariadne – Architecture Document

> *Ariadne’s Thread: Guiding you from symptoms to root cause*

---

## 1. Overview

**Ariadne** is an LLM-based system designed to guide users from their application's problematic logs to identifying the root cause. It uses an agent-based architecture (orchestrated with Langraph) and a RAG pipeline to provide accurate and contextual responses.
Note: The tool is designed to run 100% locally.

### Main Stack
| Layer | Technology |
|---|---|
| Backend | Python 3.11 |
| LLM Provider | Ollama (can be switched to OpenAI or Gemini) |
| Vector Store | Qdrant |
| Agents Orchestration | Langraph |
| CI/CD | GitHub Actions to Fly.io |

---

## 2. System Architecture

Ariadne operates in two modes: an **offline indexing pipeline** that processes incident knowledge into a vector store, and an **online query pipeline** where a LangGraph agent retrieves and reasons over that knowledge in real time.

---

### 2.1 Offline Indexing Pipeline

The indexing pipeline runs offline via DVC to populate Qdrant with embedded incident knowledge. This happens before any queries can be served.

```mermaid
flowchart TD
    A[📄 Raw Data<br/>Postmortems & GitHub Issues] --> B[🔧 Preprocess<br/>Clean & Filter]
    B --> C[✂️ Chunk<br/>Split into segments]
    C --> D[🧮 Embed<br/>Convert to vectors]
    D --> E[💾 Store<br/>Upsert to Qdrant]

    style A fill:#e3f2fd,stroke:#1976D2,color:#000
    style B fill:#e8eaf6,stroke:#3949AB,color:#000
    style C fill:#fff3e0,stroke:#F57C00,color:#000
    style D fill:#f3e5f5,stroke:#7B1FA2,color:#000
    style E fill:#e8f5e9,stroke:#2E7D32,color:#000
```

> **DVC Pipeline:** `collect → preprocess → chunk → index_{provider} → evaluate → diagnose`. Each stage produces metrics and reports for quality monitoring.

---

### 2.2 Online Query Pipeline

When a user submits incident logs, the system processes them through the LangGraph agent in real time.

```mermaid
flowchart LR
    user[👤 User<br/>Incident Logs] --> api[🌐 FastAPI<br/>/api/v1/analyze]
    api --> agent[🤖 LangGraph<br/>Agent]
    agent --> qdrant[(🗄️ Qdrant<br/>Vector Store)]
    qdrant --> agent
    agent --> llm[🧠 LLM<br/>Ollama/OpenAI/Gemini]
    llm --> agent
    agent --> answer[📋 Response<br/>Root Cause + Remediation]

    style user fill:#e3f2fd,stroke:#1976D2,color:#000
    style api fill:#e8eaf6,stroke:#3949AB,color:#000
    style agent fill:#fff3e0,stroke:#F57C00,color:#000
    style qdrant fill:#f3e5f5,stroke:#7B1FA2,color:#000
    style llm fill:#fce4ec,stroke:#C62828,color:#000
    style answer fill:#e8f5e9,stroke:#2E7D32,color:#000
```

> **Real-time flow:** User input → classification → retrieval → analysis → structured output. The agent can retry retrieval if confidence is low.

---

### 2.3 LangGraph Agent Flow

The agent is a **stateful directed graph**. Each node performs a discrete reasoning step; the conditional edge after `analyze` enables automatic retry when LLM confidence is below threshold.

```mermaid
flowchart LR
    START([▶ START]) --> classify[🏷️ Classify<br/>Detect incident type<br/>& severity]
    classify --> retrieve[🔍 Retrieve<br/>Hybrid vector + keyword<br/>search in Qdrant]
    retrieve --> analyze[🧠 Analyze<br/>LLM synthesizes context<br/>into root-cause diagnosis]
    analyze --> check{Confidence ≥ 0.7?<br/>or max retries reached?}
    check -->|✅ Yes — done| build[📋 Build Output<br/>Format structured response]
    check -->|🔄 No — retry| retrieve
    build --> END([⏹ END])

    style START fill:#1976D2,stroke:#1976D2,color:#fff
    style classify fill:#bbdefb,stroke:#1976D2,color:#000
    style retrieve fill:#e1bee7,stroke:#6A1B9A,color:#000
    style analyze fill:#ffe0b2,stroke:#E65100,color:#000
    style check fill:#fff9c4,stroke:#F9A825,color:#000
    style build fill:#c8e6c9,stroke:#2E7D32,color:#000
    style END fill:#2E7D32,stroke:#2E7D32,color:#fff
```

> **Retry logic:** if the LLM's confidence score is below `0.7` and fewer than 2 retrieval attempts have been made, the graph loops back to `retrieve` with the same query — allowing the agent to self-correct before producing its final answer.

---

### Key Components

| Component | Role |
|---|---|
| **FastAPI** | REST layer — receives incident logs, returns structured JSON analysis |
| **LangGraph Agent** | Orchestrates the `classify → retrieve → analyze` loop with retry |
| **Classifier** | LLM call to identify incident type and severity from raw logs |
| **Retriever** | Hybrid search (cosine × 0.65 + keyword overlap × 0.35) against Qdrant |
| **Analyzer** | LLM call that synthesizes retrieved context into a root-cause explanation |
| **Qdrant** | Vector store holding embedded postmortems and GitHub issues |
| **LLM** | Ollama (local) / OpenAI / Gemini — swappable via `LLM_PROVIDER` env var |

---

## 3. Agentes

### Agente Principal: [nombre]
- **Responsabilidad**: 
- **Herramientas disponibles**: 
- **Estrategia de prompting**: 
- **Flujo de decisión**: 

### [Otros agentes si aplica]

---

## 4. Pipeline RAG (Retrieval-Augmented Generation)

- **Fuente de datos**: 
- **Chunking strategy**: 
- **Modelo de embeddings**: 
- **Vector Store**: 
- **Estrategia de retrieval**: 

---

## 5. Evaluación

- **Framework de evaluación**: 
- **Métricas clave**: 
- **Dataset de evaluación**: 
- **Resultados baseline**: 

---

## 6. Monitoreo

- **Health check**: `GET /health`
- **Ready check**: `GET /ready`
- **Logging**: 
- **Alertas**: 

---

## 7. CI/CD Pipeline

```
push/PR → [test] → (solo main) → [deploy-backend] → [smoke-test]
```

1. **test**: Corre `pytest tests/unit` con `VECTOR_STORE=none` (sin dependencias externas)
2. **deploy-backend**: `flyctl deploy` desde `infra/fly.toml`
3. **smoke-test**: Verifica `/health` y `/ready` post-deploy

---

## 8. Decisiones Arquitectónicas

### ADR-001: [Título de la decisión]
- **Contexto**: 
- **Decisión**: 
- **Consecuencias**: 

### ADR-002: VECTOR_STORE=none en tests unitarios
- **Contexto**: Los tests unitarios no deben depender de servicios externos
- **Decisión**: Se usa variable de entorno `VECTOR_STORE=none` para mockear el vector store
- **Consecuencias**: Tests rápidos y confiables en CI

---

## 9. Estructura del Proyecto

```
ariadne/
├── .github/workflows/
│   └── deploy.yml
├── infra/
│   └── fly.toml
├── tests/
│   └── unit/
├── requirements.txt
└── ARCHITECTURE.md
```

---

## 10. Historia y Evolución

### Semana 1 – [fecha]
- 

### Semana 2 – [fecha]
- 

---

## 11. Pendientes / Ideas Futuras

- [ ] 
- [ ] 
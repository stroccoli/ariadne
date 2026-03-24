# ADR-003: Gemini Flash as the primary production LLM

## Status: Accepted

## Date: 2026-03-24

---

## Context

Ariadne calls an LLM twice per pipeline run: once in the classifier node (incident type + confidence) and once in the analyzer node (root cause + recommended actions + confidence). On retry both calls happen again.

The codebase was built from Day 1 with a provider-agnostic `LLMClient` interface (`ariadne/core/integrations/llm/`). Three concrete implementations exist and work: `OpenAIClient`, `OllamaClient`, `GeminiClient`. The decision is which becomes the default for the public production deployment where budget is $0/month.

---

## Decision

Set `LLM_PROVIDER=gemini` and `GEMINI_MODEL=gemini-2.0-flash` as the production defaults in `ariadne/core/config.py`.

Google Gemini API free tier (Q1 2026 on `gemini-2.0-flash`):
- **15 RPM** (requests per minute)
- **1,000,000 tokens per day**

At ~600–800 tokens per call and 2 calls per analysis, the free tier supports roughly 600–800 analyses per day before hitting the token cap. Well above any realistic demo load.

`OpenAIClient` remains available (`LLM_PROVIDER=openai`) and is the provider used in the offline evaluation framework (`evals/`), where token cost is controlled by sample count.

`OllamaClient` remains the recommended development option for contributors who don't want to provision any API keys.

---

## Alternatives considered

### GPT-4o

GPT-4o produces the highest quality structured output of any model tested, with strong JSON adherence and nuanced root-cause explanations.

**Why rejected for production default:** no free tier. Cost at $2.50/1M input tokens is non-trivial at any sustained traffic level. Available via `LLM_PROVIDER=openai`, `OPENAI_MODEL=gpt-4o` for anyone willing to pay.

### GPT-4o-mini

Cheaper ($0.15/1M input tokens) and acceptable structured output quality.

**Why rejected:** still requires a paid OpenAI account. No permanent free tier — only a new-account credit that expires. Output quality is comparable to Gemini Flash for classification+analysis on incident logs, based on A/B eval runs in `evals/results/`.

### GPT-3.5-turbo

Deprecated by OpenAI in favour of GPT-4o-mini. Not a forward-looking choice.

**Rejected:** deprecated.

### Groq / Llama 3 (via Groq API)

Groq offers a free tier with Llama 3.1 8B and 70B. Inference speed is exceptional (hundreds of tokens/second).

**Why not chosen:** no `GroqClient` implementation exists in the codebase. Adding one is straightforward given the provider-agnostic interface but was not prioritised. Llama 3.1 8B's structured JSON output is less reliable than Gemini Flash on the few-shot classification tasks in the current prompts. This is a viable future option — see Roadmap in the README.

### Ollama (local models, `llama3.1:8b`)

Zero cost, zero network. Used during development.

**Why not for production default:** requires a running Ollama daemon. Deploying a 4–8 GB model alongside the API on a 512 MB Fly.io machine is not feasible.

**Kept as:** the recommended development option. Set `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=ollama` in `.env` for a fully local setup with no API keys.

---

## Consequences

**Positive:**
- Zero cost for production traffic at demo scale — permanent, not a trial credit.
- `gemini-2.0-flash` generates JSON reliably and follows system prompt instructions consistently.
- Same `GeminiClient` can handle embeddings via `EMBEDDING_PROVIDER=gemini`, making a fully Gemini-only stack possible with a single API key.
- Provider-agnostic design: switching to OpenAI in production is a one-line `.env` change.

**Negative:**
- Gemini's structured output API differs from OpenAI's `response_format={"type": "json_object"}` — prompts must elicit JSON without relying on OpenAI-specific parameters.
- Gemini Flash occasionally under-produces `recommended_actions` compared to GPT-4o on edge-case samples — visible in eval results in `evals/results/`.
- High-volume evals against Gemini can exhaust the free-tier RPM limit mid-run. Use `LLM_PROVIDER=openai` for bulk eval runs.

---

## When would we revisit this?

- If Gemini API response quality degrades noticeably on the A/B benchmark (tracked in `evals/results/`).
- If Groq gains a stable free tier with Llama 3.1 70B and a `GroqClient` is added.
- If Google changes Gemini's free tier terms unfavourably.
- If the project gains a sponsor — at that point GPT-4o is worth the cost for quality improvement on complex multi-service incidents.

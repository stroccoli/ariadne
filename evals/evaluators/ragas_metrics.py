"""RAGAS metric initialisation and the four LangSmith evaluator wrappers.

Metrics:
    faithfulness       — Is the answer grounded in the retrieved context?
    answer_relevancy   — Does the answer address the incident logs?
    context_precision  — Is relevant context ranked higher in the retrieved set?
    context_recall     — Does the retrieved context cover the expected answer?

The LLM used to score these metrics is controlled by EVAL_LLM_PROVIDER:
    ollama (default) → EVAL_OLLAMA_MODEL + OLLAMA_EMBEDDING_MODEL
    openai           → OPENAI_MODEL + OPENAI_EMBEDDING_MODEL
"""
from __future__ import annotations

import asyncio
import os

from langsmith.schemas import Example, Run
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

_metrics: dict[str, Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall] | None = None


def _get_metrics() -> dict:
    global _metrics
    if _metrics is None:
        # EVAL_LLM_PROVIDER controls which LLM scores the RAGAS metrics.
        # Defaults to "ollama". Set to "openai" in .env for a cloud fallback.
        # Note: the model must support reliable structured (JSON) output.
        #   ollama → uses EVAL_OLLAMA_MODEL (default: deepseek-r1:8b)
        #            and nomic-embed-text for embeddings
        #   openai → uses OPENAI_MODEL + OPENAI_EMBEDDING_MODEL
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory

        eval_provider = os.getenv("EVAL_LLM_PROVIDER", "ollama").lower()

        if eval_provider == "openai":
            from ragas.embeddings.openai_provider import OpenAIEmbeddings

            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            client = AsyncOpenAI()
            llm = llm_factory(model, provider="openai", client=client)
            embeddings = OpenAIEmbeddings(client=client, model=embedding_model)
        else:
            # Ollama — deepseek-r1:8b handles RAGAS structured prompts
            # better than smaller models due to its larger context window.
            from ragas.embeddings.litellm_provider import LiteLLMEmbeddings

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("EVAL_OLLAMA_MODEL", "deepseek-r1:8b").strip()
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

            client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="ollama")
            llm = llm_factory(model, provider="openai", client=client)
            embeddings = LiteLLMEmbeddings(
                model=f"ollama/{embedding_model}",
                api_base=base_url,
            )

        _metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }
    return _metrics


def _run_async(coro) -> float:
    """Execute an async RAGAS coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Notebook / async context — nest_asyncio must be installed
            import nest_asyncio  # type: ignore[import]
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def eval_faithfulness(run: Run, example: Example) -> dict:
    """Score: is the response grounded in the retrieved context?"""
    outputs = run.outputs or {}
    result = _run_async(_get_metrics()["faithfulness"].ascore(
        user_input=outputs.get("user_input", ""),
        response=outputs.get("response", ""),
        retrieved_contexts=outputs.get("retrieved_contexts") or [],
    ))
    return {"key": "faithfulness", "score": result.value}


def eval_answer_relevancy(run: Run, example: Example) -> dict:
    """Score: does the response address the incident logs?"""
    outputs = run.outputs or {}
    result = _run_async(_get_metrics()["answer_relevancy"].ascore(
        user_input=outputs.get("user_input", ""),
        response=outputs.get("response", ""),
    ))
    return {"key": "answer_relevancy", "score": result.value}


def eval_context_precision(run: Run, example: Example) -> dict:
    """Score: is relevant context ranked higher in the retrieved set?"""
    outputs = run.outputs or {}
    reference = (example.outputs or {}).get("reference", "")
    if not reference:
        return {"key": "context_precision", "score": None}
    result = _run_async(_get_metrics()["context_precision"].ascore(
        user_input=outputs.get("user_input", ""),
        reference=reference,
        retrieved_contexts=outputs.get("retrieved_contexts") or [],
    ))
    return {"key": "context_precision", "score": result.value}


def eval_context_recall(run: Run, example: Example) -> dict:
    """Score: does retrieved context cover the reference answer?"""
    outputs = run.outputs or {}
    reference = (example.outputs or {}).get("reference", "")
    if not reference:
        return {"key": "context_recall", "score": None}
    result = _run_async(_get_metrics()["context_recall"].ascore(
        user_input=outputs.get("user_input", ""),
        retrieved_contexts=outputs.get("retrieved_contexts") or [],
        reference=reference,
    ))
    return {"key": "context_recall", "score": result.value}

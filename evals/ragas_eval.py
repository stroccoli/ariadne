"""RAGAS evaluation runner for the Ariadne incident analysis pipeline.

Measures four standard RAG quality metrics and publishes results as a
native LangSmith Experiment for side-by-side comparison across runs.

Metrics:
    faithfulness       — Is the answer grounded in the retrieved context?
    answer_relevancy   — Does the answer address the incident logs?
    context_precision  — Is relevant context ranked higher in the retrieved set?
    context_recall     — Does the retrieved context cover the expected answer?

Usage:
    python evals/ragas_eval.py                             # all 50 samples
    python evals/ragas_eval.py --num-samples 5             # quick smoke-test
    python evals/ragas_eval.py --experiment-prefix v2      # named experiment
    python evals/ragas_eval.py --dataset my-dataset --num-samples 10
"""
# When invoked as `python evals/ragas_eval.py`, Python inserts evals/ into
# sys.path instead of the project root. Patch that here, before any imports,
# so that `evals` and `ariadne` packages resolve correctly.
if __name__ == "__main__":
    import pathlib
    import sys

    _root = str(pathlib.Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

import argparse
import asyncio
import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

# Suppress instructor's FutureWarning about google.generativeai at import time
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor")

import langsmith
from langsmith import evaluate
from langsmith.schemas import Example, Run

from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from ariadne.core.graph import run_graph
from evals.ragas_dataset import DEFAULT_DATASET_NAME, build_langsmith_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy metric initialisation (avoid paying LLM setup cost at import time)
# ---------------------------------------------------------------------------

_metrics: dict[str, Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall] | None = None


def _get_metrics() -> dict:
    global _metrics
    if _metrics is None:
        # EVAL_LLM_PROVIDER controls which LLM scores the RAGAS metrics.
        # Defaults to "ollama". Set to "openai" in .env for a cloud fallback.
        # Note: the model must support reliable structured (JSON) output.
        #   ollama → uses EVAL_OLLAMA_MODEL (default:  deepseek-r1:8b)
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
            # Ollama —  deepseek-r1:8b handles RAGAS structured prompts
            # better than smaller models due to its larger context window.
            from ragas.embeddings.litellm_provider import LiteLLMEmbeddings

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("EVAL_OLLAMA_MODEL", " deepseek-r1:8b")
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
# Pipeline target function
# ---------------------------------------------------------------------------

def run_pipeline(inputs: dict) -> dict:
    """Invoke the Ariadne graph and return RAGAS-compatible output fields.

    This function is passed as the target to ``langsmith.evaluate()``.
    Each call is automatically traced in LangSmith when tracing is enabled.
    """
    logs: str = inputs["logs"]
    mode: str = inputs.get("mode", "detailed")

    state = run_graph(logs, mode)

    final_output = state.final_output
    root_cause: str = final_output.root_cause if final_output else ""

    return {
        "user_input": logs,
        "retrieved_contexts": list(state.context or []),
        "response": root_cause,
    }


# ---------------------------------------------------------------------------
# Evaluators — one per RAGAS metric, using the LangSmith evaluator interface
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_ragas_eval(
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = "ragas",
    num_samples: int | None = None,
    force_refresh: bool = False,
) -> None:
    """Upload the dataset (idempotent) then run a LangSmith RAGAS experiment.

    Results appear in LangSmith:
      Datasets & Experiments → <dataset_name> → Experiments tab
    Each run shows all four metric scores per sample plus aggregate stats.
    """
    client = langsmith.Client()

    logger.info("Preparing dataset '%s'…", dataset_name)
    build_langsmith_dataset(client, dataset_name=dataset_name, num_samples=None, force_refresh=force_refresh)

    # Resolve which examples to evaluate (optionally capped at first N)
    all_examples = list(client.list_examples(dataset_id=client.read_dataset(dataset_name=dataset_name).id))
    if num_samples is not None:
        all_examples = all_examples[:num_samples]

    logger.info(
        "Starting RAGAS evaluation experiment (prefix='%s', samples=%d)…",
        experiment_prefix,
        len(all_examples),
    )

    results = evaluate(
        run_pipeline,
        data=all_examples,
        evaluators=[
            eval_faithfulness,
            eval_answer_relevancy,
            eval_context_precision,
            eval_context_recall,
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
        metadata={
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "eval_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        },
    )

    print("\nEvaluation complete.")
    print(f"  Dataset    : {dataset_name}")
    print(f"  Experiment : {experiment_prefix}")
    print(f"  LangSmith  : {results.experiment_name!r}")
    print("\nOpen LangSmith → Datasets & Experiments to view results.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Ariadne pipeline and publish to LangSmith."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help=f"LangSmith dataset name (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="ragas",
        help="Prefix for the LangSmith experiment name (default: ragas)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples — useful for quick smoke-tests (default: all 50)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Delete and recreate the LangSmith dataset before evaluating (fixes missing/stale reference data)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    run_ragas_eval(
        dataset_name=args.dataset,
        experiment_prefix=args.experiment_prefix,
        num_samples=args.num_samples,
        force_refresh=args.force_refresh,
    )

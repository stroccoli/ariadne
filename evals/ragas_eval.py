"""RAGAS evaluation runner for the Ariadne incident analysis pipeline.

Orchestrates the full evaluation loop: ensures the LangSmith dataset is
up-to-date, then runs all evaluators against the pipeline and publishes
results as a native LangSmith Experiment.

Evaluators (defined in evals/evaluators/):
    ragas_metrics  — faithfulness, answer_relevancy, context_precision, context_recall
    rubric_evals   — root_cause_quality, action_quality, final_score
    token_cost     — prompt_tokens, completion_tokens, estimated_cost_gemini_flash_usd
    ai_diagnosis   — ai_diagnosis (LLM meta-evaluator)

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
import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

# Suppress instructor's FutureWarning about google.generativeai at import time
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor")

import langsmith
from langsmith import evaluate

from evals.evaluators import (
    eval_action_quality,
    eval_ai_diagnosis,
    eval_answer_relevancy,
    eval_completion_tokens,
    eval_context_precision,
    eval_context_recall,
    eval_estimated_cost_gemini_flash,
    eval_faithfulness,
    eval_final_score,
    eval_prompt_tokens,
    eval_root_cause_quality,
)
from evals.pipeline import run_pipeline
from evals.ragas_dataset import DEFAULT_DATASET_NAME, build_langsmith_dataset

logger = logging.getLogger(__name__)


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
    Each run shows all evaluator scores per sample plus aggregate stats.
    """
    client = langsmith.Client()

    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")

    # Append provider to experiment prefix so each provider gets its own
    # experiment in LangSmith (e.g. ragas_openai, ragas_ollama, ragas_gemini).
    full_prefix = f"{experiment_prefix}_{llm_provider}"

    logger.info("Preparing dataset '%s'…", dataset_name)
    build_langsmith_dataset(client, dataset_name=dataset_name, num_samples=None, force_refresh=force_refresh)

    # Resolve which examples to evaluate (optionally capped at first N)
    all_examples = list(client.list_examples(dataset_id=client.read_dataset(dataset_name=dataset_name).id))
    if num_samples is not None:
        all_examples = all_examples[:num_samples]

    logger.info(
        "Starting RAGAS evaluation experiment (prefix='%s', samples=%d)…",
        full_prefix,
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
            eval_root_cause_quality,
            eval_action_quality,
            eval_final_score,
            eval_prompt_tokens,
            eval_completion_tokens,
            eval_estimated_cost_gemini_flash,
            eval_ai_diagnosis,
        ],
        experiment_prefix=full_prefix,
        max_concurrency=1,
        metadata={
            "llm_provider": llm_provider,
            "embedding_provider": embedding_provider,
            "eval_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "metrics": [
                "faithfulness", "answer_relevancy", "context_precision", "context_recall",
                "root_cause_quality", "action_quality", "final_score",
                "prompt_tokens", "completion_tokens", "estimated_cost_gemini_flash_usd",
                "ai_diagnosis",
            ],
        },
    )

    print("\nEvaluation complete.")
    print(f"  Dataset    : {dataset_name}")
    print(f"  Experiment : {full_prefix}")
    print(f"  Provider   : llm={llm_provider}, embedding={embedding_provider}")
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
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "Set LLM_PROVIDER and EMBEDDING_PROVIDER for this run "
            "(e.g. openai, ollama, gemini). Overrides env vars."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        os.environ["EMBEDDING_PROVIDER"] = args.provider
        from ariadne.core.config import reset_provider_caches
        reset_provider_caches()

    run_ragas_eval(
        dataset_name=args.dataset,
        experiment_prefix=args.experiment_prefix,
        num_samples=args.num_samples,
        force_refresh=args.force_refresh,
    )

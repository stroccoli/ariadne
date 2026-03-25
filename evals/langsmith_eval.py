"""LangSmith evaluation suite for Ariadne.

Metrics
-------
* recall@k    — retrieval quality: fraction of expected relevant docs found in top-k retrieved docs.
* faithfulness — generation quality: LLM judge scores whether the root_cause is grounded in the
                 retrieved context (0.0–1.0).
* correctness  — end-to-end quality: three sub-scores via rubric_scoring:
                   - type_match:          1.0 if incident_type is exact match
                   - root_cause_quality:  rubric coverage score (0.0–1.0)
                   - action_quality:      rubric coverage + count score (0.0–1.0)

Usage
-----
    # Upload dataset to LangSmith (idempotent):
    python -m evals.langsmith_eval --upload-only

    # Run full evaluation:
    python -m evals.langsmith_eval

    # Choose experiment prefix and dataset name:
    python -m evals.langsmith_eval --dataset ariadne-incidents-dev --experiment-prefix ariadne
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Any

from ariadne.core.config import get_langsmith_api_key, get_langsmith_project, is_langsmith_enabled
from ariadne.core.graph import run_graph
from evals.rubric_scoring import evaluate_action_quality, evaluate_root_cause_quality
from evals.sample_library import get_sample_by_id


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dev set: 10 samples (2 per incident type) + ground-truth relevant doc titles
# ---------------------------------------------------------------------------

DEV_SAMPLE_IDS: tuple[str, ...] = (
    "timeout_checkout_pricing",
    "timeout_media_renderer",
    "dependency_auth_gateway",
    "dependency_flags_web",
    "database_pool_checkout",
    "database_deadlock_inventory",
    "memory_oom_report_worker",
    "memory_leak_scheduler",
    "unknown_gateway_tls_vs_auth",
    "unknown_worker_dependency_vs_disk",
)

# For each sample: the doc titles (exactly as stored in incident_knowledge.json) that a
# well-functioning retriever should surface in the top-k results.
_GROUND_TRUTH: dict[str, list[str]] = {
    "timeout_checkout_pricing": [
        "Checkout timeouts caused by a slow dependency",
        "Retry storm feedback loop",
        "Gateway 504 caused by downstream saturation",
    ],
    "timeout_media_renderer": [
        "Checkout timeouts caused by a slow dependency",
        "Thread pool starvation and latency",
        "Retry storm feedback loop",
    ],
    "dependency_auth_gateway": [
        "Dependency outage with connection refusal",
        "Circuit breaker opening after dependency instability",
    ],
    "dependency_flags_web": [
        "Service discovery and DNS failures",
        "Dependency outage with connection refusal",
    ],
    "database_pool_checkout": [
        "Database pool exhaustion pattern",
        "Storage saturation driving database latency",
    ],
    "database_deadlock_inventory": [
        "Database deadlocks under concurrent writes",
        "Database pool exhaustion pattern",
    ],
    "memory_oom_report_worker": [
        "Kubernetes OOMKilled restart sequence",
        "Memory pressure before restarts",
    ],
    "memory_leak_scheduler": [
        "Long-lived process memory leak",
        "Memory pressure before restarts",
    ],
    "unknown_gateway_tls_vs_auth": [
        "TLS or token failures caused by clock skew",
        "Expired certificate on ingress or edge",
    ],
    "unknown_worker_dependency_vs_disk": [
        "Application host out of disk space",
        "Mixed partner timeout and database contention signals",
    ],
}

# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

DEFAULT_DATASET_NAME = "ariadne-incidents-dev"
DEFAULT_EXPERIMENT_PREFIX = "ariadne"
DEFAULT_MODE = "detailed"


def _build_examples() -> list[dict[str, Any]]:
    """Build the 10 dataset examples with inputs and reference outputs."""
    examples = []
    for sample_id in DEV_SAMPLE_IDS:
        sample = get_sample_by_id(sample_id)
        examples.append(
            {
                "inputs": {
                    "logs": sample.logs,
                    "mode": DEFAULT_MODE,
                },
                "reference_outputs": {
                    "sample_id": sample_id,
                    "expected_incident_type": sample.expected_incident_type,
                    "relevant_doc_titles": _GROUND_TRUTH[sample_id],
                },
            }
        )
    return examples


def upload_dataset(dataset_name: str = DEFAULT_DATASET_NAME) -> Any:
    """Create (or return existing) LangSmith dataset with the dev-set examples.

    Idempotent: if the dataset already exists it is returned unchanged.
    """
    from langsmith import Client

    client = Client(api_key=get_langsmith_api_key())

    if client.has_dataset(dataset_name=dataset_name):
        logger.info("Dataset '%s' already exists — skipping upload", dataset_name)
        return client.read_dataset(dataset_name=dataset_name)

    logger.info("Creating dataset '%s' with %d examples", dataset_name, len(DEV_SAMPLE_IDS))
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Ariadne incident analysis dev set — 10 curated samples (2 per incident type) "
            "with ground-truth relevant doc titles for recall@k evaluation."
        ),
    )

    examples = _build_examples()
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["reference_outputs"] for e in examples],
        dataset_id=dataset.id,
    )

    logger.info("Dataset '%s' created successfully (id=%s)", dataset_name, dataset.id)
    return dataset


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


def run_pipeline(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the Ariadne pipeline and return a flat dict with all eval-relevant fields."""
    logs: str = inputs["logs"]
    mode: str = inputs.get("mode", DEFAULT_MODE)

    state = run_graph(logs, mode=mode)

    if state.final_output is None:
        return {
            "incident_type": "unknown",
            "root_cause": "",
            "actions": [],
            "confidence": 0.0,
            "retrieved_doc_titles": [],
            "retrieved_context": "",
        }

    return {
        "incident_type": state.final_output.incident_type,
        "root_cause": state.final_output.root_cause,
        "actions": state.final_output.recommended_actions,
        "confidence": state.final_output.confidence,
        "retrieved_doc_titles": state.retrieved_doc_titles,
        "retrieved_context": "\n\n".join(state.context),
    }


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def recall_at_k(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Fraction of expected relevant docs found in the retrieved top-k set."""
    retrieved: list[str] = outputs.get("retrieved_doc_titles", [])
    expected: list[str] = reference_outputs.get("relevant_doc_titles", [])

    if not expected:
        return {"key": "recall@k", "score": 1.0, "comment": "no ground truth defined"}

    # Normalise for loose matching: lowercase, collapse whitespace
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    retrieved_norm = {_norm(t) for t in retrieved}
    hits = sum(1 for title in expected if _norm(title) in retrieved_norm)
    score = hits / len(expected)

    return {
        "key": "recall@k",
        "score": round(score, 3),
        "comment": f"hits={hits}/{len(expected)} | retrieved={retrieved}",
    }


def faithfulness(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """LLM judge: is the root_cause grounded in the retrieved context?

    Prompt asks for a score between 0 and 10. Normalised to 0.0–1.0.
    """
    root_cause: str = outputs.get("root_cause", "")
    context: str = outputs.get("retrieved_context", "")

    if not root_cause:
        return {"key": "faithfulness", "score": 0.0, "comment": "empty root_cause"}

    if not context:
        return {
            "key": "faithfulness",
            "score": 0.0,
            "comment": "no retrieved context available",
        }

    prompt = (
        "You are an objective evaluator. Score how well the root cause explanation is "
        "supported by the retrieved context documents.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Root cause explanation:\n{root_cause}\n\n"
        "Rules:\n"
        "- Score 10: every key claim in the root cause is directly supported by the context.\n"
        "- Score 5: some claims are supported, others are inferred or missing from context.\n"
        "- Score 0: the root cause contradicts or ignores the context entirely.\n\n"
        "Respond with ONLY a JSON object in this exact format (no markdown, no extra text):\n"
        '{"score": <integer 0-10>, "reason": "<one sentence>"}'
    )

    from ariadne.core.config import get_llm_client

    try:
        response = get_llm_client().generate(prompt, json_output=True)
        data = json.loads(response.text)
        raw_score = float(data.get("score", 0))
        normalised = round(max(0.0, min(10.0, raw_score)) / 10.0, 2)
        reason = data.get("reason", "")
    except Exception as exc:
        logger.warning("faithfulness judge failed: %s", exc)
        return {"key": "faithfulness", "score": None, "comment": f"judge error: {exc}"}

    return {"key": "faithfulness", "score": normalised, "comment": reason}


def correctness(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Three rubric-based sub-scores using the existing sample_library rubrics.

    Returns a list of dicts because LangSmith evaluators that return a list emit one
    feedback entry per element.
    """
    sample_id: str = reference_outputs.get("sample_id", "")

    try:
        sample = get_sample_by_id(sample_id)
    except ValueError:
        return [
            {"key": "type_match", "score": None, "comment": f"unknown sample_id: {sample_id}"},
            {"key": "root_cause_quality", "score": None, "comment": f"unknown sample_id: {sample_id}"},
            {"key": "action_quality", "score": None, "comment": f"unknown sample_id: {sample_id}"},
        ]

    predicted_type: str = outputs.get("incident_type", "")
    root_cause: str = outputs.get("root_cause", "")
    actions: list[str] = outputs.get("actions", [])
    expected_type: str = reference_outputs.get("expected_incident_type", "")

    type_match_score = 1.0 if predicted_type == expected_type else 0.0

    rc_result = evaluate_root_cause_quality(sample, root_cause)
    action_result = evaluate_action_quality(sample, actions)

    return [
        {
            "key": "type_match",
            "score": type_match_score,
            "comment": f"predicted={predicted_type} expected={expected_type}",
        },
        {
            "key": "root_cause_quality",
            "score": rc_result.score,
            "comment": (
                f"matched={list(rc_result.matched_concepts)} "
                f"missed={list(rc_result.missed_concepts)}"
            ),
        },
        {
            "key": "action_quality",
            "score": action_result.score,
            "comment": (
                f"actions={action_result.action_count} "
                f"matched={list(action_result.matched_concepts)} "
                f"missed={list(action_result.missed_concepts)}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_langsmith_eval(
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
) -> Any:
    """Run the full evaluation suite and return the LangSmith ExperimentResults."""
    from langsmith.evaluation import evaluate

    results = evaluate(
        run_pipeline,
        data=dataset_name,
        evaluators=[recall_at_k, faithfulness, correctness],
        experiment_prefix=experiment_prefix,
        metadata={
            "description": "recall@k, faithfulness, correctness (type_match + rubric quality)",
            "dataset": dataset_name,
        },
        max_concurrency=1,
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluations for Ariadne (recall@k, faithfulness, correctness)."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help="LangSmith dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-prefix",
        default=DEFAULT_EXPERIMENT_PREFIX,
        help="Prefix for the experiment name in LangSmith (default: %(default)s)",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload the dataset to LangSmith; do not run evaluations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not is_langsmith_enabled():
        print(
            "ERROR: LangSmith is not enabled. Set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY (or LANGSMITH_* equivalents).",
            file=sys.stderr,
        )
        sys.exit(1)

    args = _parse_args()

    print(f"LangSmith project: {get_langsmith_project()}")

    upload_dataset(dataset_name=args.dataset)

    if args.upload_only:
        print(f"Dataset '{args.dataset}' is ready in LangSmith. Skipping evaluation run.")
        sys.exit(0)

    print(f"Running evaluation on dataset '{args.dataset}' ...")
    results = run_langsmith_eval(
        dataset_name=args.dataset,
        experiment_prefix=args.experiment_prefix,
    )
    print("Evaluation complete. Results are visible in LangSmith.")
    print(results)

"""Build and upload the LangSmith evaluation dataset for RAGAS.

Each sample is uploaded as a LangSmith Example with:
  inputs  → {"logs": str, "mode": str}
  outputs → {"reference": str}   ← auto-generated from rubric required_concepts

The reference string is used by context_precision and context_recall as ground
truth for what the answer should cover.

Usage (idempotent — safe to re-run):
    python evals/ragas_dataset.py
    python evals/ragas_dataset.py --dataset my-custom-name --num-samples 10
"""
if __name__ == "__main__":
    import pathlib
    import sys

    _root = str(pathlib.Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

import langsmith

from evals.sample_library import DEFAULT_AB_TEST_SAMPLES, IncidentSample

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "ariadne-incidents-dev"


# ---------------------------------------------------------------------------
# Ground truth generation
# ---------------------------------------------------------------------------

def _reference_from_rubric(sample: IncidentSample) -> str:
    """Synthesise a plain-text reference answer from the sample rubric.

    Concatenates the required_concept labels from both the root cause and
    action rubrics into a short prose description. This gives context_recall
    and context_precision a meaningful reference without manual annotation.
    """
    root_labels = [c.label for c in sample.root_cause_rubric.required_concepts]
    action_labels = [c.label for c in sample.action_rubric.required_concepts]

    parts: list[str] = [
        f"Incident type: {sample.expected_incident_type}.",
        f"Root cause involves: {', '.join(root_labels)}.",
        f"Recommended actions should cover: {', '.join(action_labels)}.",
    ]

    if sample.root_cause_rubric.require_uncertainty:
        parts.append("Answer should acknowledge uncertainty due to ambiguous signals.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

def build_langsmith_dataset(
    client: langsmith.Client,
    dataset_name: str = DEFAULT_DATASET_NAME,
    num_samples: int | None = None,
    force_refresh: bool = False,
) -> str:
    """Create (or reuse) a LangSmith dataset and upsert examples.

    Returns the dataset name — ready to pass to ``langsmith.evaluate()``.

    If ``force_refresh=True``, deletes the existing dataset and recreates it
    from scratch — useful when reference data was missing or samples changed.
    Otherwise the operation is idempotent: only new samples are added, and
    existing ones whose reference field is empty are patched in-place.
    """
    samples = DEFAULT_AB_TEST_SAMPLES
    if num_samples is not None:
        samples = samples[:num_samples]

    existing = [ds for ds in client.list_datasets(dataset_name=dataset_name)]

    if existing and force_refresh:
        client.delete_dataset(dataset_id=existing[0].id)
        logger.info("Deleted existing dataset '%s' for force-refresh", dataset_name)
        existing = []

    if existing:
        dataset = existing[0]
        logger.info("Reusing existing LangSmith dataset '%s' (id=%s)", dataset_name, dataset.id)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Ariadne incident analysis samples for RAGAS evaluation. "
                "Each example contains raw logs and a reference answer derived from curated rubrics."
            ),
        )
        logger.info("Created LangSmith dataset '%s' (id=%s)", dataset_name, dataset.id)

    # Build a map of sample_id → existing Example for efficient lookup
    existing_examples: dict[str, object] = {
        ex.metadata.get("sample_id"): ex
        for ex in client.list_examples(dataset_id=dataset.id)
        if ex.metadata and ex.metadata.get("sample_id")
    }

    new_samples = [s for s in samples if s.sample_id not in existing_examples]
    patch_samples = [
        s for s in samples
        if s.sample_id in existing_examples
        and not (existing_examples[s.sample_id].outputs or {}).get("reference")
    ]

    if new_samples:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"logs": s.logs, "mode": "detailed"} for s in new_samples],
            outputs=[{"reference": _reference_from_rubric(s)} for s in new_samples],
            metadata=[
                {
                    "sample_id": s.sample_id,
                    "description": s.description,
                    "expected_incident_type": s.expected_incident_type,
                }
                for s in new_samples
            ],
        )
        logger.info("Uploaded %d new examples to '%s'", len(new_samples), dataset_name)

    if patch_samples:
        for s in patch_samples:
            ex = existing_examples[s.sample_id]
            client.update_example(
                example_id=ex.id,
                outputs={"reference": _reference_from_rubric(s)},
            )
        logger.info("Patched reference field on %d existing examples in '%s'", len(patch_samples), dataset_name)

    if not new_samples and not patch_samples:
        logger.info("All %d examples already up-to-date in '%s'", len(samples), dataset_name)

    return dataset_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the Ariadne RAGAS evaluation dataset to LangSmith.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="LangSmith dataset name")
    parser.add_argument("--num-samples", type=int, default=None, help="Upload only the first N samples (default: all 50)")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Delete and recreate the dataset from scratch (use when samples or references changed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    ls_client = langsmith.Client()
    name = build_langsmith_dataset(
        ls_client,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        force_refresh=args.force_refresh,
    )
    print(f"Dataset ready: {name}")

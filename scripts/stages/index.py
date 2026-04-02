"""DVC stage: index chunks into vector store."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ariadne.core.config import reset_provider_caches  # noqa: E402
from ariadne.core.ingestion import IngestionPipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index chunks into the vector store.")
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "Override EMBEDDING_PROVIDER for this run "
            "(e.g. openai, ollama, gemini). Also sets LLM_PROVIDER."
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help=(
            "Skip chunks whose content_hash already matches the indexed version. "
            "Saves embedding API quota when re-indexing an unchanged corpus."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    if args.provider:
        os.environ["EMBEDDING_PROVIDER"] = args.provider
        os.environ["LLM_PROVIDER"] = args.provider
        reset_provider_caches()

    params = yaml.safe_load((_REPO_ROOT / "params.yaml").read_text())["ingest"]
    preset_name = params.get("chunk_preset", "medium")
    batch_size = params.get("embedding_batch_size", 32)

    pipeline = IngestionPipeline(_REPO_ROOT)

    # Write provider-specific index metrics when --provider is used
    if args.provider:
        pipeline.index_metrics_path = pipeline.data_dir / f"index_metrics_{args.provider}.json"

    chunk_path = pipeline.data_dir / f"chunks_{preset_name}.json"

    if not chunk_path.exists():
        print(
            f"ERROR: No chunks found at {chunk_path}. Run chunk stage first.",
            file=sys.stderr,
        )
        sys.exit(1)

    chunks = pipeline._load_docs(chunk_path)
    # When --incremental is passed, skip chunks whose content hasn't changed
    # (content_hash comparison).  Otherwise do a full upsert so updated payloads
    # (e.g. tags added by preprocessing) are written to Qdrant.  DVC guarantees
    # this stage only re-runs when chunk data changes.
    incremental = args.incremental
    embedding_model_check = params.get("embedding_model_check", True)
    pipeline.index(
        chunks,
        embedding_batch_size=batch_size,
        incremental=incremental,
        embedding_model_check=embedding_model_check,
    )

    # Warn loudly if partial failure occurred
    import json as _json
    if pipeline.index_metrics_path.exists():
        metrics = _json.loads(pipeline.index_metrics_path.read_text(encoding="utf-8"))
        if metrics.get("partial_failure"):
            print(
                f"\n⚠  WARNING: Partial indexing failure detected! "
                f"{metrics.get('upsert_error_count', 0)} batch(es) failed. "
                f"See {pipeline.index_metrics_path} for rollback IDs.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()

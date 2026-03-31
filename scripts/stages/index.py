"""DVC stage: index chunks into vector store."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ariadne.core.ingestion import IngestionPipeline  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    params = yaml.safe_load((_REPO_ROOT / "params.yaml").read_text())["ingest"]
    preset_name = params.get("chunk_preset", "medium")
    batch_size = params.get("embedding_batch_size", 32)

    pipeline = IngestionPipeline(_REPO_ROOT)
    chunk_path = pipeline.data_dir / f"chunks_{preset_name}.json"

    if not chunk_path.exists():
        print(
            f"ERROR: No chunks found at {chunk_path}. Run chunk stage first.",
            file=sys.stderr,
        )
        sys.exit(1)

    chunks = pipeline._load_docs(chunk_path)
    # DVC guarantees this stage only re-runs when chunk data changes, so always
    # do a full upsert (incremental=False) to ensure updated payloads (e.g. tags
    # added by preprocessing) are written to Qdrant.
    embedding_model_check = params.get("embedding_model_check", True)
    pipeline.index(
        chunks,
        embedding_batch_size=batch_size,
        incremental=False,
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

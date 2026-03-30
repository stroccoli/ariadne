"""DVC stage: chunk preprocessed documents."""
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

    pipeline = IngestionPipeline(_REPO_ROOT)
    clean_checkpoint = pipeline.checkpoints["clean"]

    if not clean_checkpoint.exists():
        print("ERROR: No clean_docs.json found. Run preprocess first.", file=sys.stderr)
        sys.exit(1)

    clean_docs = pipeline._load_docs(clean_checkpoint)
    pipeline.chunk(clean_docs, preset_name=preset_name)


if __name__ == "__main__":
    main()

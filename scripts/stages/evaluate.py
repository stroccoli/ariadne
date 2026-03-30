"""DVC stage: evaluate retrieval quality."""
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
    chunk_path = pipeline.data_dir / f"chunks_{preset_name}.json"

    if not chunk_path.exists():
        print(
            f"ERROR: No chunks found at {chunk_path}. Run index stage first.",
            file=sys.stderr,
        )
        sys.exit(1)

    chunks = pipeline._load_docs(chunk_path)
    report = pipeline.evaluate(chunks)
    print(report.summary())


if __name__ == "__main__":
    main()

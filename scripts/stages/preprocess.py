"""DVC stage: preprocess raw documents."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ariadne.core.ingestion import IngestionPipeline  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = IngestionPipeline(_REPO_ROOT)

    all_raw_docs = []
    for key in ("postmortems", "github"):
        checkpoint = pipeline.checkpoints[key]
        if checkpoint.exists():
            all_raw_docs.extend(pipeline._load_docs(checkpoint))

    if not all_raw_docs:
        print(
            "ERROR: No raw data found. Run collect_postmortems or collect_github first.",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline.preprocess(all_raw_docs)


if __name__ == "__main__":
    main()

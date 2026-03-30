"""DVC stage: collect postmortems."""
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

    pipeline = IngestionPipeline(_REPO_ROOT)
    pipeline.collect_postmortems(
        max_entries=params.get("max_postmortems", 800),
    )


if __name__ == "__main__":
    main()

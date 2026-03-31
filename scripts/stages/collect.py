"""DVC stage: collect all raw documents (postmortems + optionally GitHub issues)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ariadne.core.ingestion import IngestionPipeline  # noqa: E402

logger = logging.getLogger(__name__)


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

    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        checkpoint_path = pipeline.data_dir / "raw_github_partial.json"
        pipeline.collect_github(
            github_token=github_token,
            max_per_repo=params.get("max_per_repo", 200),
            checkpoint_path=checkpoint_path,
        )
        # Clean up partial checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    else:
        logger.info("GITHUB_TOKEN not set — skipping GitHub issues collection.")


if __name__ == "__main__":
    main()

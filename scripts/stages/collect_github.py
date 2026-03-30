"""DVC stage: collect GitHub issues."""
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print(
            "ERROR: GITHUB_TOKEN environment variable is required.\n"
            "  export GITHUB_TOKEN=ghp_...\n"
            "  dvc repro collect_github",
            file=sys.stderr,
        )
        sys.exit(1)

    params = yaml.safe_load((_REPO_ROOT / "params.yaml").read_text())["ingest"]

    pipeline = IngestionPipeline(_REPO_ROOT)
    pipeline.collect_github(
        github_token=github_token,
        max_per_repo=params.get("max_per_repo", 200),
    )


if __name__ == "__main__":
    main()

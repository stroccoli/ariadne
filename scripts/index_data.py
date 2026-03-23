from __future__ import annotations

import argparse
import logging
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


logger = logging.getLogger(__name__)
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "incident_knowledge.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index incident knowledge into the configured vector store.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the JSON dataset to index.",
    )
    return parser
def main() -> None:
    from ariadne.core.config import get_vector_store
    from ariadne.core.logging_config import configure_logging
    from ariadne.core.retrieval.dataset import load_documents

    configure_logging()
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    documents = load_documents(dataset_path)

    logger.info("Loaded %d documents from %s", len(documents), dataset_path)
    get_vector_store().index(documents)
    logger.info("Indexing completed")


if __name__ == "__main__":
    main()

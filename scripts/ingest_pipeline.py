from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ariadne.core.config import get_vector_store  # noqa: E402
from ariadne.core.retrieval.chunking import CHUNK_PRESETS, chunk_documents  # noqa: E402
from ariadne.core.retrieval.document import IngestionDocument  # noqa: E402
from ariadne.core.retrieval.preprocessing import preprocess_documents  # noqa: E402
from ariadne.core.retrieval.vector_stores.qdrant_store import QdrantVectorStore  # noqa: E402
from evals.retrieval_eval import (  # noqa: E402
    EvalReport,
    RetrievalEvaluator,
    build_eval_set_from_titles,
)

logger = logging.getLogger(__name__)


DATA_DIR = _REPO_ROOT / "data" / "pipeline"
CHECKPOINT_GITHUB = DATA_DIR / "raw_github.json"
CHECKPOINT_POSTMORTEMS = DATA_DIR / "raw_postmortems.json"
CHECKPOINT_CLEAN = DATA_DIR / "clean_docs.json"
CHECKPOINT_CHUNKS = DATA_DIR / "chunks.json"
EVAL_REPORT_PATH = DATA_DIR / "eval_report.json"

def _save_docs(docs: list[IngestionDocument], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [doc.model_dump(mode="json") for doc in docs]
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Checkpoint saved: %s (%d docs)", path.name, len(docs))


def _load_docs(path: Path) -> list[IngestionDocument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [IngestionDocument.model_validate(item) for item in raw]


def stage_collect_github(github_token: str, max_per_repo: int) -> list[IngestionDocument]:
    if CHECKPOINT_GITHUB.exists():
        logger.info("SKIP collect_github (checkpoint found: %s)", CHECKPOINT_GITHUB)
        return _load_docs(CHECKPOINT_GITHUB)

    logger.info("START collect_github (max_per_repo=%d)...", max_per_repo)
    from scripts.collect.github_issues import GitHubIssuesCollector
    collector = GitHubIssuesCollector(token=github_token)
    docs = collector.collect(max_per_repo=max_per_repo)
    _save_docs(docs, CHECKPOINT_GITHUB)
    return docs


def stage_collect_postmortems(
    github_token: str | None = None,
    max_entries: int = 800,
) -> list[IngestionDocument]:
    if CHECKPOINT_POSTMORTEMS.exists():
        logger.info("SKIP collect_postmortems (checkpoint found: %s)", CHECKPOINT_POSTMORTEMS)
        return _load_docs(CHECKPOINT_POSTMORTEMS)

    logger.info("START collect_postmortems (max=%d)...", max_entries)
    from scripts.collect.postmortems import PostmortemsCollector
    collector = PostmortemsCollector(github_token=github_token)
    docs = collector.collect(max_entries=max_entries)
    _save_docs(docs, CHECKPOINT_POSTMORTEMS)
    return docs


def stage_preprocess(raw_docs: list[IngestionDocument]) -> list[IngestionDocument]:
    if CHECKPOINT_CLEAN.exists():
        logger.info("SKIP preprocess (checkpoint found: %s)", CHECKPOINT_CLEAN)
        return _load_docs(CHECKPOINT_CLEAN)

    logger.info("START preprocess (%d raw docs)...", len(raw_docs))
    clean_docs = preprocess_documents(raw_docs, verbose=True)
    _save_docs(clean_docs, CHECKPOINT_CLEAN)
    return clean_docs


def stage_chunk(
    clean_docs: list[IngestionDocument],
    preset_name: str = "medium",
) -> list[IngestionDocument]:
    checkpoint = DATA_DIR / f"chunks_{preset_name}.json"
    if checkpoint.exists():
        logger.info("SKIP chunk_%s (checkpoint found: %s)", preset_name, checkpoint)
        return _load_docs(checkpoint)

    preset = CHUNK_PRESETS[preset_name]
    logger.info(
        "START chunk (preset=%s, size=%d, overlap=%d)...",
        preset_name,
        preset["chunk_size"],
        preset["chunk_overlap"],
    )
    chunks = chunk_documents(clean_docs, **preset)
    _save_docs(chunks, checkpoint)
    return chunks


def stage_index(
    chunks: list[IngestionDocument],
    vector_store: QdrantVectorStore,
    *,
    embedding_batch_size: int = 32,
) -> None:
    logger.info("START index (%d chunks)...", len(chunks))
    n_batches = (len(chunks) + embedding_batch_size - 1) // embedding_batch_size

    with tqdm(total=n_batches, desc="Embedding+Indexing", unit="batch") as pbar:
        for i in range(0, len(chunks), embedding_batch_size):
            batch = chunks[i : i + embedding_batch_size]
            try:
                vector_store.index_documents(batch, embedding_batch_size=len(batch))
            except Exception as exc:
                logger.warning("Batch %d failed: %s. Retrying once...", i // embedding_batch_size, exc)
                time.sleep(2)
                vector_store.index_documents(batch, embedding_batch_size=len(batch))
            pbar.update(1)

    logger.info("Indexing complete.")


def stage_eval(vector_store: QdrantVectorStore, chunks: list[IngestionDocument]) -> EvalReport:
    logger.info("START eval...")

    seen_parents: set[str] = set()
    title_pairs: list[tuple[str, str]] = []
    for chunk in chunks:
        parent_id = chunk.parent_id or chunk.id
        if parent_id not in seen_parents and chunk.title:
            seen_parents.add(parent_id)
            title_pairs.append((chunk.id, chunk.title))

    eval_queries = build_eval_set_from_titles(title_pairs, max_queries=50)

    if not eval_queries:
        logger.warning("No eval queries could be built (no titles found)")
        return EvalReport(recall_at_k={}, mrr=0.0, n_queries=0)

    evaluator = RetrievalEvaluator(
        search_fn=vector_store.search,
        id_extractor=lambda text: text[:50],
    )
    report = evaluator.evaluate(eval_queries, k_values=[1, 3, 5])
    evaluator.save_report(report, EVAL_REPORT_PATH)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ariadne RAG ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with GitHub token
  python scripts/ingest_pipeline.py --github-token=ghp_... --mode=full

  # Post-mortems only (no token needed)
  python scripts/ingest_pipeline.py --mode=postmortems-only

  # Re-index from existing checkpoints
  python scripts/ingest_pipeline.py --mode=index

  # Evaluate the current retriever
  python scripts/ingest_pipeline.py --mode=eval

  # Force re-run (ignore checkpoints)
  python scripts/ingest_pipeline.py --mode=full --force
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "postmortems-only", "index", "eval"],
        default="postmortems-only",
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub Personal Access Token (required for mode 'full')",
    )
    parser.add_argument(
        "--max-per-repo",
        type=int,
        default=200,
        help="Max issues to download per GitHub repo",
    )
    parser.add_argument(
        "--max-postmortems",
        type=int,
        default=800,
        help="Max post-mortems to collect",
    )
    parser.add_argument(
        "--chunk-preset",
        choices=list(CHUNK_PRESETS.keys()),
        default="medium",
        help="Chunking preset to use",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Texts per embedding batch",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoints and re-run all stages",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.force:
        logger.info("--force: clearing all checkpoints")
        for checkpoint in [
            CHECKPOINT_GITHUB,
            CHECKPOINT_POSTMORTEMS,
            CHECKPOINT_CLEAN,
            DATA_DIR / f"chunks_{args.chunk_preset}.json",
        ]:
            if checkpoint.exists():
                checkpoint.unlink()
                logger.info("  Removed %s", checkpoint.name)

    start_time = time.time()
    all_raw_docs: list[IngestionDocument] = []


    if args.mode in ("full", "postmortems-only"):
        pm_docs = stage_collect_postmortems(
            github_token=args.github_token,
            max_entries=args.max_postmortems,
        )
        all_raw_docs.extend(pm_docs)

        if args.mode == "full":
            if not args.github_token:
                logger.error("--github-token is required for --mode=full")
                sys.exit(1)
            gh_docs = stage_collect_github(args.github_token, args.max_per_repo)
            all_raw_docs.extend(gh_docs)

        logger.info("Total raw docs collected: %d", len(all_raw_docs))

    elif args.mode in ("index", "eval"):
        for checkpoint in [CHECKPOINT_GITHUB, CHECKPOINT_POSTMORTEMS]:
            if checkpoint.exists():
                all_raw_docs.extend(_load_docs(checkpoint))
        if not all_raw_docs:
            logger.error("No checkpoints found. Run with --mode=full or --mode=postmortems-only first.")
            sys.exit(1)

    if args.mode != "eval":
        clean_docs = stage_preprocess(all_raw_docs)

        chunks = stage_chunk(clean_docs, preset_name=args.chunk_preset)

        vector_store = get_vector_store()
        if not isinstance(vector_store, QdrantVectorStore):
            logger.error("Vector store is not QdrantVectorStore. Set VECTOR_STORE=qdrant in .env")
            sys.exit(1)

        stage_index(chunks, vector_store, embedding_batch_size=args.embedding_batch_size)
    else:
        chunk_checkpoint = DATA_DIR / f"chunks_{args.chunk_preset}.json"
        if not chunk_checkpoint.exists():
            logger.error("No chunk checkpoint found for preset '%s'. Run indexing first.", args.chunk_preset)
            sys.exit(1)
        chunks = _load_docs(chunk_checkpoint)
        vector_store = get_vector_store()
        if not isinstance(vector_store, QdrantVectorStore):
            logger.error("Vector store is not QdrantVectorStore.")
            sys.exit(1)


    report = stage_eval(vector_store, chunks)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(report.summary())
    print(f"Eval report saved to: {EVAL_REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

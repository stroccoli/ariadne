from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ariadne.core.config import get_vector_store
from ariadne.core.retrieval.chunking import CHUNK_PRESETS, chunk_documents
from ariadne.core.retrieval.document import IngestionDocument
from ariadne.core.retrieval.preprocessing import PreprocessReport, preprocess_documents
from ariadne.core.retrieval.vector_stores.qdrant_store import QdrantVectorStore
from evals.retrieval_eval import (
    EvalReport,
    RetrievalEvaluator,
    build_eval_set_from_titles,
    load_curated_eval_set,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Manages the document ingestion pipeline for Ariadne RAG system."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.data_dir = repo_root / "data" / "pipeline"
        self.checkpoints = {
            "github": self.data_dir / "raw_github.json",
            "postmortems": self.data_dir / "raw_postmortems.json",
            "clean": self.data_dir / "clean_docs.json",
            "chunks": self.data_dir / "chunks.json",
        }
        self.eval_report_path = self.data_dir / "eval_report.json"
        self.preprocess_report_path = self.data_dir / "preprocess_report.json"
        self.curated_eval_path = repo_root / "data" / "eval_queries.json"
        self.vector_store: Optional[QdrantVectorStore] = None

    def _save_docs(self, docs: list[IngestionDocument], path: Path) -> None:
        """Save documents to a checkpoint file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [doc.model_dump(mode="json") for doc in docs]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Checkpoint saved: %s (%d docs)", path.name, len(docs))

    def _load_docs(self, path: Path) -> list[IngestionDocument]:
        """Load documents from a checkpoint file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [IngestionDocument.model_validate(item) for item in raw]

    # ------------------------------------------------------------------
    # Stage methods (stateless — always execute, DVC handles caching)
    # ------------------------------------------------------------------

    def collect_github(self, github_token: str, max_per_repo: int = 200) -> list[IngestionDocument]:
        """Collect GitHub issues documents."""
        logger.info("START collect_github (max_per_repo=%d)...", max_per_repo)
        from scripts.collect.github_issues import GitHubIssuesCollector
        collector = GitHubIssuesCollector(token=github_token)
        docs = collector.collect(max_per_repo=max_per_repo)
        self._save_docs(docs, self.checkpoints["github"])
        return docs

    def collect_postmortems(
        self,
        github_token: Optional[str] = None,
        max_entries: int = 800,
    ) -> list[IngestionDocument]:
        """Collect postmortems documents."""
        logger.info("START collect_postmortems (max=%d)...", max_entries)
        from scripts.collect.postmortems import PostmortemsCollector
        collector = PostmortemsCollector(github_token=github_token)
        docs = collector.collect(max_entries=max_entries)
        self._save_docs(docs, self.checkpoints["postmortems"])
        return docs

    def preprocess(self, raw_docs: list[IngestionDocument]) -> list[IngestionDocument]:
        """Preprocess raw documents and save quality report."""
        logger.info("START preprocess (%d raw docs)...", len(raw_docs))
        clean_docs, report = preprocess_documents(raw_docs, verbose=True)
        self._save_docs(clean_docs, self.checkpoints["clean"])
        report.save(self.preprocess_report_path)
        return clean_docs

    def chunk(
        self,
        clean_docs: list[IngestionDocument],
        preset_name: str = "medium",
    ) -> list[IngestionDocument]:
        """Chunk documents using specified preset."""
        preset = CHUNK_PRESETS[preset_name]
        logger.info(
            "START chunk (preset=%s, size=%d, overlap=%d)...",
            preset_name,
            preset["chunk_size"],
            preset["chunk_overlap"],
        )
        chunks = chunk_documents(clean_docs, **preset)
        checkpoint = self.data_dir / f"chunks_{preset_name}.json"
        self._save_docs(chunks, checkpoint)
        return chunks

    def index(
        self,
        chunks: list[IngestionDocument],
        embedding_batch_size: int = 32,
        incremental: bool = True,
    ) -> None:
        """Index chunks into the vector store with optional incremental mode."""
        if self.vector_store is None:
            self.vector_store = get_vector_store()
            if not isinstance(self.vector_store, QdrantVectorStore):
                raise ValueError("Vector store is not QdrantVectorStore. Set VECTOR_STORE=qdrant in .env")

        if incremental:
            new_chunks = self.vector_store.filter_new_docs(chunks)
            n_skipped = len(chunks) - len(new_chunks)
            if n_skipped > 0:
                logger.info(
                    "Incremental index: %d docs skipped (already indexed), %d new docs to index",
                    n_skipped,
                    len(new_chunks),
                )
            if not new_chunks:
                logger.info("All %d chunks already indexed. Nothing to do.", len(chunks))
                return
            chunks = new_chunks

        logger.info("START index (%d chunks)...", len(chunks))
        n_batches = (len(chunks) + embedding_batch_size - 1) // embedding_batch_size

        with tqdm(total=n_batches, desc="Embedding+Indexing", unit="batch") as pbar:
            for i in range(0, len(chunks), embedding_batch_size):
                batch = chunks[i : i + embedding_batch_size]
                self.vector_store.index_documents(batch, embedding_batch_size=len(batch))
                pbar.update(1)

        logger.info("Indexing complete.")

    def evaluate(self, chunks: list[IngestionDocument]) -> EvalReport:
        """Evaluate the retrieval system using curated queries or title-based fallback."""
        if self.vector_store is None:
            self.vector_store = get_vector_store()
            if not isinstance(self.vector_store, QdrantVectorStore):
                raise ValueError("Vector store is not QdrantVectorStore.")

        logger.info("START eval...")

        # Prefer curated eval set if available
        if self.curated_eval_path.exists():
            logger.info("Using curated eval set: %s", self.curated_eval_path)
            eval_queries = load_curated_eval_set(self.curated_eval_path)
        else:
            logger.info("No curated eval set found; falling back to title-based eval (smoke test)")
            seen_parents: set[str] = set()
            title_pairs: list[tuple[str, str]] = []
            for chunk in chunks:
                parent_id = chunk.parent_id or chunk.id
                if parent_id not in seen_parents and chunk.title:
                    seen_parents.add(parent_id)
                    title_pairs.append((chunk.id, chunk.title))
            eval_queries = build_eval_set_from_titles(title_pairs, max_queries=50)

        if not eval_queries:
            logger.warning("No eval queries could be built")
            return EvalReport(recall_at_k={}, mrr=0.0, n_queries=0)

        evaluator = RetrievalEvaluator(
            search_fn=self.vector_store.search,
            id_extractor=lambda text: text[:50],
        )
        report = evaluator.evaluate(eval_queries, k_values=[1, 3, 5])
        evaluator.save_report(report, self.eval_report_path)
        return report

    # ------------------------------------------------------------------
    # Legacy CLI orchestrator (backward compatible)
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        mode: str,
        github_token: Optional[str] = None,
        max_per_repo: int = 200,
        max_postmortems: int = 800,
        chunk_preset: str = "medium",
        embedding_batch_size: int = 32,
        force: bool = False,
        run_eval: bool = False,
    ) -> Optional[EvalReport]:
        """Run the full ingestion pipeline (legacy CLI mode with checkpoint logic)."""
        start_time = time.time()
        all_raw_docs: list[IngestionDocument] = []

        if mode in ("full", "postmortems-only"):
            # Collect postmortems
            pm_checkpoint = self.checkpoints["postmortems"]
            if pm_checkpoint.exists() and not force:
                logger.info("SKIP collect_postmortems (checkpoint found)")
                pm_docs = self._load_docs(pm_checkpoint)
            else:
                pm_docs = self.collect_postmortems(
                    github_token=github_token,
                    max_entries=max_postmortems,
                )
            all_raw_docs.extend(pm_docs)

            if mode == "full":
                if not github_token:
                    raise ValueError(
                        "GitHub token is required for --mode=full. "
                        "Set GITHUB_TOKEN env var or pass --github-token."
                    )
                gh_checkpoint = self.checkpoints["github"]
                if gh_checkpoint.exists() and not force:
                    logger.info("SKIP collect_github (checkpoint found)")
                    gh_docs = self._load_docs(gh_checkpoint)
                else:
                    gh_docs = self.collect_github(github_token, max_per_repo)
                all_raw_docs.extend(gh_docs)

            logger.info("Total raw docs collected: %d", len(all_raw_docs))

        elif mode in ("index", "eval"):
            for checkpoint in [self.checkpoints["github"], self.checkpoints["postmortems"]]:
                if checkpoint.exists():
                    all_raw_docs.extend(self._load_docs(checkpoint))
            if not all_raw_docs:
                raise ValueError("No checkpoints found. Run with --mode=full or --mode=postmortems-only first.")

        if mode != "eval":
            clean_checkpoint = self.checkpoints["clean"]
            if clean_checkpoint.exists() and not force:
                logger.info("SKIP preprocess (checkpoint found)")
                clean_docs = self._load_docs(clean_checkpoint)
            else:
                clean_docs = self.preprocess(all_raw_docs)

            chunk_checkpoint = self.data_dir / f"chunks_{chunk_preset}.json"
            if chunk_checkpoint.exists() and not force:
                logger.info("SKIP chunk (checkpoint found)")
                chunks = self._load_docs(chunk_checkpoint)
            else:
                chunks = self.chunk(clean_docs, preset_name=chunk_preset)

            self.index(chunks, embedding_batch_size=embedding_batch_size)
        else:
            chunk_checkpoint = self.data_dir / f"chunks_{chunk_preset}.json"
            if not chunk_checkpoint.exists():
                raise ValueError(f"No chunk checkpoint found for preset '{chunk_preset}'. Run indexing first.")
            chunks = self._load_docs(chunk_checkpoint)

        report = None
        if run_eval or mode == "eval":
            report = self.evaluate(chunks)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Pipeline complete in {elapsed:.1f}s")
        if report:
            print(report.summary())
            print(f"Eval report saved to: {self.eval_report_path}")
        print("=" * 60)

        return report
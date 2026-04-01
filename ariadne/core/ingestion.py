from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ariadne.core.config import get_vector_store
from ariadne.core.retrieval.chunking import CHUNK_PRESETS, chunk_documents
from ariadne.core.retrieval.document import IngestionDocument
from ariadne.core.retrieval.pipeline_report import (
    CorpusDistributionMetrics,
    DataQualityMetrics,
    PerformanceMetrics,
    PipelineHealthReport,
    VectorHealthMetrics,
    compute_norm_stats,
)
from ariadne.core.retrieval.preprocessing import PreprocessReport, preprocess_documents
from ariadne.core.retrieval.vector_stores.qdrant_store import QdrantVectorStore
from evals.retrieval_eval import (
    EvalQuery,
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
        self.data_dir = repo_root / "data" / "datasets"
        self.checkpoints = {
            "github": self.data_dir / "raw_github.json",
            "postmortems": self.data_dir / "raw_postmortems.json",
            "clean": self.data_dir / "clean_docs.json",
            "chunks": self.data_dir / "chunks.json",
        }
        self.eval_report_path = self.data_dir / "eval_report.json"
        self.preprocess_report_path = self.data_dir / "preprocess_report.json"
        self.pipeline_report_path = self.data_dir / "pipeline_report.json"
        self.index_metrics_path = self.data_dir / "index_metrics.json"
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

    def collect_github(
        self,
        github_token: str,
        max_per_repo: int = 200,
        checkpoint_path: Path | None = None,
    ) -> list[IngestionDocument]:
        """Collect GitHub issues documents."""
        logger.info("START collect_github (max_per_repo=%d)...", max_per_repo)
        from scripts.collect.github_issues import GitHubIssuesCollector
        collector = GitHubIssuesCollector(token=github_token)
        docs = collector.collect(max_per_repo=max_per_repo, checkpoint_path=checkpoint_path)
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

    def preprocess(
        self,
        raw_docs: list[IngestionDocument],
        *,
        max_age_days: int | None = None,
        semantic_dedup_threshold: float | None = 0.85,
    ) -> list[IngestionDocument]:
        """Preprocess raw documents and save quality report."""
        logger.info("START preprocess (%d raw docs)...", len(raw_docs))
        clean_docs, report = preprocess_documents(
            raw_docs,
            verbose=True,
            max_age_days=max_age_days,
            semantic_dedup_threshold=semantic_dedup_threshold,
        )
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
        force_reindex: bool = False,
        embedding_model_check: bool = True,
    ) -> None:
        """Index chunks into the vector store with optional incremental mode."""
        if self.vector_store is None:
            self.vector_store = get_vector_store()
            if not isinstance(self.vector_store, QdrantVectorStore):
                raise ValueError("Vector store is not QdrantVectorStore. Set VECTOR_STORE=qdrant in .env")

        # --- Fase 8: embedding model version guard ---
        current_model = os.environ.get("EMBEDDING_PROVIDER", "unknown")
        if embedding_model_check:
            stored_model = self.vector_store.validate_embedding_model(current_model)
            if stored_model is not None and stored_model != "unknown":
                msg = (
                    f"Embedding model mismatch: collection was built with '{stored_model}', "
                    f"current model is '{current_model}'. "
                    "Delete the collection or run with force_reindex=True to rebuild."
                )
                if force_reindex:
                    logger.warning("%s Deleting collection before reindex.", msg)
                    self.vector_store.client.delete_collection(self.vector_store.collection_name)
                else:
                    raise ValueError(msg)
            elif stored_model == "unknown":
                logger.warning(
                    "Collection exists but has no 'embedding_model' in payload. "
                    "This corpus was indexed before model versioning was added. "
                    "Consider running with force_reindex=True."
                )

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

        # --- Fase 11: record pre-index state ---
        pre_index_count = self.vector_store.get_collection_stats().get("count", 0)
        successfully_indexed_ids: list[str] = []
        null_vector_count = 0
        malformed_payload_count = 0
        upsert_error_count = 0
        start = time.monotonic()

        n_batches = (len(chunks) + embedding_batch_size - 1) // embedding_batch_size

        with tqdm(total=n_batches, desc="Embedding+Indexing", unit="batch") as pbar:
            for i in range(0, len(chunks), embedding_batch_size):
                batch = chunks[i : i + embedding_batch_size]

                # --- Fase 3/12: detect malformed payloads before embedding ---
                for doc in batch:
                    if not doc.content.strip():
                        malformed_payload_count += 1
                        logger.warning("Malformed doc (empty content): %s", doc.id)

                try:
                    self.vector_store.index_documents(batch, embedding_batch_size=len(batch))
                    # Track IDs for potential rollback (Fase 11)
                    successfully_indexed_ids.extend(
                        str(self.vector_store._doc_point_id(doc)) for doc in batch
                    )
                except Exception as exc:
                    upsert_error_count += 1
                    logger.error("Batch %d failed after retries: %s", i // embedding_batch_size + 1, exc)

                pbar.update(1)

        elapsed = time.monotonic() - start
        post_index_count = self.vector_store.get_collection_stats().get("count", 0)
        partial_failure = upsert_error_count > 0

        # --- Fase 3: persist index_metrics.json ---
        index_metrics = {
            "embedding_model": current_model,
            "chunks_submitted": len(chunks),
            "pre_index_vector_count": pre_index_count,
            "post_index_vector_count": post_index_count,
            "successfully_indexed_count": len(successfully_indexed_ids),
            "null_vector_count": null_vector_count,
            "malformed_payload_count": malformed_payload_count,
            "upsert_error_count": upsert_error_count,
            "partial_failure": partial_failure,
            "duration_seconds": round(elapsed, 3),
            "embedding_batches_total": n_batches,
            "total_embedding_tokens_estimated": sum(doc.token_count for doc in chunks),
            "successfully_indexed_ids": successfully_indexed_ids,
        }
        self.index_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_metrics_path.write_text(
            json.dumps(index_metrics, indent=2), encoding="utf-8"
        )

        if partial_failure:
            logger.warning(
                "INDEX PARTIAL FAILURE: %d batch(es) failed. "
                "See %s for rollback IDs.",
                upsert_error_count,
                self.index_metrics_path,
            )
        else:
            logger.info(
                "Indexing complete: %d chunks in %.1fs (%.1f docs/s).",
                len(chunks),
                elapsed,
                len(chunks) / elapsed if elapsed > 0 else 0,
            )

    def evaluate(self, chunks: list[IngestionDocument]) -> EvalReport:
        """Evaluate the retrieval system using curated queries or title-based fallback.

        .. deprecated::
            Use :meth:`run_retrieval_eval` instead.  This alias is kept for
            backward compatibility with the legacy CLI.
        """
        return self.run_retrieval_eval(chunks)

    def run_retrieval_eval(self, chunks: list[IngestionDocument]) -> EvalReport:
        """Evaluate the retrieval system using curated queries or title-based fallback."""
        if self.vector_store is None:
            self.vector_store = get_vector_store()
            if not isinstance(self.vector_store, QdrantVectorStore):
                raise ValueError("Vector store is not QdrantVectorStore.")

        logger.info("START eval...")

        if self.curated_eval_path.exists():
            logger.info("Using curated eval set: %s", self.curated_eval_path)
            eval_queries = load_curated_eval_set(self.curated_eval_path)
            if not eval_queries:
                logger.warning("Curated eval set is empty")
                return EvalReport(recall_at_k={}, mrr=0.0, n_queries=0)
            report = self._doc_id_eval(eval_queries, k_values=[1, 3, 5])
        else:
            # Title-based fallback: use the document title as the query and as the
            # expected retrieved identifier (to_embedding_text() = "{title}\n{content}").
            logger.info("No curated eval set found; falling back to title-based eval (smoke test)")
            seen_parents: set[str] = set()
            title_pairs: list[tuple[str, str]] = []
            for chunk in chunks:
                parent_id = chunk.parent_id or chunk.id
                if parent_id not in seen_parents and chunk.title:
                    seen_parents.add(parent_id)
                    title_pairs.append((chunk.title, chunk.title))
            eval_queries = build_eval_set_from_titles(title_pairs, max_queries=50)
            if not eval_queries:
                logger.warning("No eval queries could be built (no titles found)")
                return EvalReport(recall_at_k={}, mrr=0.0, n_queries=0)
            evaluator = RetrievalEvaluator(
                search_fn=self.vector_store.search,
                # to_embedding_text() = "{title}\n{content}"; the first line is the title
                id_extractor=lambda text: text.split("\n")[0],
            )
            report = evaluator.evaluate(eval_queries, k_values=[1, 3, 5])

        RetrievalEvaluator(search_fn=self.vector_store.search).save_report(report, self.eval_report_path)
        return report

    # ------------------------------------------------------------------
    # Fase 5: consolidated pipeline health report
    # ------------------------------------------------------------------

    def generate_pipeline_report(
        self,
        chunks: list[IngestionDocument],
        *,
        chunk_preset: str = "medium",
        previous_report_path: Path | None = None,
    ) -> PipelineHealthReport:
        """Build a consolidated PipelineHealthReport from all stage sub-reports.

        Loads preprocess_report.json and index_metrics.json produced by earlier
        stages, samples vectors from Qdrant, and computes corpus distribution
        stats from *chunks*.
        """
        if self.vector_store is None:
            self.vector_store = get_vector_store()
            if not isinstance(self.vector_store, QdrantVectorStore):
                raise ValueError("Vector store is not QdrantVectorStore.")

        embedding_model = os.environ.get("EMBEDDING_PROVIDER", "unknown")

        # --- Load sub-reports ---
        preprocess_data: dict = {}
        if self.preprocess_report_path.exists():
            try:
                preprocess_data = json.loads(self.preprocess_report_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load preprocess_report.json: %s", exc)

        index_data: dict = {}
        if self.index_metrics_path.exists():
            try:
                index_data = json.loads(self.index_metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load index_metrics.json: %s", exc)

        # --- Performance metrics ---
        idx_duration = index_data.get("duration_seconds", 0.0)
        idx_chunks = index_data.get("chunks_submitted", len(chunks))
        performance = PerformanceMetrics(
            preprocess_duration_seconds=preprocess_data.get("duration_seconds", 0.0),
            index_duration_seconds=idx_duration,
            index_throughput_docs_per_sec=idx_chunks / idx_duration if idx_duration > 0 else 0.0,
            embedding_batches_total=index_data.get("embedding_batches_total", 0),
            total_embedding_tokens_estimated=index_data.get("total_embedding_tokens_estimated", 0),
        )

        # --- Data quality metrics ---
        raw_count = preprocess_data.get("input_count", 0)
        clean_count = preprocess_data.get("output_count", 0)
        post_index_count = index_data.get("post_index_vector_count", 0)
        data_quality = DataQualityMetrics(
            total_raw_docs=raw_count,
            total_clean_docs=clean_count,
            total_chunks=len(chunks),
            total_chunks_indexed=index_data.get(
                "successfully_indexed_count",
                index_data.get(
                    "post_index_vector_count",
                    index_data.get("chunks_submitted", len(chunks)),
                ),
            ),
            extraction_error_rate=preprocess_data.get("extraction_error_rate", 0.0),
            duplicate_ratio=preprocess_data.get("duplicate_ratio", 0.0),
            semantic_duplicate_ratio=preprocess_data.get("semantic_duplicate_ratio", 0.0),
            null_vector_count=index_data.get("null_vector_count", 0),
            malformed_payload_count=index_data.get("malformed_payload_count", 0),
            upsert_error_count=index_data.get("upsert_error_count", 0),
            partial_failure=index_data.get("partial_failure", False),
            per_source_counts=preprocess_data.get("per_source_counts", {}),
            content_length_chars=preprocess_data.get("content_length_stats", {}),
            token_count=preprocess_data.get("token_count_stats", {}),
        )

        # --- Vector health metrics (Fase 4) ---
        coll_stats = self.vector_store.get_collection_stats()
        collection_count = coll_stats.get("count", 0)
        embedding_dim = coll_stats.get("vector_size", 0)

        sample_vecs = self.vector_store.sample_vectors(limit=200)
        norm_mean, norm_std, norm_min, norm_max, near_zero = compute_norm_stats(sample_vecs)

        # Drift vs previous run
        norm_drift: float | None = None
        if previous_report_path is not None and previous_report_path.exists():
            try:
                prev = PipelineHealthReport.load(previous_report_path)
                prev_mean = prev.vector_health.vector_norm_mean
                if prev_mean > 0:
                    norm_drift = abs(norm_mean - prev_mean) / prev_mean
            except Exception as exc:
                logger.warning("Could not load previous pipeline report for drift calc: %s", exc)

        index_fill_ratio = (
            collection_count / len(chunks) if chunks else 0.0
        )
        vector_health = VectorHealthMetrics(
            collection_vector_count=collection_count,
            embedding_dim=embedding_dim,
            index_fill_ratio=min(index_fill_ratio, 1.0),
            vector_norm_mean=norm_mean,
            vector_norm_std=norm_std,
            vector_norm_min=norm_min,
            vector_norm_max=norm_max,
            near_zero_vector_count=near_zero,
            sample_size=len(sample_vecs),
            norm_drift_from_previous=norm_drift,
        )

        # --- Corpus distribution metrics ---
        source_counter: Counter = Counter()
        severity_counter: Counter = Counter()
        service_counter: Counter = Counter()
        tag_counter: Counter = Counter()
        parent_ids: set[str] = set()
        single_chunk_parents: set[str] = set()

        for chunk in chunks:
            source_counter[chunk.source] += 1
            severity_counter[chunk.severity] += 1
            if chunk.service:
                service_counter[chunk.service] += 1
            for tag in chunk.tags:
                tag_counter[tag] += 1
            pid = chunk.parent_id or chunk.id
            parent_ids.add(pid)
            if chunk.chunk_total == 1:
                single_chunk_parents.add(pid)

        multi_chunk_parents = parent_ids - single_chunk_parents
        avg_chunks_per_doc = len(chunks) / len(parent_ids) if parent_ids else 0.0

        top10_services = dict(service_counter.most_common(10))
        top10_tags = dict(tag_counter.most_common(10))

        corpus_distribution = CorpusDistributionMetrics(
            chunks_by_source=dict(source_counter),
            chunks_by_severity=dict(severity_counter),
            unique_services=len(service_counter),
            top10_services=top10_services,
            unique_tags=len(tag_counter),
            top10_tags=top10_tags,
            avg_chunks_per_doc=avg_chunks_per_doc,
            docs_single_chunk=len(single_chunk_parents),
            docs_multi_chunk=len(multi_chunk_parents),
            chunk_size_preset=chunk_preset,
        )

        report = PipelineHealthReport(
            embedding_model=embedding_model,
            performance=performance,
            data_quality=data_quality,
            vector_health=vector_health,
            corpus_distribution=corpus_distribution,
        )
        report.save(self.pipeline_report_path)
        return report

    def _doc_id_eval(
        self,
        eval_queries: list[EvalQuery],
        *,
        k_values: list[int] | None = None,
    ) -> EvalReport:
        """ID-based recall for curated queries.

        Each EvalQuery.relevant_doc_ids contains actual document parent IDs.
        Recall@k: fraction of relevant IDs found in the parent_ids of top-k retrieved docs.
        MRR: reciprocal rank of the first retrieved doc whose parent_id is relevant.
        """
        if k_values is None:
            k_values = [1, 3, 5]

        recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        per_query_results: list[dict] = []

        for i, eq in enumerate(eval_queries):
            relevant_ids = set(eq.relevant_doc_ids)
            try:
                retrieved_docs = self.vector_store.search_with_metadata(eq.query_text)
            except Exception as exc:
                logger.warning("Search failed for query '%s': %s", eq.query_text[:50], exc)
                retrieved_docs = []

            # parent_id falls back to id when not set (e.g. non-chunked docs)
            retrieved_parent_ids = [
                d.get("parent_id") or d.get("id", "") for d in retrieved_docs
            ]

            # MRR
            rr = 0.0
            for rank, pid in enumerate(retrieved_parent_ids, start=1):
                if pid in relevant_ids:
                    rr = 1.0 / rank
                    break
            mrr_sum += rr

            per_query_result: dict = {
                "query_id": eq.query_id,
                "query_text": eq.query_text[:80],
                "retrieved_parent_ids": retrieved_parent_ids,
                "relevant_doc_ids": list(relevant_ids),
                "reciprocal_rank": round(rr, 4),
            }
            for k in k_values:
                top_k_ids = set(retrieved_parent_ids[:k])
                covered = len(top_k_ids & relevant_ids)
                r_at_k = covered / len(relevant_ids) if relevant_ids else 0.0
                recall_sums[k] += r_at_k
                per_query_result[f"recall_at_{k}"] = round(r_at_k, 4)

            per_query_results.append(per_query_result)
            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d/%d queries...", i + 1, len(eval_queries))

        n = len(eval_queries)
        report = EvalReport(
            recall_at_k={k: round(recall_sums[k] / n, 4) for k in k_values},
            mrr=round(mrr_sum / n, 4),
            n_queries=n,
            per_query=per_query_results,
        )
        logger.info(report.summary())
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

        pipeline_report = None
        if run_eval or mode == "eval":
            pipeline_report = self.generate_pipeline_report(
                chunks,
                chunk_preset=chunk_preset,
                previous_report_path=self.pipeline_report_path,
            )

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Pipeline complete in {elapsed:.1f}s")
        if pipeline_report:
            print(pipeline_report.summary())
            print(f"Pipeline report saved to: {self.pipeline_report_path}")
        print("=" * 60)

        return pipeline_report
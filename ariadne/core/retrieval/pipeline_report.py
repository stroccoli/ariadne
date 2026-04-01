from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Sub-report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Throughput and latency metrics for each pipeline stage."""

    collect_duration_seconds: float | None = None
    preprocess_duration_seconds: float = 0.0
    chunk_duration_seconds: float = 0.0
    index_duration_seconds: float = 0.0
    index_throughput_docs_per_sec: float = 0.0
    embedding_batches_total: int = 0
    total_embedding_tokens_estimated: int = 0
    total_upsert_retries: int = 0
    total_pipeline_duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "collect_duration_seconds": self.collect_duration_seconds,
            "preprocess_duration_seconds": round(self.preprocess_duration_seconds, 3),
            "chunk_duration_seconds": round(self.chunk_duration_seconds, 3),
            "index_duration_seconds": round(self.index_duration_seconds, 3),
            "index_throughput_docs_per_sec": round(self.index_throughput_docs_per_sec, 2),
            "embedding_batches_total": self.embedding_batches_total,
            "total_embedding_tokens_estimated": self.total_embedding_tokens_estimated,
            "total_upsert_retries": self.total_upsert_retries,
            "total_pipeline_duration_seconds": (
                round(self.total_pipeline_duration_seconds, 3)
                if self.total_pipeline_duration_seconds is not None
                else None
            ),
        }


@dataclass
class DataQualityMetrics:
    """Data integrity and quality metrics across all pipeline stages."""

    total_raw_docs: int = 0
    total_clean_docs: int = 0
    total_chunks: int = 0
    total_chunks_indexed: int = 0

    # Error rates
    extraction_error_rate: float = 0.0   # (rejected + dedup) / raw_docs
    duplicate_ratio: float = 0.0          # hash-dedup count / (clean + hash-dedup)
    semantic_duplicate_ratio: float = 0.0 # semantic-dedup count / input to semantic dedup

    # Embedding-level quality
    null_vector_count: int = 0            # vectors where |sum| < 1e-6
    malformed_payload_count: int = 0      # chunks with empty/missing "document" field
    upsert_error_count: int = 0           # batches that failed after all retries
    partial_failure: bool = False         # True if any batch failed

    # Source breakdown: {source: {input, output, rejected}}
    per_source_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    # Content length distribution (over clean docs, in characters)
    content_length_chars: dict[str, float] = field(default_factory=dict)

    # Token count distribution (over chunks)
    token_count: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_raw_docs": self.total_raw_docs,
            "total_clean_docs": self.total_clean_docs,
            "total_chunks": self.total_chunks,
            "total_chunks_indexed": self.total_chunks_indexed,
            "extraction_error_rate": round(self.extraction_error_rate, 4),
            "duplicate_ratio": round(self.duplicate_ratio, 4),
            "semantic_duplicate_ratio": round(self.semantic_duplicate_ratio, 4),
            "null_vector_count": self.null_vector_count,
            "malformed_payload_count": self.malformed_payload_count,
            "upsert_error_count": self.upsert_error_count,
            "partial_failure": self.partial_failure,
            "per_source_counts": self.per_source_counts,
            "content_length_chars": {k: round(v, 1) for k, v in self.content_length_chars.items()},
            "token_count": {k: round(v, 1) for k, v in self.token_count.items()},
        }


@dataclass
class VectorHealthMetrics:
    """Health and distribution of vectors in the Qdrant index."""

    collection_vector_count: int = 0
    embedding_dim: int = 0
    index_fill_ratio: float = 0.0   # indexed / input_chunks

    # L2 norm statistics over a sample of vectors
    vector_norm_mean: float = 0.0
    vector_norm_std: float = 0.0
    vector_norm_min: float = 0.0
    vector_norm_max: float = 0.0
    near_zero_vector_count: int = 0  # ||v|| < 0.01
    sample_size: int = 0

    # Drift vs previous run: |current_mean - prev_mean| / prev_mean
    norm_drift_from_previous: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_vector_count": self.collection_vector_count,
            "embedding_dim": self.embedding_dim,
            "index_fill_ratio": round(self.index_fill_ratio, 4),
            "vector_norm_mean": round(self.vector_norm_mean, 6),
            "vector_norm_std": round(self.vector_norm_std, 6),
            "vector_norm_min": round(self.vector_norm_min, 6),
            "vector_norm_max": round(self.vector_norm_max, 6),
            "near_zero_vector_count": self.near_zero_vector_count,
            "sample_size": self.sample_size,
            "norm_drift_from_previous": (
                round(self.norm_drift_from_previous, 6)
                if self.norm_drift_from_previous is not None
                else None
            ),
        }


@dataclass
class CorpusDistributionMetrics:
    """Distribution of documents across sources, severities, services, and tags."""

    chunks_by_source: dict[str, int] = field(default_factory=dict)
    chunks_by_severity: dict[str, int] = field(default_factory=dict)
    unique_services: int = 0
    top10_services: dict[str, int] = field(default_factory=dict)
    unique_tags: int = 0
    top10_tags: dict[str, int] = field(default_factory=dict)
    avg_chunks_per_doc: float = 0.0
    docs_single_chunk: int = 0
    docs_multi_chunk: int = 0
    chunk_size_preset: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_by_source": self.chunks_by_source,
            "chunks_by_severity": self.chunks_by_severity,
            "unique_services": self.unique_services,
            "top10_services": self.top10_services,
            "unique_tags": self.unique_tags,
            "top10_tags": self.top10_tags,
            "avg_chunks_per_doc": round(self.avg_chunks_per_doc, 2),
            "docs_single_chunk": self.docs_single_chunk,
            "docs_multi_chunk": self.docs_multi_chunk,
            "chunk_size_preset": self.chunk_size_preset,
        }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclass
class PipelineHealthReport:
    """Consolidated health report for one pipeline run."""

    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding_model: str = "unknown"
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    data_quality: DataQualityMetrics = field(default_factory=DataQualityMetrics)
    vector_health: VectorHealthMetrics = field(default_factory=VectorHealthMetrics)
    corpus_distribution: CorpusDistributionMetrics = field(default_factory=CorpusDistributionMetrics)

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_at": self.run_at,
            "embedding_model": self.embedding_model,
            "performance": self.performance.to_dict(),
            "data_quality": self.data_quality.to_dict(),
            "vector_health": self.vector_health.to_dict(),
            "corpus_distribution": self.corpus_distribution.to_dict(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineHealthReport":
        data = json.loads(path.read_text(encoding="utf-8"))
        report = cls(
            run_at=data.get("run_at", ""),
            embedding_model=data.get("embedding_model", "unknown"),
        )
        p = data.get("performance", {})
        report.performance = PerformanceMetrics(
            collect_duration_seconds=p.get("collect_duration_seconds"),
            preprocess_duration_seconds=p.get("preprocess_duration_seconds", 0.0),
            chunk_duration_seconds=p.get("chunk_duration_seconds", 0.0),
            index_duration_seconds=p.get("index_duration_seconds", 0.0),
            index_throughput_docs_per_sec=p.get("index_throughput_docs_per_sec", 0.0),
            embedding_batches_total=p.get("embedding_batches_total", 0),
            total_embedding_tokens_estimated=p.get("total_embedding_tokens_estimated", 0),
            total_upsert_retries=p.get("total_upsert_retries", 0),
            total_pipeline_duration_seconds=p.get("total_pipeline_duration_seconds"),
        )
        dq = data.get("data_quality", {})
        report.data_quality = DataQualityMetrics(
            total_raw_docs=dq.get("total_raw_docs", 0),
            total_clean_docs=dq.get("total_clean_docs", 0),
            total_chunks=dq.get("total_chunks", 0),
            total_chunks_indexed=dq.get("total_chunks_indexed", 0),
            extraction_error_rate=dq.get("extraction_error_rate", 0.0),
            duplicate_ratio=dq.get("duplicate_ratio", 0.0),
            semantic_duplicate_ratio=dq.get("semantic_duplicate_ratio", 0.0),
            null_vector_count=dq.get("null_vector_count", 0),
            malformed_payload_count=dq.get("malformed_payload_count", 0),
            upsert_error_count=dq.get("upsert_error_count", 0),
            partial_failure=dq.get("partial_failure", False),
            per_source_counts=dq.get("per_source_counts", {}),
            content_length_chars=dq.get("content_length_chars", {}),
            token_count=dq.get("token_count", {}),
        )
        vh = data.get("vector_health", {})
        report.vector_health = VectorHealthMetrics(
            collection_vector_count=vh.get("collection_vector_count", 0),
            embedding_dim=vh.get("embedding_dim", 0),
            index_fill_ratio=vh.get("index_fill_ratio", 0.0),
            vector_norm_mean=vh.get("vector_norm_mean", 0.0),
            vector_norm_std=vh.get("vector_norm_std", 0.0),
            vector_norm_min=vh.get("vector_norm_min", 0.0),
            vector_norm_max=vh.get("vector_norm_max", 0.0),
            near_zero_vector_count=vh.get("near_zero_vector_count", 0),
            sample_size=vh.get("sample_size", 0),
            norm_drift_from_previous=vh.get("norm_drift_from_previous"),
        )
        cd = data.get("corpus_distribution", {})
        report.corpus_distribution = CorpusDistributionMetrics(
            chunks_by_source=cd.get("chunks_by_source", {}),
            chunks_by_severity=cd.get("chunks_by_severity", {}),
            unique_services=cd.get("unique_services", 0),
            top10_services=cd.get("top10_services", {}),
            unique_tags=cd.get("unique_tags", 0),
            top10_tags=cd.get("top10_tags", {}),
            avg_chunks_per_doc=cd.get("avg_chunks_per_doc", 0.0),
            docs_single_chunk=cd.get("docs_single_chunk", 0),
            docs_multi_chunk=cd.get("docs_multi_chunk", 0),
            chunk_size_preset=cd.get("chunk_size_preset", "medium"),
        )
        return report

    # ---------------------------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------------------------

    def summary(self) -> str:
        p = self.performance
        dq = self.data_quality
        vh = self.vector_health
        cd = self.corpus_distribution

        lines = [
            "=" * 60,
            f"  Pipeline Health Report  —  {self.run_at[:19].replace('T', ' ')} UTC",
            f"  Embedding model: {self.embedding_model}",
            "=" * 60,
            "",
            "[ Performance ]",
            f"  Preprocess:   {p.preprocess_duration_seconds:.1f}s",
            f"  Chunk:        {p.chunk_duration_seconds:.1f}s",
            f"  Index:        {p.index_duration_seconds:.1f}s  ({p.index_throughput_docs_per_sec:.1f} docs/s)",
            f"  Embed batches: {p.embedding_batches_total}  (~{p.total_embedding_tokens_estimated:,} tokens estimated)",
            f"  Upsert retries: {p.total_upsert_retries}",
            "",
            "[ Data Quality & Integrity ]",
            f"  Raw docs:         {dq.total_raw_docs:,}",
            f"  Clean docs:       {dq.total_clean_docs:,}",
            f"  Extraction error rate: {dq.extraction_error_rate:.2%}",
            f"  Hash duplicate ratio:  {dq.duplicate_ratio:.2%}",
            f"  Semantic dup ratio:    {dq.semantic_duplicate_ratio:.2%}",
            f"  Chunks total:         {dq.total_chunks:,}  →  indexed: {dq.total_chunks_indexed:,}",
            f"  Null vectors:         {dq.null_vector_count}",
            f"  Malformed payloads:   {dq.malformed_payload_count}",
            f"  Upsert errors:        {dq.upsert_error_count}{'  ⚠ PARTIAL FAILURE' if dq.partial_failure else ''}",
        ]

        if dq.content_length_chars:
            cl = dq.content_length_chars
            lines.append(
                f"  Content length (chars): mean={cl.get('mean', 0):.0f}  "
                f"p50={cl.get('p50', 0):.0f}  p95={cl.get('p95', 0):.0f}"
            )
        if dq.token_count:
            tc = dq.token_count
            lines.append(
                f"  Token count:            mean={tc.get('mean', 0):.0f}  "
                f"p50={tc.get('p50', 0):.0f}  p95={tc.get('p95', 0):.0f}"
            )

        if dq.per_source_counts:
            lines.append("  Per-source:")
            for src, counts in sorted(dq.per_source_counts.items()):
                lines.append(
                    f"    {src}: in={counts.get('input', 0)}  out={counts.get('output', 0)}  rej={counts.get('rejected', 0)}"
                )

        lines += [
            "",
            "[ Vector Health ]",
            f"  Collection size:   {vh.collection_vector_count:,}  (dim={vh.embedding_dim})",
            f"  Index fill ratio:  {vh.index_fill_ratio:.2%}",
            f"  Norm mean/std:     {vh.vector_norm_mean:.4f} / {vh.vector_norm_std:.4f}",
            f"  Norm min/max:      {vh.vector_norm_min:.4f} / {vh.vector_norm_max:.4f}",
            f"  Near-zero vectors: {vh.near_zero_vector_count}  (sampled {vh.sample_size})",
        ]
        if vh.norm_drift_from_previous is not None:
            drift_pct = vh.norm_drift_from_previous * 100
            warning = "  ⚠ HIGH DRIFT" if drift_pct > 5 else ""
            lines.append(f"  Norm drift vs prev: {drift_pct:.2f}%{warning}")

        lines += [
            "",
            "[ Corpus Distribution ]",
            f"  Preset: {cd.chunk_size_preset}  |  avg chunks/doc: {cd.avg_chunks_per_doc:.1f}",
            f"  Single-chunk docs: {cd.docs_single_chunk:,}  Multi-chunk: {cd.docs_multi_chunk:,}",
            f"  Unique services: {cd.unique_services}  |  Unique tags: {cd.unique_tags}",
        ]
        if cd.chunks_by_source:
            lines.append("  By source: " + "  ".join(f"{k}={v}" for k, v in sorted(cd.chunks_by_source.items())))
        if cd.chunks_by_severity:
            lines.append("  By severity: " + "  ".join(f"{k}={v}" for k, v in sorted(cd.chunks_by_severity.items())))
        if cd.top10_services:
            top = list(cd.top10_services.items())[:5]
            lines.append("  Top services: " + "  ".join(f"{k}={v}" for k, v in top))
        if cd.top10_tags:
            top = list(cd.top10_tags.items())[:5]
            lines.append("  Top tags: " + "  ".join(f"{k}={v}" for k, v in top))

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: compute vector norm statistics from a list of vectors
# ---------------------------------------------------------------------------


def compute_norm_stats(
    vectors: list[list[float]],
) -> tuple[float, float, float, float, int]:
    """Return (mean, std, min, max, near_zero_count) of L2 norms.

    Avoids numpy dependency — uses plain Python math.
    """
    if not vectors:
        return 0.0, 0.0, 0.0, 0.0, 0

    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]
    n = len(norms)
    mean = sum(norms) / n
    variance = sum((x - mean) ** 2 for x in norms) / n
    std = math.sqrt(variance)
    return mean, std, min(norms), max(norms), sum(1 for x in norms if x < 0.01)



from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ariadne.core.retrieval.document import (
    IngestionDocument,
    compute_content_hash,
    estimate_token_count,
)
from ariadne.core.retrieval.text_utils import jaccard_similarity

logger = logging.getLogger(__name__)

MIN_CONTENT_CHARS = 80
MAX_CONTENT_CHARS = 20_000
MAX_CODE_RATIO = 0.70

# ---------------------------------------------------------------------------
# Incident semantic tag inference
# Maps canonical incident category tags → content keywords that signal them.
# Applied to ALL documents during preprocessing regardless of source.
# This normalises the tag vocabulary so the retrieval eval can measure recall
# against semantic categories (timeout, database, memory, …) instead of
# source-specific labels (GitHub issue labels, etc.).
# ---------------------------------------------------------------------------
_HTML_TAGS = re.compile(r"<[^>]+>")
_HTML_ENTITIES = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")
_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.*?)\1")
_MD_BLOCKQUOTE = re.compile(r"^>\s*", re.MULTILINE)
_MD_HR = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_GITHUB_BOILERPLATE = re.compile(
    r"(<!--.*?-->|/cc\s+@\S+|cc\s+@\S+|"
    r"Please\s+fill\s+in\s+this\s+template|"
    r"Expected\s+behavior:|Actual\s+behavior:|"
    r"Steps\s+to\s+reproduce:|"
    r"Environment information:|"
    r"What\s+version\s+of\s+.+?\s+are\s+you\s+running\?)",
    re.IGNORECASE | re.DOTALL,
)

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def strip_html(text: str) -> str:
    text = _HTML_TAGS.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = _HTML_ENTITIES.sub(" ", text)
    return text


def strip_code_blocks(text: str) -> str:
    text = _CODE_BLOCK.sub(" [CODE BLOCK REMOVED] ", text)
    text = _INLINE_CODE.sub(" ", text)
    return text


def strip_markdown(text: str) -> str:
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD_ITALIC.sub(r"\2", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _MD_HR.sub("", text)
    return text


def strip_github_boilerplate(text: str) -> str:
    return _GITHUB_BOILERPLATE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = strip_html(text)
    text = strip_code_blocks(text)
    text = strip_github_boilerplate(text)
    text = strip_markdown(text)
    text = normalize_whitespace(text)
    return text


def clean_document(doc: IngestionDocument) -> IngestionDocument:
    cleaned_content = clean_text(doc.content)
    return doc.model_copy(
        update={
            "content": cleaned_content,
            "token_count": estimate_token_count(cleaned_content),
            "content_hash": compute_content_hash(cleaned_content),
        }
    )


def passes_quality_filter(doc: IngestionDocument) -> tuple[bool, str]:
    content = doc.content

    if len(content) < MIN_CONTENT_CHARS:
        return False, f"too_short ({len(content)} chars < {MIN_CONTENT_CHARS})"

    if len(content) > MAX_CONTENT_CHARS:
        return False, f"too_long ({len(content)} chars > {MAX_CONTENT_CHARS})"

    code_markers = content.count("[CODE BLOCK REMOVED]")
    total_lines = max(1, content.count("\n") + 1)
    if code_markers / total_lines > MAX_CODE_RATIO:
        return False, f"mostly_code (code_ratio={code_markers / total_lines:.2f})"

    return True, ""


def deduplicate_documents(
    docs: list[IngestionDocument],
) -> tuple[list[IngestionDocument], int]:
    """Hash-based deduplication. Returns (unique_docs, n_removed)."""  
    seen_hashes: set[str] = set()
    unique: list[IngestionDocument] = []
    n_removed = 0

    for doc in docs:
        h = doc.content_hash or compute_content_hash(doc.content)
        if h in seen_hashes:
            n_removed += 1
            logger.debug("Duplicate removed: %s (hash=%s)", doc.id, h)
        else:
            seen_hashes.add(h)
            unique.append(doc)

    return unique, n_removed


@dataclass
class PreprocessReport:
    """Quality report for a preprocessing run."""
    input_count: int = 0
    output_count: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_mostly_code: int = 0
    rejected_stale: int = 0
    deduplicated: int = 0
    semantic_duplicates: int = 0
    rejection_details: dict[str, int] = field(default_factory=dict)
    # Per-source breakdown: {source: {input, output, rejected}}
    per_source_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Content length distribution (chars) over clean docs
    content_length_stats: dict[str, float] = field(default_factory=dict)
    # Token count distribution over clean docs
    token_count_stats: dict[str, float] = field(default_factory=dict)
    # Derived rates
    extraction_error_rate: float = 0.0   # (total_rejected + deduplicated) / input_count
    duplicate_ratio: float = 0.0          # deduplicated / (output_count + deduplicated)
    semantic_duplicate_ratio: float = 0.0
    # Temporal span of accepted docs
    oldest_doc_date: str | None = None
    newest_doc_date: str | None = None
    # Timing
    duration_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Preprocess Report: {self.input_count} → {self.output_count} docs  ({self.duration_seconds:.1f}s)",
            f"  Rejected (too_short):    {self.rejected_too_short}",
            f"  Rejected (too_long):     {self.rejected_too_long}",
            f"  Rejected (mostly_code):  {self.rejected_mostly_code}",
            f"  Rejected (stale):        {self.rejected_stale}",
            f"  Hash-deduplicated:       {self.deduplicated}",
            f"  Semantic-deduplicated:   {self.semantic_duplicates}",
            f"  Extraction error rate:   {self.extraction_error_rate:.2%}",
            f"  Hash dup ratio:          {self.duplicate_ratio:.2%}",
            f"  Semantic dup ratio:      {self.semantic_duplicate_ratio:.2%}",
        ]
        if self.oldest_doc_date:
            lines.append(f"  Doc date range: {self.oldest_doc_date} → {self.newest_doc_date}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "rejected_too_short": self.rejected_too_short,
            "rejected_too_long": self.rejected_too_long,
            "rejected_mostly_code": self.rejected_mostly_code,
            "rejected_stale": self.rejected_stale,
            "deduplicated": self.deduplicated,
            "semantic_duplicates": self.semantic_duplicates,
            "rejection_details": self.rejection_details,
            "per_source_counts": self.per_source_counts,
            "content_length_stats": {k: round(v, 1) for k, v in self.content_length_stats.items()},
            "token_count_stats": {k: round(v, 1) for k, v in self.token_count_stats.items()},
            "extraction_error_rate": round(self.extraction_error_rate, 4),
            "duplicate_ratio": round(self.duplicate_ratio, 4),
            "semantic_duplicate_ratio": round(self.semantic_duplicate_ratio, 4),
            "oldest_doc_date": self.oldest_doc_date,
            "newest_doc_date": self.newest_doc_date,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Preprocess report saved to %s", path)


def filter_by_age(
    docs: list[IngestionDocument],
    max_age_days: int,
) -> tuple[list[IngestionDocument], int]:
    """Drop documents older than *max_age_days*.

    Documents without a `created_at` timestamp are always kept — their age
    cannot be determined.  Returns (kept_docs, n_rejected_stale).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    kept: list[IngestionDocument] = []
    n_stale = 0
    for doc in docs:
        if doc.created_at is None:
            kept.append(doc)
        elif doc.created_at < cutoff:
            n_stale += 1
            logger.debug("Stale doc removed: %s (created_at=%s)", doc.id, doc.created_at.date())
        else:
            kept.append(doc)
    return kept, n_stale


def semantic_dedup_documents(
    docs: list[IngestionDocument],
    threshold: float = 0.85,
) -> tuple[list[IngestionDocument], int]:
    """Near-duplicate detection using Jaccard similarity on token sets.

    Compares each doc against all already-accepted docs.  When two docs exceed
    *threshold*, the shorter one is dropped.  O(n²) over ~1 k docs is fast
    enough (< 1 s in practice).

    Returns (unique_docs, n_removed).
    """
    accepted: list[IngestionDocument] = []
    n_removed = 0
    for doc in docs:
        is_near_dup = False
        for accepted_doc in accepted:
            sim = jaccard_similarity(doc.content, accepted_doc.content)
            if sim >= threshold:
                is_near_dup = True
                # Keep the longer document
                if len(doc.content) > len(accepted_doc.content):
                    accepted.remove(accepted_doc)
                    accepted.append(doc)
                n_removed += 1
                logger.debug(
                    "Semantic dup removed (sim=%.3f): %s vs %s",
                    sim,
                    doc.id,
                    accepted_doc.id,
                )
                break
        if not is_near_dup:
            accepted.append(doc)
    return accepted, n_removed


def _percentile_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, p50, p95, min, max from a list of numeric values."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    # Linear interpolation quantiles
    def _quantile(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])
    return {
        "mean": mean,
        "p50": _quantile(0.50),
        "p95": _quantile(0.95),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


def preprocess_documents(
    docs: list[IngestionDocument],
    *,
    verbose: bool = True,
    max_age_days: int | None = None,
    semantic_dedup_threshold: float | None = 0.85,
) -> tuple[list[IngestionDocument], PreprocessReport]:
    start = time.monotonic()
    report = PreprocessReport(input_count=len(docs))

    # --- Freshness filter (Fase 10) ---
    if max_age_days is not None:
        docs, n_stale = filter_by_age(docs, max_age_days)
        report.rejected_stale = n_stale
        report.rejection_details["stale"] = n_stale
        logger.info("Age filter (<=%d days): %d docs removed", max_age_days, n_stale)

    # --- Clean text ---
    cleaned = [clean_document(doc) for doc in docs]

    # --- Quality filter ---
    filtered: list[IngestionDocument] = []
    for doc in cleaned:
        ok, reason = passes_quality_filter(doc)
        if ok:
            filtered.append(doc)
        else:
            category = reason.split("(")[0].strip()
            report.rejection_details[category] = report.rejection_details.get(category, 0) + 1
            if category == "too_short":
                report.rejected_too_short += 1
            elif category == "too_long":
                report.rejected_too_long += 1
            elif category == "mostly_code":
                report.rejected_mostly_code += 1

    # --- Hash deduplication ---
    unique, n_hash_removed = deduplicate_documents(filtered)
    report.deduplicated = n_hash_removed

    # --- Semantic near-duplicate deduplication (Fase 9) ---
    if semantic_dedup_threshold is not None and unique:
        unique, n_sem_removed = semantic_dedup_documents(unique, threshold=semantic_dedup_threshold)
        report.semantic_duplicates = n_sem_removed
        input_to_sem = len(unique) + n_sem_removed
        report.semantic_duplicate_ratio = (
            n_sem_removed / input_to_sem if input_to_sem > 0 else 0.0
        )

    report.output_count = len(unique)
    report.duration_seconds = time.monotonic() - start

    # --- Derived rates ---
    total_rejected = (
        report.rejected_too_short
        + report.rejected_too_long
        + report.rejected_mostly_code
        + report.rejected_stale
        + report.deduplicated
    )
    report.extraction_error_rate = total_rejected / report.input_count if report.input_count else 0.0
    denom = report.output_count + report.deduplicated
    report.duplicate_ratio = report.deduplicated / denom if denom > 0 else 0.0

    # --- Per-source counts (Fase 2) ---
    source_input: dict[str, int] = {}
    for doc in docs:
        source_input[doc.source] = source_input.get(doc.source, 0) + 1
    source_output: dict[str, int] = {}
    for doc in unique:
        source_output[doc.source] = source_output.get(doc.source, 0) + 1
    for src, count in source_input.items():
        report.per_source_counts[src] = {
            "input": count,
            "output": source_output.get(src, 0),
            "rejected": count - source_output.get(src, 0),
        }

    # --- Content length + token count percentiles ---
    char_lengths = [len(doc.content) for doc in unique]
    token_counts = [float(doc.token_count) for doc in unique]
    report.content_length_stats = _percentile_stats([float(v) for v in char_lengths])
    report.token_count_stats = _percentile_stats(token_counts)

    # --- Temporal span ---
    dated_docs = [doc for doc in unique if doc.created_at is not None]
    if dated_docs:
        oldest = min(dated_docs, key=lambda d: d.created_at)  # type: ignore[arg-type]
        newest = max(dated_docs, key=lambda d: d.created_at)  # type: ignore[arg-type]
        report.oldest_doc_date = str(oldest.created_at.date())  # type: ignore[union-attr]
        report.newest_doc_date = str(newest.created_at.date())  # type: ignore[union-attr]

    if verbose:
        logger.info(report.summary())

    return unique, report

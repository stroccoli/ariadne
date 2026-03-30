

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ariadne.core.retrieval.document import (
    IngestionDocument,
    compute_content_hash,
    estimate_token_count,
)

logger = logging.getLogger(__name__)

MIN_CONTENT_CHARS = 80
MAX_CONTENT_CHARS = 20_000
MAX_CODE_RATIO = 0.70

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
    deduplicated: int = 0
    rejection_details: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Preprocess Report: {self.input_count} → {self.output_count} docs",
            f"  Rejected (too_short):    {self.rejected_too_short}",
            f"  Rejected (too_long):     {self.rejected_too_long}",
            f"  Rejected (mostly_code):  {self.rejected_mostly_code}",
            f"  Deduplicated:            {self.deduplicated}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "rejected_too_short": self.rejected_too_short,
            "rejected_too_long": self.rejected_too_long,
            "rejected_mostly_code": self.rejected_mostly_code,
            "deduplicated": self.deduplicated,
            "rejection_details": self.rejection_details,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Preprocess report saved to %s", path)


def preprocess_documents(
    docs: list[IngestionDocument],
    *,
    verbose: bool = True,
) -> tuple[list[IngestionDocument], PreprocessReport]:
    report = PreprocessReport(input_count=len(docs))
    cleaned = [clean_document(doc) for doc in docs]

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

    unique, n_removed = deduplicate_documents(filtered)
    report.deduplicated = n_removed
    report.output_count = len(unique)

    if verbose:
        logger.info(report.summary())

    return unique, report

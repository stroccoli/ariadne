

from __future__ import annotations

import logging
import re

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


def preprocess_documents(
    docs: list[IngestionDocument],
    *,
    verbose: bool = True,
) -> list[IngestionDocument]:
    n_input = len(docs)
    cleaned = [clean_document(doc) for doc in docs]
    filtered: list[IngestionDocument] = []
    rejection_reasons: dict[str, int] = {}
    for doc in cleaned:
        ok, reason = passes_quality_filter(doc)
        if ok:
            filtered.append(doc)
        else:
            rejection_reasons[reason.split("(")[0].strip()] = (
                rejection_reasons.get(reason.split("(")[0].strip(), 0) + 1
            )
    n_after_filter = len(filtered)
    unique, n_removed = deduplicate_documents(filtered)
    n_final = len(unique)

    if verbose:
        logger.info(
            "Preprocessing complete: %d → %d (filtered=%d, deduped=%d)",
            n_input,
            n_final,
            n_input - n_after_filter,
            n_removed,
        )
        if rejection_reasons:
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
                logger.info("  Rejected (%s): %d docs", reason, count)

    return unique

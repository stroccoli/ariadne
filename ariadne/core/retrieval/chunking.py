

from __future__ import annotations

import logging
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ariadne.core.retrieval.document import (
    IngestionDocument,
    compute_content_hash,
    estimate_token_count,
)

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["recursive", "fixed", "sentence"]

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
_SENTENCE_SEPARATORS = [". ", "! ", "? ", "; ", "\n\n", "\n", " ", ""]
_FIXED_SEPARATORS = [" ", ""]

SHORT_DOC_THRESHOLD = 600


def chunk_document(
    doc: IngestionDocument,
    *,
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 75,
) -> list[IngestionDocument]:
    """Split a document into chunks, preserving all parent metadata."""
    if len(doc.content) <= SHORT_DOC_THRESHOLD:
        if not doc.content_hash:
            return [
                doc.model_copy(
                    update={
                        "token_count": estimate_token_count(doc.content),
                        "content_hash": compute_content_hash(doc.content),
                    }
                )
            ]
        return [doc]

    splitter = _build_splitter(strategy, chunk_size, chunk_overlap)
    text_chunks = splitter.split_text(doc.content)

    if len(text_chunks) <= 1:
        return [doc]

    total = len(text_chunks)
    chunks: list[IngestionDocument] = []

    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc.id}-chunk-{i}-of-{total}"

        chunk = doc.model_copy(
            update={
                "id": chunk_id,
                "content": chunk_text,
                "chunk_index": i,
                "chunk_total": total,
                "parent_id": doc.id,
                "token_count": estimate_token_count(chunk_text),
                "content_hash": compute_content_hash(chunk_text),
            }
        )
        chunks.append(chunk)

    logger.debug(
        "Chunked doc '%s' (%d chars) → %d chunks (strategy=%s, size=%d, overlap=%d)",
        doc.id,
        len(doc.content),
        total,
        strategy,
        chunk_size,
        chunk_overlap,
    )

    return chunks


def chunk_documents(
    docs: list[IngestionDocument],
    *,
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 75,
) -> list[IngestionDocument]:
    result: list[IngestionDocument] = []
    n_chunked = 0

    for doc in docs:
        chunks = chunk_document(doc, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        result.extend(chunks)
        if len(chunks) > 1:
            n_chunked += 1

    logger.info(
        "Chunking complete: %d docs → %d chunks (%d docs were split)",
        len(docs),
        len(result),
        n_chunked,
    )

    return result


CHUNK_PRESETS: dict[str, dict] = {
    "small": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 50},
    "medium": {"strategy": "recursive", "chunk_size": 500, "chunk_overlap": 75},
    "large": {"strategy": "recursive", "chunk_size": 800, "chunk_overlap": 120},
    "sentence": {"strategy": "sentence", "chunk_size": 400, "chunk_overlap": 60},
}


def _build_splitter(
    strategy: ChunkStrategy, chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    separators = {
        "recursive": _RECURSIVE_SEPARATORS,
        "fixed": _FIXED_SEPARATORS,
        "sentence": _SENTENCE_SEPARATORS,
    }.get(strategy, _RECURSIVE_SEPARATORS)

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

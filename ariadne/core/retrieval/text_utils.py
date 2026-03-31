from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9_.:/-]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "after",
        "and",
        "at",
        "because",
        "before",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "while",
        "with",
    }
)


def tokenize_text(text: str) -> set[str]:
    """Tokenize text into a set of meaningful lowercase tokens.

    Filters out stopwords and tokens shorter than 3 characters.
    Used for keyword overlap scoring and semantic deduplication.
    """
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def keyword_overlap_score(query: str, document: str) -> float:
    """Fraction of query tokens that appear in the document.

    Returns 0.0 if either string produces no tokens.
    """
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return 0.0
    document_tokens = tokenize_text(document)
    if not document_tokens:
        return 0.0
    overlap = query_tokens & document_tokens
    return len(overlap) / len(query_tokens)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between the token sets of two texts.

    Returns a value in [0.0, 1.0]. Returns 0.0 when both tokenize to empty.
    """
    tokens_a = tokenize_text(text_a)
    tokens_b = tokenize_text(text_b)
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / len(union)

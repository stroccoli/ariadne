from __future__ import annotations

import hashlib
import math
import re

from ariadne.core.integrations.embeddings.base import EmbeddingClient


TOKEN_PATTERN = re.compile(r"[a-z0-9_.:/-]+")


class LocalHashEmbeddingClient(EmbeddingClient):
    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        normalized_text = text.lower()
        tokens = TOKEN_PATTERN.findall(normalized_text)
        vector = [0.0] * self.dimensions

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + math.log1p(len(token))
            vector[index] += sign * weight

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector

        return [value / magnitude for value in vector]

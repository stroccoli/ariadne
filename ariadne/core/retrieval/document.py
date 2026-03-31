from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["github_issues", "postmortem", "manual", "unknown"]
SeverityType = Literal["critical", "high", "medium", "low", "unknown"]


class IngestionDocument(BaseModel):
    id: str
    title: str = Field(default="")
    content: str
    source: SourceType = Field(default="unknown")
    source_url: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    severity: SeverityType = Field(default="unknown")
    service: str = Field(default="")
    created_at: datetime | None = Field(default=None)
    chunk_index: int = Field(default=0)
    chunk_total: int = Field(default=1)
    parent_id: str = Field(default="")
    token_count: int = Field(default=0)
    content_hash: str = Field(default="")

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(t).strip().lower() for t in value if str(t).strip()]

    @field_validator("service", mode="before")
    @classmethod
    def normalize_service(cls, value: object) -> str:
        return str(value or "").strip().lower()

    @field_validator("content", mode="before")
    @classmethod
    def content_must_not_be_empty(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("content must not be empty")
        return text

    @property
    def is_chunk(self) -> bool:
        return self.chunk_total > 1

    def to_embedding_text(self) -> str:
        if self.title:
            return f"{self.title}\n{self.content}"
        return self.content

    def to_payload(self) -> dict:
        return {
            "document": self.to_embedding_text(),
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "source_url": self.source_url,
            "tags": self.tags,
            "severity": self.severity,
            "service": self.service,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "chunk_index": self.chunk_index,
            "chunk_total": self.chunk_total,
            "parent_id": self.parent_id,
            "token_count": self.token_count,
            "embedding_model": os.environ.get("EMBEDDING_PROVIDER", "unknown"),
        }



def compute_content_hash(text: str) -> str:
    """SHA256 (16 chars) of normalized (lowercase, collapsed whitespace) text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def estimate_token_count(text: str) -> int:
    """Fast token estimate: words / 0.75 (rule of thumb for English text)."""
    words = len(text.split())
    return max(1, int(words / 0.75))

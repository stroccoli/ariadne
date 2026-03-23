from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LLMResponse:
    """Structured response from an LLM call, including token usage metadata."""

    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, json_output: bool = False) -> LLMResponse:
        """Return the model response for a prompt."""

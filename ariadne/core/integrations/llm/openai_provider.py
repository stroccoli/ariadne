from __future__ import annotations

import logging

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore[misc]
        """No-op fallback when langsmith is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False

from ariadne.core.integrations.llm.base import LLMClient, LLMResponse


logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package is required for LLM_PROVIDER=openai. "
                "Install with: pip install 'ariadne[openai]'"
            )
        self.model = model
        self.client = OpenAI(api_key=api_key)

    @traceable(run_type="llm", name="openai.generate")
    def generate(self, prompt: str, *, json_output: bool = False) -> LLMResponse:
        logger.debug("Sending prompt to OpenAI model '%s'", self.model)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        text = getattr(response, "output_text", None) or str(response)

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "output_tokens", None)
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

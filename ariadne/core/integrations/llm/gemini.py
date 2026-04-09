from __future__ import annotations

import logging

from google import genai
from google.genai import types
from google.genai.errors import ServerError
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore[misc]
        """No-op fallback when langsmith is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

from ariadne.core.integrations.llm.base import LLMClient, LLMResponse, LLMUnavailableError


logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.model_name = model
        self._client = genai.Client(api_key=api_key)

    @traceable(run_type="llm", name="gemini.generate")
    def generate(self, prompt: str, *, json_output: bool = False) -> LLMResponse:
        logger.debug("Sending prompt to Gemini model '%s'", self.model_name)
        config = (
            types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            )
            if json_output
            else None
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        except ServerError as exc:
            if exc.code == 503:
                raise LLMUnavailableError(
                    "The AI provider is experiencing high demand and is temporarily unavailable. "
                    "Please try again in a few moments."
                ) from exc
            raise

        text = response.text

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            completion_tokens = getattr(usage, "candidates_token_count", None)
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

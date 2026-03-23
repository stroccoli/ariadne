from __future__ import annotations

import logging

from langsmith import traceable
from openai import OpenAI

from ariadne.core.integrations.llm.base import LLMClient, LLMResponse


logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
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

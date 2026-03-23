from __future__ import annotations

import logging

from langsmith import traceable
import requests

from ariadne.core.integrations.llm.base import LLMClient, LLMResponse


logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout

    @traceable(run_type="llm", name="ollama.generate")
    def generate(self, prompt: str, *, json_output: bool = False) -> LLMResponse:
        logger.debug("Sending prompt to Ollama model '%s' at '%s'", self.model, self.base_url)
        request_payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if json_output:
            request_payload["format"] = "json"
            request_payload["options"] = {"temperature": 0}

        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=request_payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        text = payload.get("response", "")
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

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
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

from ariadne.core.integrations.llm.base import LLMClient, LLMResponse


logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError(
                "requests package is required for LLM_PROVIDER=ollama. "
                "Install with: pip install 'ariadne[ollama]'"
            )
        self.session = requests.Session()
        self.timeout = timeout
        # keep_alive: how long Ollama holds the model in memory after the request.
        # Set to "-1" to keep forever, "0" to unload immediately.
        # Defaults to Ollama server setting (typically 5m) when None.
        self.keep_alive = keep_alive
        # num_ctx: token context window size. Smaller values reduce KV-cache
        # memory, allowing multiple models to coexist in GPU/RAM.
        self.num_ctx = num_ctx

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
        if self.num_ctx is not None:
            request_payload.setdefault("options", {})["num_ctx"] = self.num_ctx
        if self.keep_alive is not None:
            request_payload["keep_alive"] = self.keep_alive

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

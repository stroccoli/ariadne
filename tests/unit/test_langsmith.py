from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ariadne.core.config import get_langsmith_api_key, get_langsmith_endpoint, get_langsmith_project
from ariadne.core.config import get_langsmith_workspace_id, is_langsmith_enabled
from ariadne.core.integrations.llm.ollama import OllamaClient
from ariadne.core.integrations.llm.openai_provider import OpenAIClient


class LangSmithConfigTests(unittest.TestCase):
    def test_langsmith_aliases_enable_tracing(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "test-key",
                "LANGSMITH_PROJECT": "ops-observability",
                "LANGSMITH_ENDPOINT": "https://smith.example.test",
                "LANGSMITH_WORKSPACE_ID": "workspace-123",
            },
            clear=True,
        ):
            self.assertTrue(is_langsmith_enabled())
            self.assertEqual(get_langsmith_api_key(), "test-key")
            self.assertEqual(get_langsmith_project(), "ops-observability")
            self.assertEqual(get_langsmith_endpoint(), "https://smith.example.test")
            self.assertEqual(get_langsmith_workspace_id(), "workspace-123")

    def test_langchain_names_take_precedence_when_both_are_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": "langchain-key",
                "LANGCHAIN_PROJECT": "ariadne-prod",
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "langsmith-key",
                "LANGSMITH_PROJECT": "ariadne-dev",
            },
            clear=True,
        ):
            self.assertTrue(is_langsmith_enabled())
            self.assertEqual(get_langsmith_api_key(), "langchain-key")
            self.assertEqual(get_langsmith_project(), "ariadne-prod")


class OpenAIClientTests(unittest.TestCase):
    @patch("ariadne.core.integrations.llm.openai_provider._OPENAI_AVAILABLE", True)
    @patch("ariadne.core.integrations.llm.openai_provider.OpenAI")
    def test_generate_returns_usage_metadata(self, mock_openai) -> None:
        fake_response = SimpleNamespace(
            output_text='{"ok": true}',
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_: fake_response)
        )
        mock_openai.return_value = fake_client

        client = OpenAIClient(api_key="test-key", model="gpt-test")
        response = client.generate("respond with json", json_output=True)

        self.assertEqual(response.text, '{"ok": true}')
        self.assertEqual(response.prompt_tokens, 11)
        self.assertEqual(response.completion_tokens, 7)
        self.assertEqual(response.total_tokens, 18)


class OllamaClientTests(unittest.TestCase):
    def test_generate_includes_json_format_when_requested(self) -> None:
        captured_request: dict = {}

        class _FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "response": '{"status": "ok"}',
                    "prompt_eval_count": 9,
                    "eval_count": 4,
                }

        client = OllamaClient(model="llama-test", base_url="http://ollama.test", timeout=5)

        def _fake_post(url: str, *, json: dict, timeout: int):
            captured_request["url"] = url
            captured_request["json"] = json
            captured_request["timeout"] = timeout
            return _FakeResponse()

        client.session.post = _fake_post  # type: ignore[method-assign]
        response = client.generate("return json", json_output=True)

        self.assertEqual(captured_request["url"], "http://ollama.test/api/generate")
        self.assertEqual(captured_request["json"]["format"], "json")
        self.assertEqual(captured_request["json"]["options"], {"temperature": 0})
        self.assertEqual(captured_request["timeout"], 5)
        self.assertEqual(response.text, '{"status": "ok"}')
        self.assertEqual(response.total_tokens, 13)


if __name__ == "__main__":
    unittest.main()
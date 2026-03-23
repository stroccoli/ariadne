from __future__ import annotations

import unittest
from unittest.mock import patch

from ariadne.core.agents.analyzer import analyze
from ariadne.core.integrations.llm.base import LLMResponse


class _FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(self, prompt: str, *, json_output: bool = False) -> LLMResponse:
        return LLMResponse(text=next(self._responses))


class AnalyzerTests(unittest.TestCase):
    @patch("ariadne.core.agents.analyzer.get_llm_client")
    def test_analyze_repairs_missing_confidence(self, mock_get_llm_client) -> None:
        mock_get_llm_client.return_value = _FakeLLMClient(
            responses=[
                (
                    '{"root_cause": "Database pool exhaustion due to retries.", '
                    '"recommended_actions": ["Inspect pool usage", "Reduce retry load"]}'
                ),
                '{"confidence": 0.81}',
            ]
        )

        result, _stats = analyze(
            "ERROR postgres connection pool exhausted after repeated retries",
            "Retrieved context: [Context 1] Database pool exhaustion pattern",
            mode="compact",
        )

        self.assertEqual(result.root_cause, "Database pool exhaustion due to retries.")
        self.assertEqual(result.recommended_actions, ["Inspect pool usage", "Reduce retry load"])
        self.assertEqual(result.confidence, 0.81)

    @patch("ariadne.core.agents.analyzer.get_llm_client")
    def test_analyze_returns_zero_confidence_when_repair_fails(self, mock_get_llm_client) -> None:
        mock_get_llm_client.return_value = _FakeLLMClient(
            responses=[
                (
                    '{"root_cause": "Database pool exhaustion due to retries.", '
                    '"recommended_actions": ["Inspect pool usage"]}'
                ),
                'not json',
            ]
        )

        result, _stats = analyze(
            "ERROR postgres connection pool exhausted after repeated retries",
            "Retrieved context: [Context 1] Database pool exhaustion pattern",
            mode="compact",
        )

        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
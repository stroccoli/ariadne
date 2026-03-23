from __future__ import annotations

import unittest
from unittest.mock import patch

from ariadne.core.models import AnalysisOutput
from ariadne.core.state import IncidentState


class GraphOrchestratorTests(unittest.TestCase):
    """Test the LangGraph-based orchestrator with mocked agents."""

    # --- helpers to build mock agent behaviours ---

    @staticmethod
    def _mock_classifier(state: IncidentState) -> IncidentState:
        state.incident_type = "timeout"
        state.classification_confidence = 0.9
        return state

    @staticmethod
    def _mock_rag(state: IncidentState) -> IncidentState:
        state.context = ["doc1", "doc2"]
        state.retrieval_attempts += 1
        return state

    @staticmethod
    def _high_confidence_analyzer(state: IncidentState) -> IncidentState:
        state.analysis = AnalysisOutput(
            root_cause="Connection pool timeout due to excessive load.",
            recommended_actions=["Scale connection pool", "Add retry backoff"],
            confidence=0.85,
        )
        return state

    @staticmethod
    def _low_then_high_confidence_analyzer(state: IncidentState) -> IncidentState:
        """Returns low confidence on first call, high on second."""
        if state.analysis is None:
            state.analysis = AnalysisOutput(
                root_cause="Unclear root cause.",
                recommended_actions=[],
                confidence=0.5,
            )
        else:
            state.analysis = AnalysisOutput(
                root_cause="Connection pool timeout due to excessive load.",
                recommended_actions=["Scale connection pool"],
                confidence=0.85,
            )
        return state

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @patch("ariadne.core.graph.run_classifier")
    @patch("ariadne.core.graph.run_retrieval")
    @patch("ariadne.core.graph.run_analyzer")
    def test_high_confidence_no_retry(self, mock_analyzer, mock_rag, mock_classifier):
        """When analysis confidence >= 0.7 the graph should NOT retry."""
        mock_classifier.side_effect = self._mock_classifier
        mock_rag.side_effect = self._mock_rag
        mock_analyzer.side_effect = self._high_confidence_analyzer

        from ariadne.core.graph import run_graph

        result = run_graph("ERROR timeout", mode="detailed")

        # Each function called exactly once
        self.assertEqual(mock_classifier.call_count, 1)
        self.assertEqual(mock_rag.call_count, 1)
        self.assertEqual(mock_analyzer.call_count, 1)

        # Final output built
        self.assertIsNotNone(result.final_output)
        self.assertEqual(result.final_output.incident_type, "timeout")
        self.assertEqual(result.retrieval_attempts, 1)

    @patch("ariadne.core.graph.run_classifier")
    @patch("ariadne.core.graph.run_retrieval")
    @patch("ariadne.core.graph.run_analyzer")
    def test_low_confidence_triggers_retry(self, mock_analyzer, mock_rag, mock_classifier):
        """When analysis confidence < 0.7 the graph should retry retrieve+analyze."""
        mock_classifier.side_effect = self._mock_classifier
        mock_rag.side_effect = self._mock_rag
        mock_analyzer.side_effect = self._low_then_high_confidence_analyzer

        from ariadne.core.graph import run_graph

        result = run_graph("ERROR timeout", mode="detailed")

        # Classifier once, retrieval twice, analyzer twice
        self.assertEqual(mock_classifier.call_count, 1)
        self.assertEqual(mock_rag.call_count, 2)
        self.assertEqual(mock_analyzer.call_count, 2)

        # Final confidence should be from the second (high) analysis run
        self.assertIsNotNone(result.final_output)
        self.assertGreaterEqual(result.analysis.confidence, 0.7)
        self.assertEqual(result.retrieval_attempts, 2)

    @patch("ariadne.core.graph.run_classifier")
    @patch("ariadne.core.graph.run_retrieval")
    @patch("ariadne.core.graph.run_analyzer")
    def test_retry_capped_at_max_retrieval_attempts(self, mock_analyzer, mock_rag, mock_classifier):
        """Even if confidence stays low, the graph retries at most MAX_RETRIEVAL_ATTEMPTS."""
        mock_classifier.side_effect = self._mock_classifier
        mock_rag.side_effect = self._mock_rag

        def always_low(state: IncidentState) -> IncidentState:
            state.analysis = AnalysisOutput(
                root_cause="Unclear.", recommended_actions=[], confidence=0.3
            )
            return state

        mock_analyzer.side_effect = always_low

        from ariadne.core.graph import MAX_RETRIEVAL_ATTEMPTS, run_graph

        result = run_graph("ERROR timeout", mode="detailed")

        # 1 initial + 1 retry = MAX_RETRIEVAL_ATTEMPTS total
        self.assertEqual(mock_rag.call_count, MAX_RETRIEVAL_ATTEMPTS)
        self.assertEqual(mock_analyzer.call_count, MAX_RETRIEVAL_ATTEMPTS)

        # Output still built even with low confidence
        self.assertIsNotNone(result.final_output)
        self.assertEqual(result.retrieval_attempts, MAX_RETRIEVAL_ATTEMPTS)

    @patch("ariadne.core.graph.run_classifier")
    @patch("ariadne.core.graph.run_retrieval")
    @patch("ariadne.core.graph.run_analyzer")
    def test_graph_produces_correct_output_fields(self, mock_analyzer, mock_rag, mock_classifier):
        """Verify final_output fields match the assembled state."""
        mock_classifier.side_effect = self._mock_classifier
        mock_rag.side_effect = self._mock_rag
        mock_analyzer.side_effect = self._high_confidence_analyzer

        from ariadne.core.graph import run_graph

        result = run_graph("ERROR timeout")

        self.assertEqual(result.final_output.incident_type, "timeout")
        self.assertEqual(
            result.final_output.root_cause,
            "Connection pool timeout due to excessive load.",
        )
        self.assertEqual(
            result.final_output.recommended_actions,
            ["Scale connection pool", "Add retry backoff"],
        )
        # overall confidence = min(classification=0.9, analysis=0.85) = 0.85
        self.assertEqual(result.final_output.confidence, 0.85)

    @patch("ariadne.core.graph.run_classifier")
    @patch("ariadne.core.graph.run_retrieval")
    @patch("ariadne.core.graph.run_analyzer")
    def test_analyze_incident_delegates_to_graph(self, mock_analyzer, mock_rag, mock_classifier):
        """The public run_graph() API produces a final output."""
        mock_classifier.side_effect = self._mock_classifier
        mock_rag.side_effect = self._mock_rag
        mock_analyzer.side_effect = self._high_confidence_analyzer

        from ariadne.core.graph import run_graph

        state = run_graph("ERROR timeout", mode="detailed")

        self.assertEqual(state.final_output.incident_type, "timeout")
        self.assertIsNotNone(state.final_output.root_cause)
        self.assertGreater(mock_classifier.call_count, 0)


if __name__ == "__main__":
    unittest.main()

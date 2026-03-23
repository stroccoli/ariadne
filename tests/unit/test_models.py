from __future__ import annotations

import unittest

from ariadne.core.models import AnalysisOutput, ClassificationOutput, build_output
from ariadne.core.state import IncidentState


class ModelValidationTests(unittest.TestCase):
    def test_classification_output_normalizes_case_and_confidence(self) -> None:
        result = ClassificationOutput.model_validate(
            {"incident_type": " TIMEOUT ", "confidence": "1.8"}
        )

        self.assertEqual(result.incident_type, "timeout")
        self.assertEqual(result.confidence, 1.0)

    def test_analysis_output_strips_empty_actions(self) -> None:
        result = AnalysisOutput.model_validate(
            {
                "root_cause": " Database pool exhaustion ",
                "recommended_actions": [" restart pool ", "", "   ", "check connections"],
                "confidence": "0.42",
            }
        )

        self.assertEqual(result.root_cause, "Database pool exhaustion")
        self.assertEqual(result.recommended_actions, ["restart pool", "check connections"])
        self.assertEqual(result.confidence, 0.42)

    def test_build_output_uses_conservative_confidence(self) -> None:
        classification = ClassificationOutput(incident_type="database_issue", confidence=0.83)
        analysis = AnalysisOutput(
            root_cause="Database connection pool is exhausted.",
            recommended_actions=["Reduce concurrency", "Inspect pool saturation"],
            confidence=0.517,
        )

        result = build_output(classification, analysis)

        self.assertEqual(result.incident_type, "database_issue")
        self.assertEqual(result.root_cause, "Database connection pool is exhausted.")
        self.assertEqual(result.confidence, 0.52)
        self.assertEqual(result.recommended_actions, ["Reduce concurrency", "Inspect pool saturation"])


class IncidentStateTests(unittest.TestCase):
    def test_default_state(self) -> None:
        state = IncidentState(logs="some error")
        self.assertIsNone(state.incident_type)
        self.assertIsNone(state.classification_confidence)
        self.assertEqual(state.context, [])
        self.assertIsNone(state.analysis)
        self.assertIsNone(state.final_output)
        self.assertEqual(state.retrieval_attempts, 0)
        self.assertEqual(state.mode, "detailed")

    def test_state_round_trip(self) -> None:
        state = IncidentState(logs="error", mode="compact")
        state.incident_type = "timeout"
        state.classification_confidence = 0.9
        state.context = ["doc1"]
        self.assertEqual(state.incident_type, "timeout")
        self.assertEqual(state.classification_confidence, 0.9)
        self.assertEqual(state.context, ["doc1"])


if __name__ == "__main__":
    unittest.main()
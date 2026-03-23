from __future__ import annotations

import unittest

from ariadne.core.models import AnalysisOutput, ClassificationOutput
from ariadne.core.models import build_output


class OutputAssemblyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import unittest

from ariadne.core.utils.output import parse_json_response


class OutputParsingTests(unittest.TestCase):
    def test_parse_json_response_ignores_trailing_text(self) -> None:
        parsed = parse_json_response(
            '{"incident_type": "timeout", "confidence": 0.8}\nextra explanation that should be ignored',
            {"incident_type": "unknown", "confidence": 0.0},
        )

        self.assertEqual(parsed["incident_type"], "timeout")
        self.assertEqual(parsed["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
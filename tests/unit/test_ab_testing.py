from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from evals.ab_testing import (
    ABTestRun,
    ActionRubric,
    ConceptCriterion,
    DEFAULT_AB_TEST_SAMPLES,
    IncidentSample,
    RootCauseRubric,
    ab_test_to_dict,
    evaluate_action_quality,
    evaluate_root_cause_quality,
    get_sample_by_id,
    render_ab_test_report,
    render_comparison_report,
    run_ab_test,
    save_ab_test_results,
    sample_rubric_to_dict,
    summarize_ab_test,
    compare_result_payloads,
)
from ariadne.core.models import IncidentReportOutput


TEST_SAMPLE_SET = (
    DEFAULT_AB_TEST_SAMPLES[0],
    DEFAULT_AB_TEST_SAMPLES[10],
    DEFAULT_AB_TEST_SAMPLES[40],
)


def fake_analyze_incident(logs: str, mode: str) -> IncidentReportOutput:
    if "request to pricing-service exceeded" in logs:
        incident_type = "timeout"
        confidence = 0.61 if mode == "compact" else 0.88
        root_cause = (
            "Pricing-service timed out repeatedly and exhausted retries."
            if mode == "compact"
            else "Pricing-service latency caused repeated timeout failures and exhausted the checkout retry budget."
        )
        actions = (
            ["Check pricing-service latency", "Review timeout settings"]
            if mode == "compact"
            else [
                "Validate pricing-service latency and saturation",
                "Review checkout timeout and retry backoff settings",
            ]
        )
    elif "auth-service connection refused" in logs:
        incident_type = "unknown" if mode == "compact" else "dependency_failure"
        confidence = 0.35 if mode == "compact" else 0.79
        root_cause = (
            "Auth-service is failing."
            if mode == "compact"
            else "The downstream auth-service is unavailable, with connection refusal and circuit breaker activity showing a dependency outage."
        )
        actions = (
            ["Investigate further"]
            if mode == "compact"
            else [
                "Validate auth-service connectivity and health",
                "Keep circuit breaker and degraded login fallback in place until auth-service recovers",
            ]
        )
    else:
        incident_type = "timeout" if mode == "compact" else "unknown"
        confidence = 0.41 if mode == "compact" else 0.64
        root_cause = (
            "Partner feed timeouts look related to postgres errors."
            if mode == "compact"
            else "The evidence is mixed between partner feed timeouts and transient postgres serialization failures, so the root cause remains uncertain."
        )
        actions = (
            ["Check logs", "Trace requests"]
            if mode == "compact"
            else [
                "Correlate partner feed latency with retry timing to separate dependency failures from local contention",
                "Confirm whether postgres serialization failures persist independently of partner feed timeouts",
            ]
        )

    return IncidentReportOutput(
        incident_type=incident_type,
        root_cause=root_cause,
        confidence=confidence,
        recommended_actions=actions,
    )


class AbTestingTests(unittest.TestCase):
    def test_default_samples_cover_fifty_cases(self) -> None:
        self.assertEqual(len(DEFAULT_AB_TEST_SAMPLES), 50)
        self.assertEqual(sum(sample.expected_incident_type == "timeout" for sample in DEFAULT_AB_TEST_SAMPLES), 10)
        self.assertEqual(sum(sample.expected_incident_type == "dependency_failure" for sample in DEFAULT_AB_TEST_SAMPLES), 10)
        self.assertEqual(sum(sample.expected_incident_type == "database_issue" for sample in DEFAULT_AB_TEST_SAMPLES), 10)
        self.assertEqual(sum(sample.expected_incident_type == "memory_issue" for sample in DEFAULT_AB_TEST_SAMPLES), 10)
        self.assertEqual(sum(sample.expected_incident_type == "unknown" for sample in DEFAULT_AB_TEST_SAMPLES), 10)

    def test_quality_rubrics_score_concept_coverage_without_exact_match(self) -> None:
        sample = IncidentSample(
            sample_id="test_sample",
            description="Synthetic timeout sample",
            logs="timeout logs",
            expected_incident_type="timeout",
            expectation_note="Synthetic sample",
            root_cause_rubric=RootCauseRubric(
                required_concepts=(
                    ConceptCriterion("dependency named", ("pricing-service",)),
                    ConceptCriterion("timeout evidence", ("timed out", "timeout")),
                )
            ),
            action_rubric=ActionRubric(
                required_concepts=(
                    ConceptCriterion("inspect dependency latency", ("pricing-service", "latency")),
                    ConceptCriterion("review timeouts", ("timeout", "retry")),
                ),
                minimum_actions=2,
            ),
        )

        root_cause_quality = evaluate_root_cause_quality(
            sample,
            "Pricing-service timed out repeatedly under load.",
        )
        action_quality = evaluate_action_quality(
            sample,
            [
                "Validate pricing-service latency and saturation",
                "Review timeout and retry settings in the caller",
            ],
        )

        self.assertEqual(root_cause_quality.score, 1.0)
        self.assertEqual(action_quality.score, 1.0)

    def test_sample_rubric_description_includes_expected_concepts(self) -> None:
        sample = get_sample_by_id("timeout_checkout_pricing")
        payload = sample_rubric_to_dict(sample)

        self.assertEqual(payload["sample_id"], "timeout_checkout_pricing")
        self.assertEqual(payload["root_cause_rubric"]["required_concepts"][0]["label"], "dependency named")
        self.assertEqual(payload["action_rubric"]["minimum_actions"], 2)

    def test_run_ab_test_builds_per_sample_comparisons(self) -> None:
        run = run_ab_test(samples=TEST_SAMPLE_SET, analyze_fn=fake_analyze_incident)

        self.assertEqual(len(run.comparisons), 3)
        self.assertEqual([evaluation.mode for evaluation in run.comparisons[0].evaluations], ["compact", "detailed"])
        self.assertTrue(run.comparisons[0].evaluations[0].incident_type_match)
        self.assertFalse(run.comparisons[1].evaluations[0].incident_type_match)
        self.assertTrue(run.comparisons[1].evaluations[1].incident_type_match)
        self.assertGreaterEqual(run.total_duration_seconds, 0.0)

    def test_summary_scores_modes_independently(self) -> None:
        run = run_ab_test(samples=TEST_SAMPLE_SET, analyze_fn=fake_analyze_incident)
        summaries = summarize_ab_test(run.comparisons)

        self.assertEqual(summaries[0].mode, "compact")
        self.assertEqual(summaries[0].incident_type_matches, 1)
        self.assertEqual(summaries[0].accuracy, 0.33)
        self.assertEqual(summaries[0].average_confidence, 0.46)
        self.assertEqual(summaries[0].average_actions, 1.67)
        self.assertGreater(summaries[0].average_root_cause_quality, 0.0)
        self.assertGreater(summaries[0].average_action_quality, 0.0)

        self.assertEqual(summaries[1].mode, "detailed")
        self.assertEqual(summaries[1].incident_type_matches, 3)
        self.assertEqual(summaries[1].accuracy, 1.0)
        self.assertEqual(summaries[1].average_confidence, 0.77)
        self.assertEqual(summaries[1].average_actions, 2.0)

    def test_report_json_and_save_include_quality_and_runtime(self) -> None:
        run = run_ab_test(samples=TEST_SAMPLE_SET, analyze_fn=fake_analyze_incident)

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = save_ab_test_results(run, results_dir=Path(temp_dir))
            payload = ab_test_to_dict(run, saved_results_path=saved_path)
            report = render_ab_test_report(run, saved_results_path=saved_path)
            saved_payload = json.loads(saved_path.read_text(encoding="utf-8"))

        self.assertIn("Total runtime", report)
        self.assertIn("root_cause_q", report)
        self.assertIn("action_q", report)
        self.assertNotIn("Detailed results", report)
        self.assertEqual(payload["sample_count"], 3)
        self.assertEqual(payload["summary"][0]["mode"], "compact")
        self.assertIn("saved_results_path", payload)
        self.assertEqual(saved_payload["sample_count"], 3)

    def test_detailed_report_can_be_enabled_explicitly(self) -> None:
        run = run_ab_test(samples=TEST_SAMPLE_SET, analyze_fn=fake_analyze_incident)

        report = render_ab_test_report(run, include_details=True)

        self.assertIn("Detailed results", report)
        self.assertIn(TEST_SAMPLE_SET[0].sample_id, report)

    def test_save_results_rotates_latest_into_previous(self) -> None:
        run = run_ab_test(samples=TEST_SAMPLE_SET, analyze_fn=fake_analyze_incident)

        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            first_path = save_ab_test_results(run, results_dir=results_dir)
            first_latest = json.loads((results_dir / "latest.json").read_text(encoding="utf-8"))

            second_path = save_ab_test_results(run, results_dir=results_dir)
            previous_payload = json.loads((results_dir / "previous.json").read_text(encoding="utf-8"))
            second_latest = json.loads((results_dir / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(previous_payload["saved_results_path"], str(first_path))
        self.assertEqual(second_latest["saved_results_path"], str(second_path))
        self.assertEqual(first_latest["saved_results_path"], str(first_path))

    def test_ab_test_run_serialization_shape(self) -> None:
        run = ABTestRun(
            comparisons=tuple(),
            summaries=tuple(),
            started_at="2026-03-19T00:00:00+00:00",
            finished_at="2026-03-19T00:00:01+00:00",
            total_duration_seconds=1.0,
        )

        payload = ab_test_to_dict(run)

        self.assertEqual(payload["sample_count"], 0)
        self.assertEqual(payload["total_duration_seconds"], 1.0)

    def test_compare_results_reports_mode_deltas(self) -> None:
        baseline = {
            "saved_results_path": "baseline.json",
            "started_at": "2026-03-19T00:00:00+00:00",
            "summary": [
                {
                    "mode": "compact",
                    "incident_type_matches": 10,
                    "accuracy": 0.2,
                    "average_confidence": 0.1,
                    "average_root_cause_quality": 0.3,
                    "average_action_quality": 0.2,
                    "average_duration_seconds": 4.5,
                }
            ],
        }
        candidate = {
            "saved_results_path": "candidate.json",
            "started_at": "2026-03-19T01:00:00+00:00",
            "summary": [
                {
                    "mode": "compact",
                    "incident_type_matches": 14,
                    "accuracy": 0.28,
                    "average_confidence": 0.16,
                    "average_root_cause_quality": 0.4,
                    "average_action_quality": 0.35,
                    "average_duration_seconds": 3.1,
                }
            ],
        }

        comparison = compare_result_payloads(baseline, candidate)
        report = render_comparison_report(comparison)

        self.assertEqual(comparison["rows"][0]["accuracy_delta"], 0.08)
        self.assertEqual(comparison["rows"][0]["incident_type_matches_delta"], 4)
        self.assertIn("A/B Result Comparison", report)
        self.assertIn("compact", report)


if __name__ == "__main__":
    unittest.main()
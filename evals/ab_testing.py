"""Offline A/B evaluation public API.

This module re-exports the benchmark surface so callers keep one stable import
path while the implementation remains split into smaller modules.
"""

from evals.benchmark_models import (
    ABTestRun,
    ActionQualityResult,
    ActionRubric,
    ConceptCriterion,
    IncidentSample,
    ModeEvaluation,
    ModeSummary,
    RootCauseQualityResult,
    RootCauseRubric,
    SampleEvaluation,
)
from evals.benchmark_runner import run_ab_test, summarize_ab_test
from evals.provider_runner import run_provider_test
from evals.reporting import (
    ab_test_to_dict,
    compare_result_payloads,
    load_ab_test_results,
    render_ab_test_report,
    render_comparison_report,
    render_provider_test_report,
    render_sample_rubric,
    save_ab_test_results,
    save_provider_test_results,
)
from evals.rubric_scoring import evaluate_action_quality, evaluate_root_cause_quality, sample_rubric_to_dict
from evals.sample_library import DEFAULT_AB_TEST_SAMPLES, get_random_samples, get_sample_by_id, list_sample_ids

__all__ = [
    "ABTestRun",
    "ActionQualityResult",
    "ActionRubric",
    "ConceptCriterion",
    "DEFAULT_AB_TEST_SAMPLES",
    "get_random_samples",
    "IncidentSample",
    "ModeEvaluation",
    "ModeSummary",
    "RootCauseQualityResult",
    "RootCauseRubric",
    "SampleEvaluation",
    "ab_test_to_dict",
    "compare_result_payloads",
    "evaluate_action_quality",
    "evaluate_root_cause_quality",
    "get_sample_by_id",
    "list_sample_ids",
    "load_ab_test_results",
    "render_ab_test_report",
    "render_comparison_report",
    "render_provider_test_report",
    "render_sample_rubric",
    "run_ab_test",
    "run_provider_test",
    "sample_rubric_to_dict",
    "save_ab_test_results",
    "save_provider_test_results",
    "summarize_ab_test",
]
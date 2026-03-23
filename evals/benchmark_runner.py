from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Callable, Sequence

from evals.benchmark_models import ABTestRun, ModeEvaluation, ModeSummary, SampleEvaluation
from evals.rubric_scoring import evaluate_action_quality, evaluate_root_cause_quality
from evals.sample_library import DEFAULT_AB_TEST_SAMPLES
from ariadne.core.models import ALLOWED_PROMPT_MODES, IncidentReportOutput
from ariadne.core.graph import run_graph
from ariadne.core.state import IncidentState


def _analyze_incident(logs: str, mode: str) -> tuple[IncidentReportOutput, IncidentState]:
    state = run_graph(logs, mode=mode)
    if state.final_output is None:
        raise RuntimeError("Graph produced no output")
    return state.final_output, state


def run_ab_test(
    samples: Sequence = DEFAULT_AB_TEST_SAMPLES,
    modes: Sequence[str] = ("compact", "detailed"),
    analyze_fn: Callable[[str, str], tuple[IncidentReportOutput, IncidentState] | IncidentReportOutput] = _analyze_incident,
) -> ABTestRun:
    normalized_modes: list[str] = []
    for mode in modes:
        if mode in ALLOWED_PROMPT_MODES and mode not in normalized_modes:
            normalized_modes.append(mode)

    if not normalized_modes:
        raise ValueError("At least one valid prompt mode is required for A/B testing")

    selected_samples = tuple(samples)
    started_at = datetime.now(timezone.utc).isoformat()
    suite_start = perf_counter()
    comparisons: list[SampleEvaluation] = []

    for sample in selected_samples:
        evaluations: list[ModeEvaluation] = []
        for mode in normalized_modes:
            evaluation_start = perf_counter()
            result = analyze_fn(sample.logs, mode)
            duration_seconds = round(perf_counter() - evaluation_start, 4)

            # Support both old (report-only) and new (report, state) return styles
            if isinstance(result, tuple):
                report, run_state = result
                llm_calls = getattr(run_state, "total_llm_calls", 0)
                prompt_tokens = getattr(run_state, "total_prompt_tokens", 0)
                completion_tokens = getattr(run_state, "total_completion_tokens", 0)
            else:
                report = result
                llm_calls = 0
                prompt_tokens = 0
                completion_tokens = 0

            evaluations.append(
                ModeEvaluation(
                    mode=mode,
                    report=report,
                    incident_type_match=report.incident_type == sample.expected_incident_type,
                    root_cause_quality=evaluate_root_cause_quality(sample, report.root_cause),
                    action_quality=evaluate_action_quality(sample, report.recommended_actions),
                    duration_seconds=duration_seconds,
                    llm_calls=llm_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

        comparisons.append(SampleEvaluation(sample=sample, evaluations=tuple(evaluations)))

    total_duration_seconds = round(perf_counter() - suite_start, 4)
    return ABTestRun(
        comparisons=tuple(comparisons),
        summaries=summarize_ab_test(comparisons),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        total_duration_seconds=total_duration_seconds,
    )


def summarize_ab_test(comparisons: Sequence[SampleEvaluation]) -> tuple[ModeSummary, ...]:
    ordered_modes: list[str] = []
    evaluations_by_mode: dict[str, list[ModeEvaluation]] = {}

    for comparison in comparisons:
        for evaluation in comparison.evaluations:
            if evaluation.mode not in evaluations_by_mode:
                ordered_modes.append(evaluation.mode)
                evaluations_by_mode[evaluation.mode] = []
            evaluations_by_mode[evaluation.mode].append(evaluation)

    summaries: list[ModeSummary] = []
    for mode in ordered_modes:
        evaluations = evaluations_by_mode[mode]
        total_samples = len(evaluations)
        matches = sum(evaluation.incident_type_match for evaluation in evaluations)
        total_duration_seconds = round(sum(evaluation.duration_seconds for evaluation in evaluations), 4)
        total_llm_calls = sum(evaluation.llm_calls for evaluation in evaluations)
        total_prompt_tokens = sum(evaluation.prompt_tokens for evaluation in evaluations)
        total_completion_tokens = sum(evaluation.completion_tokens for evaluation in evaluations)
        summaries.append(
            ModeSummary(
                mode=mode,
                total_samples=total_samples,
                incident_type_matches=matches,
                accuracy=round(matches / total_samples, 2) if total_samples else 0.0,
                average_confidence=round(mean(evaluation.report.confidence for evaluation in evaluations), 2)
                if total_samples
                else 0.0,
                average_actions=round(mean(len(evaluation.report.recommended_actions) for evaluation in evaluations), 2)
                if total_samples
                else 0.0,
                average_root_cause_quality=round(mean(evaluation.root_cause_quality.score for evaluation in evaluations), 2)
                if total_samples
                else 0.0,
                average_action_quality=round(mean(evaluation.action_quality.score for evaluation in evaluations), 2)
                if total_samples
                else 0.0,
                total_duration_seconds=total_duration_seconds,
                average_duration_seconds=round(total_duration_seconds / total_samples, 4) if total_samples else 0.0,
                total_llm_calls=total_llm_calls,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
            )
        )

    return tuple(summaries)
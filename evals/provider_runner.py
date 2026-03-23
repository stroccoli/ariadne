"""Provider comparison test runner.

Compares multiple LLM providers against the same set of randomly-selected
incident samples, using a fixed prompt mode for every provider.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Sequence

from evals.benchmark_models import ABTestRun, ModeEvaluation, SampleEvaluation
from evals.benchmark_runner import summarize_ab_test
from evals.rubric_scoring import evaluate_action_quality, evaluate_root_cause_quality
from evals.sample_library import get_random_samples

from ariadne.core.config import get_llm_client
from ariadne.core.graph import run_graph
from ariadne.core.models import IncidentReportOutput
from ariadne.core.state import IncidentState

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("openai", "gemini", "ollama")


def _make_provider_analyze_fn(
    provider: str,
    mode: str,
) -> Callable[[str, str], tuple[IncidentReportOutput, IncidentState]]:
    """Return a closure that analyses logs using a specific LLM provider."""

    def _analyze(logs: str, _mode: str) -> tuple[IncidentReportOutput, IncidentState]:
        os.environ["LLM_PROVIDER"] = provider
        get_llm_client.cache_clear()

        state = run_graph(logs, mode=mode)
        if state.final_output is None:
            raise RuntimeError(f"Graph produced no output for provider {provider}")
        return state.final_output, state

    return _analyze


def run_provider_test(
    providers: Sequence[str] = SUPPORTED_PROVIDERS,
    mode: str = "detailed",
    num_samples: int = 10,
    seed: int | None = None,
    analyze_fn_factory: Callable[
        [str, str], Callable[[str, str], tuple[IncidentReportOutput, IncidentState]]
    ]
    | None = None,
) -> ABTestRun:
    """Run the same samples against each provider and return an ``ABTestRun``.

    Parameters
    ----------
    providers:
        Provider names to compare (e.g. ``("openai", "gemini", "ollama")``).
    mode:
        Fixed prompt mode used for every provider (``"detailed"`` or ``"compact"``).
    num_samples:
        How many random samples to draw from the library.
    seed:
        Optional seed for reproducible sample selection.
    analyze_fn_factory:
        Override for building per-provider analysis functions (useful for testing).
        Signature: ``(provider, mode) -> analyze_fn`` where
        ``analyze_fn(logs, mode) -> (report, state)``.
    """
    factory = analyze_fn_factory or _make_provider_analyze_fn
    samples = get_random_samples(num_samples, seed=seed)

    started_at = datetime.now(timezone.utc).isoformat()
    suite_start = perf_counter()
    comparisons: list[SampleEvaluation] = []

    for sample in samples:
        evaluations: list[ModeEvaluation] = []

        for provider in providers:
            analyze_fn = factory(provider, mode)
            eval_start = perf_counter()

            try:
                result = analyze_fn(sample.logs, mode)
            except Exception:
                logger.exception("Provider '%s' failed on sample '%s' — skipping", provider, sample.sample_id)
                continue

            duration = round(perf_counter() - eval_start, 4)

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
                    mode=provider,
                    report=report,
                    incident_type_match=report.incident_type == sample.expected_incident_type,
                    root_cause_quality=evaluate_root_cause_quality(sample, report.root_cause),
                    action_quality=evaluate_action_quality(sample, report.recommended_actions),
                    duration_seconds=duration,
                    llm_calls=llm_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

        comparisons.append(SampleEvaluation(sample=sample, evaluations=tuple(evaluations)))

    total_duration = round(perf_counter() - suite_start, 4)

    return ABTestRun(
        comparisons=tuple(comparisons),
        summaries=summarize_ab_test(comparisons),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        total_duration_seconds=total_duration,
    )

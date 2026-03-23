from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Sequence

from evals.benchmark_models import ABTestRun, IncidentSample, SampleEvaluation
from evals.rubric_scoring import sample_rubric_to_dict


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ab_test_to_dict(run: ABTestRun, saved_results_path: Path | None = None) -> dict:
    return {
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "total_duration_seconds": run.total_duration_seconds,
        "sample_count": len(run.comparisons),
        "saved_results_path": str(saved_results_path) if saved_results_path else None,
        "samples": [_sample_evaluation_to_dict(comparison) for comparison in run.comparisons],
        "summary": [
            {
                "mode": summary.mode,
                "total_samples": summary.total_samples,
                "incident_type_matches": summary.incident_type_matches,
                "accuracy": summary.accuracy,
                "average_confidence": summary.average_confidence,
                "average_actions": summary.average_actions,
                "average_root_cause_quality": summary.average_root_cause_quality,
                "average_action_quality": summary.average_action_quality,
                "total_duration_seconds": summary.total_duration_seconds,
                "average_duration_seconds": summary.average_duration_seconds,
                "total_llm_calls": summary.total_llm_calls,
                "total_prompt_tokens": summary.total_prompt_tokens,
                "total_completion_tokens": summary.total_completion_tokens,
            }
            for summary in run.summaries
        ],
    }


def save_ab_test_results(run: ABTestRun, results_dir: Path | None = None) -> Path:
    target_dir = results_dir or DEFAULT_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    file_path = target_dir / f"ab_test_{timestamp}.json"
    latest_path = target_dir / "latest.json"
    previous_path = target_dir / "previous.json"
    payload = ab_test_to_dict(run, saved_results_path=file_path)
    serialized = json.dumps(payload, indent=2) + "\n"

    if latest_path.exists():
        shutil.copyfile(latest_path, previous_path)

    file_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")
    return file_path


def load_ab_test_results(file_path: str | Path) -> dict:
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def render_ab_test_report(
    run: ABTestRun,
    saved_results_path: Path | None = None,
    *,
    include_details: bool = False,
) -> str:
    if not run.comparisons:
        return "No A/B test samples were provided."

    mode_order = [evaluation.mode for evaluation in run.comparisons[0].evaluations]
    overview_rows: list[list[str]] = []
    for comparison in run.comparisons:
        row = [comparison.sample.sample_id, comparison.sample.expected_incident_type]
        for mode in mode_order:
            evaluation = next(item for item in comparison.evaluations if item.mode == mode)
            status = "match" if evaluation.incident_type_match else "miss"
            row.append(f"{evaluation.report.incident_type} ({status})")
        overview_rows.append(row)

    summary_rows = [
        [
            summary.mode,
            f"{summary.incident_type_matches}/{summary.total_samples}",
            f"{summary.accuracy:.2f}",
            f"{summary.average_root_cause_quality:.2f}",
            f"{summary.average_action_quality:.2f}",
            f"{summary.total_duration_seconds:.2f}s",
            str(summary.total_llm_calls),
            str(summary.total_prompt_tokens + summary.total_completion_tokens),
        ]
        for summary in run.summaries
    ]

    lines = [
        "Prompt A/B Evaluation",
        f"Total samples: {len(run.comparisons)}",
        f"Total runtime: {run.total_duration_seconds:.2f}s",
    ]
    if saved_results_path:
        lines.append(f"Results saved to: {saved_results_path}")

    lines.extend(["", "Sample overview"])
    lines.extend(_format_table(["sample", "expected", *mode_order], overview_rows))
    lines.extend(["", "Mode summary"])
    lines.extend(
        _format_table(
            ["mode", "matches", "accuracy", "root_cause_q", "action_q", "total_time", "llm_calls", "tokens"],
            summary_rows,
        )
    )

    if include_details:
        lines.append("")
        lines.append("Detailed results")

        for comparison in run.comparisons:
            lines.extend(_render_sample_details(comparison))

    return "\n".join(lines)


def render_sample_rubric(sample: IncidentSample) -> str:
    payload = sample_rubric_to_dict(sample)
    lines = [
        f"Sample: {payload['sample_id']}",
        f"Description: {payload['description']}",
        f"Expected incident type: {payload['expected_incident_type']}",
        f"Why this case exists: {payload['expectation_note']}",
        "",
        "Root-cause rubric",
    ]
    for criterion in payload["root_cause_rubric"]["required_concepts"]:
        lines.append(f"- {criterion['label']}: {', '.join(criterion['keywords'])}")
    lines.append(
        f"Require uncertainty: {'yes' if payload['root_cause_rubric']['require_uncertainty'] else 'no'}"
    )
    if payload["root_cause_rubric"]["forbidden_terms"]:
        lines.append(
            f"Forbidden terms: {', '.join(payload['root_cause_rubric']['forbidden_terms'])}"
        )

    lines.extend(["", "Action rubric"])
    for criterion in payload["action_rubric"]["required_concepts"]:
        lines.append(f"- {criterion['label']}: {', '.join(criterion['keywords'])}")
    lines.append(f"Minimum actions: {payload['action_rubric']['minimum_actions']}")
    lines.append(
        f"Discouraged phrases: {', '.join(payload['action_rubric']['discouraged_phrases'])}"
    )
    lines.extend(["", "Logs", payload["logs"]])
    return "\n".join(lines)


def compare_result_payloads(baseline: dict, candidate: dict) -> dict:
    baseline_summary = {item["mode"]: item for item in baseline.get("summary", [])}
    candidate_summary = {item["mode"]: item for item in candidate.get("summary", [])}
    ordered_modes = list(dict.fromkeys([*baseline_summary.keys(), *candidate_summary.keys()]))
    comparison_rows = []
    for mode in ordered_modes:
        baseline_mode = baseline_summary.get(mode, {})
        candidate_mode = candidate_summary.get(mode, {})
        comparison_rows.append(
            {
                "mode": mode,
                "accuracy_delta": round(candidate_mode.get("accuracy", 0.0) - baseline_mode.get("accuracy", 0.0), 2),
                "confidence_delta": round(candidate_mode.get("average_confidence", 0.0) - baseline_mode.get("average_confidence", 0.0), 2),
                "root_cause_quality_delta": round(candidate_mode.get("average_root_cause_quality", 0.0) - baseline_mode.get("average_root_cause_quality", 0.0), 2),
                "action_quality_delta": round(candidate_mode.get("average_action_quality", 0.0) - baseline_mode.get("average_action_quality", 0.0), 2),
                "average_duration_seconds_delta": round(candidate_mode.get("average_duration_seconds", 0.0) - baseline_mode.get("average_duration_seconds", 0.0), 4),
                "incident_type_matches_delta": candidate_mode.get("incident_type_matches", 0) - baseline_mode.get("incident_type_matches", 0),
                "baseline": baseline_mode,
                "candidate": candidate_mode,
            }
        )

    return {
        "baseline_path": baseline.get("saved_results_path"),
        "candidate_path": candidate.get("saved_results_path"),
        "baseline_started_at": baseline.get("started_at"),
        "candidate_started_at": candidate.get("started_at"),
        "rows": comparison_rows,
    }


def render_comparison_report(comparison: dict) -> str:
    rows = [
        [
            row["mode"],
            _format_delta(row["accuracy_delta"], precision=2),
            _format_delta(row["root_cause_quality_delta"], precision=2),
            _format_delta(row["action_quality_delta"], precision=2),
            _format_delta(row["confidence_delta"], precision=2),
            _format_delta(row["average_duration_seconds_delta"], precision=4, suffix="s"),
            _format_delta(row["incident_type_matches_delta"], integer=True),
        ]
        for row in comparison["rows"]
    ]
    lines = [
        "A/B Result Comparison",
        f"Baseline: {comparison['baseline_path']}",
        f"Candidate: {comparison['candidate_path']}",
        "",
        "Mode deltas",
    ]
    lines.extend(
        _format_table(
            ["mode", "accuracy", "root_cause_q", "action_q", "confidence", "avg_time", "matches"],
            rows,
        )
    )
    return "\n".join(lines)


def _sample_evaluation_to_dict(comparison: SampleEvaluation) -> dict:
    return {
        "sample_id": comparison.sample.sample_id,
        "description": comparison.sample.description,
        "expected_incident_type": comparison.sample.expected_incident_type,
        "expectation_note": comparison.sample.expectation_note,
        "logs": comparison.sample.logs,
        "evaluations": [
            {
                "mode": evaluation.mode,
                "incident_type_match": evaluation.incident_type_match,
                "duration_seconds": evaluation.duration_seconds,
                "root_cause_quality": {
                    "score": evaluation.root_cause_quality.score,
                    "matched_concepts": list(evaluation.root_cause_quality.matched_concepts),
                    "missed_concepts": list(evaluation.root_cause_quality.missed_concepts),
                    "forbidden_hits": list(evaluation.root_cause_quality.forbidden_hits),
                    "uncertainty_satisfied": evaluation.root_cause_quality.uncertainty_satisfied,
                },
                "action_quality": {
                    "score": evaluation.action_quality.score,
                    "matched_concepts": list(evaluation.action_quality.matched_concepts),
                    "missed_concepts": list(evaluation.action_quality.missed_concepts),
                    "discouraged_hits": list(evaluation.action_quality.discouraged_hits),
                    "action_count": evaluation.action_quality.action_count,
                    "minimum_actions_met": evaluation.action_quality.minimum_actions_met,
                },
                "report": evaluation.report.model_dump(),
                "llm_calls": evaluation.llm_calls,
                "prompt_tokens": evaluation.prompt_tokens,
                "completion_tokens": evaluation.completion_tokens,
            }
            for evaluation in comparison.evaluations
        ],
    }


def _render_sample_details(comparison: SampleEvaluation) -> list[str]:
    lines = [
        "",
        f"[{comparison.sample.sample_id}] {comparison.sample.description}",
        f"Expected incident type: {comparison.sample.expected_incident_type}",
        f"Why this case exists: {comparison.sample.expectation_note}",
    ]
    for evaluation in comparison.evaluations:
        lines.append(
            f"- {evaluation.mode}: type={evaluation.report.incident_type}, "
            f"match={'yes' if evaluation.incident_type_match else 'no'}, "
            f"confidence={evaluation.report.confidence:.2f}, "
            f"root_cause_q={evaluation.root_cause_quality.score:.2f}, "
            f"action_q={evaluation.action_quality.score:.2f}, "
            f"time={evaluation.duration_seconds:.2f}s, "
            f"llm_calls={evaluation.llm_calls}, "
            f"tokens={evaluation.prompt_tokens + evaluation.completion_tokens}"
        )
        lines.append(f"  root_cause: {evaluation.report.root_cause}")
        lines.append(
            f"  root_cause rubric: matched={', '.join(evaluation.root_cause_quality.matched_concepts) or 'none'}; "
            f"missed={', '.join(evaluation.root_cause_quality.missed_concepts) or 'none'}; "
            f"uncertainty_ok={'yes' if evaluation.root_cause_quality.uncertainty_satisfied else 'no'}"
        )
        if evaluation.report.recommended_actions:
            lines.append(f"  recommended_actions: {'; '.join(evaluation.report.recommended_actions)}")
        else:
            lines.append("  recommended_actions: none")
        lines.append(
            f"  action rubric: matched={', '.join(evaluation.action_quality.matched_concepts) or 'none'}; "
            f"missed={', '.join(evaluation.action_quality.missed_concepts) or 'none'}; "
            f"minimum_actions_met={'yes' if evaluation.action_quality.minimum_actions_met else 'no'}"
        )
    return lines


def _format_delta(value: float | int, precision: int = 2, suffix: str = "", integer: bool = False) -> str:
    if integer:
        return f"{value:+d}{suffix}"
    return f"{value:+.{precision}f}{suffix}"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    formatted_rows = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]

    for row in rows:
        formatted_rows.append(
            "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        )

    return formatted_rows


# ---------------------------------------------------------------------------
# Provider comparison helpers
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "provider"


def save_provider_test_results(run: ABTestRun, results_dir: Path | None = None) -> Path:
    """Persist a provider comparison run to *results_dir* (default ``evals/results/provider/``)."""
    target_dir = results_dir or DEFAULT_PROVIDER_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    file_path = target_dir / f"provider_test_{timestamp}.json"
    latest_path = target_dir / "latest.json"
    previous_path = target_dir / "previous.json"

    payload = ab_test_to_dict(run, saved_results_path=file_path)
    serialized = json.dumps(payload, indent=2) + "\n"

    if latest_path.exists():
        shutil.copyfile(latest_path, previous_path)

    file_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")
    return file_path


def render_provider_test_report(
    run: ABTestRun,
    saved_results_path: Path | None = None,
    *,
    include_details: bool = False,
) -> str:
    """Render a human-readable provider comparison report."""
    if not run.comparisons:
        return "No provider test samples were provided."

    provider_order = [evaluation.mode for evaluation in run.comparisons[0].evaluations]
    overview_rows: list[list[str]] = []
    for comparison in run.comparisons:
        row = [comparison.sample.sample_id, comparison.sample.expected_incident_type]
        for provider in provider_order:
            evaluation = next(
                (item for item in comparison.evaluations if item.mode == provider), None
            )
            if evaluation is None:
                row.append("—")
            else:
                status = "match" if evaluation.incident_type_match else "miss"
                row.append(f"{evaluation.report.incident_type} ({status})")
        overview_rows.append(row)

    summary_rows = [
        [
            summary.mode,
            f"{summary.incident_type_matches}/{summary.total_samples}",
            f"{summary.accuracy:.2f}",
            f"{summary.average_root_cause_quality:.2f}",
            f"{summary.average_action_quality:.2f}",
            f"{summary.total_duration_seconds:.2f}s",
            str(summary.total_llm_calls),
            str(summary.total_prompt_tokens + summary.total_completion_tokens),
        ]
        for summary in run.summaries
    ]

    lines = [
        "Provider Comparison Evaluation",
        f"Providers: {', '.join(provider_order)}",
        f"Total samples: {len(run.comparisons)}",
        f"Total runtime: {run.total_duration_seconds:.2f}s",
    ]
    if saved_results_path:
        lines.append(f"Results saved to: {saved_results_path}")

    lines.extend(["", "Sample overview"])
    lines.extend(_format_table(["sample", "expected", *provider_order], overview_rows))
    lines.extend(["", "Provider summary"])
    lines.extend(
        _format_table(
            ["provider", "matches", "accuracy", "root_cause_q", "action_q", "total_time", "llm_calls", "tokens"],
            summary_rows,
        )
    )

    if include_details:
        lines.append("")
        lines.append("Detailed results")
        for comparison in run.comparisons:
            lines.extend(_render_sample_details(comparison))

    return "\n".join(lines)
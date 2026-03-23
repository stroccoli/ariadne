from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from evals.reporting import compare_result_payloads, load_ab_test_results, render_comparison_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two saved Ariadne A/B benchmark result files.")
    parser.add_argument(
        "baseline",
        nargs="?",
        default=str(REPO_ROOT / "evals" / "results" / "previous.json"),
        help="Path to the baseline benchmark result JSON file. Defaults to evals/results/previous.json.",
    )
    parser.add_argument(
        "candidate",
        nargs="?",
        default=str(REPO_ROOT / "evals" / "results" / "latest.json"),
        help="Path to the candidate benchmark result JSON file. Defaults to evals/results/latest.json.",
    )
    parser.add_argument("--json", action="store_true", help="Print the comparison payload as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = compare_result_payloads(
        load_ab_test_results(Path(args.baseline)),
        load_ab_test_results(Path(args.candidate)),
    )
    if args.json:
        print(json.dumps(comparison, indent=2))
        return

    print(render_comparison_report(comparison))


if __name__ == "__main__":
    main()
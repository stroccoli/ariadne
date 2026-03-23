from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `evals.*` imports resolve
# when this script is invoked directly (e.g. `python evals/run_provider_test.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.ab_testing import (
    ab_test_to_dict,
    get_sample_by_id,
    list_sample_ids,
    render_sample_rubric,
    sample_rubric_to_dict,
)
from evals.provider_runner import SUPPORTED_PROVIDERS, run_provider_test
from evals.reporting import render_provider_test_report, save_provider_test_results
from ariadne.core.logging_config import configure_logging


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Ariadne provider comparison benchmark.",
    )
    parser.add_argument(
        "--providers",
        default=",".join(SUPPORTED_PROVIDERS),
        help="Comma-separated list of LLM providers to compare (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        default="detailed",
        choices=("detailed", "compact"),
        help="Prompt mode used for every provider (default: %(default)s).",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=10,
        help="Number of random samples to draw from the library (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sample selection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full benchmark payload as JSON.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist results under evals/results/provider.",
    )
    parser.add_argument(
        "--results-dir",
        help="Optional override for the directory where JSON result files are written.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available benchmark sample ids and exit.",
    )
    parser.add_argument(
        "--describe-sample",
        help="Print the rubric and logs for a specific benchmark sample id and exit.",
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    if args.list_samples:
        sample_ids = list(list_sample_ids())
        if args.json:
            print(json.dumps(sample_ids, indent=2))
            return
        print("\n".join(sample_ids))
        return

    if args.describe_sample:
        sample = get_sample_by_id(args.describe_sample)
        if args.json:
            print(json.dumps(sample_rubric_to_dict(sample), indent=2))
            return
        print(render_sample_rubric(sample))
        return

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if not providers:
        print("Error: no providers specified", file=sys.stderr)
        sys.exit(1)

    run = run_provider_test(
        providers=providers,
        mode=args.mode,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    results_dir = Path(args.results_dir) if args.results_dir else None
    saved_path = None if args.no_save else save_provider_test_results(run, results_dir=results_dir)

    if args.json:
        print(json.dumps(ab_test_to_dict(run, saved_results_path=saved_path), indent=2))
        return

    print(render_provider_test_report(run, saved_results_path=saved_path, include_details=False))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Detailed provider benchmark results\n%s",
            render_provider_test_report(run, saved_results_path=saved_path, include_details=True),
        )


if __name__ == "__main__":
    main()

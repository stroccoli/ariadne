from __future__ import annotations

import argparse
import json
import logging
import os

from ariadne.core.logging_config import configure_logging
from ariadne.core.models import ALLOWED_PROMPT_MODES
from ariadne.core.graph import run_graph
from ariadne.core.utils.output import build_run_summary


DEFAULT_LOGS = (
    "ERROR payment-service database connection timeout after 30s; retry attempts exhausted"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Ariadne incident analyzer.")
    parser.add_argument("logs", nargs="?", help="Raw incident logs to analyze.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="detailed",
        choices=sorted(ALLOWED_PROMPT_MODES),
        help="Prompt mode for single-incident analysis.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug output with structured execution recap.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"

    configure_logging()

    logs = args.logs or DEFAULT_LOGS
    mode = args.mode

    result = run_graph(logs, mode=mode)

    # --- Report ---
    print(json.dumps(result.final_output.model_dump(), indent=2))

    # --- Execution summary ---
    summary = build_run_summary(result)
    print("\n--- Execution Summary ---")
    print(json.dumps(summary, indent=2))

    # --- Debug recap ---
    if args.debug:
        print("\n--- Debug Recap ---")
        print(f"  Mode:                {mode}")
        print(f"  Incident type:       {summary['incident_type']}")
        print(f"  Confidence:          {summary['confidence']}")
        print(f"  Retrieval attempts:  {summary['retrieval_attempts']}")
        print(f"  LLM calls:           {summary['llm_calls']}")
        print(f"  Total latency:       {summary['total_latency_seconds']}s")
        print(f"  Token usage:         {summary['token_usage']}")
        print(f"  Node timings:        {summary['node_timings']}")
        if result.context:
            print(f"  Retrieved context:   {len(result.context)} document(s)")
            for i, doc in enumerate(result.context, 1):
                print(f"    [{i}] {doc[:120]}..." if len(doc) > 120 else f"    [{i}] {doc}")
        if result.analysis:
            print(f"  Root cause:          {result.analysis.root_cause}")
            print(f"  Actions:             {result.analysis.recommended_actions}")


if __name__ == "__main__":
    main()

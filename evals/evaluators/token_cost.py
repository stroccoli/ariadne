"""Token-count and cost LangSmith evaluators.

Surfaces prompt/completion token counts as numeric metrics and estimates
the equivalent cost under Gemini Flash pricing.

Pricing constants can be overridden via environment variables:
    GEMINI_FLASH_INPUT_PRICE_PER_M   (default: 0.10 USD / 1M tokens)
    GEMINI_FLASH_OUTPUT_PRICE_PER_M  (default: 0.40 USD / 1M tokens)
"""
from __future__ import annotations

import os

from langsmith.schemas import Example, Run

_GEMINI_FLASH_INPUT_PRICE_PER_M = float(os.getenv("GEMINI_FLASH_INPUT_PRICE_PER_M", "0.10"))
_GEMINI_FLASH_OUTPUT_PRICE_PER_M = float(os.getenv("GEMINI_FLASH_OUTPUT_PRICE_PER_M", "0.40"))


def eval_prompt_tokens(run: Run, example: Example) -> dict:
    """Surface prompt token count as a numeric metric in LangSmith."""
    tokens = (run.outputs or {}).get("prompt_tokens", 0) or 0
    return {"key": "prompt_tokens", "score": tokens}


def eval_completion_tokens(run: Run, example: Example) -> dict:
    """Surface completion token count as a numeric metric in LangSmith."""
    tokens = (run.outputs or {}).get("completion_tokens", 0) or 0
    return {"key": "completion_tokens", "score": tokens}


def eval_estimated_cost_gemini_flash(run: Run, example: Example) -> dict:
    """Estimate what this run would cost using Gemini Flash pricing (USD)."""
    outputs = run.outputs or {}
    prompt_tokens = outputs.get("prompt_tokens", 0) or 0
    completion_tokens = outputs.get("completion_tokens", 0) or 0
    cost = (
        prompt_tokens * _GEMINI_FLASH_INPUT_PRICE_PER_M
        + completion_tokens * _GEMINI_FLASH_OUTPUT_PRICE_PER_M
    ) / 1_000_000
    return {"key": "estimated_cost_gemini_flash_usd", "score": round(cost, 8)}

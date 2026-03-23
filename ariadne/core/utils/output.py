"""Output and parsing utilities.

Design choice:
- Keep JSON extraction in one place so both agents follow the same loose parsing rules.

Tradeoff:
- The parser is intentionally forgiving because LLMs often add wrappers around JSON.

Production caveat:
- This will still fail on malformed nested JSON or mixed natural-language responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


def extract_json_object(raw_response: str) -> str:
    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    if start == -1:
        return cleaned

    return cleaned[start:]


def coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default

    return max(0.0, min(1.0, confidence))


def parse_json_response(raw_response: str, fallback: dict) -> dict:
    candidate = extract_json_object(raw_response)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response: %s", error)
            return fallback

    if not isinstance(parsed, dict):
        logger.warning("Failed to parse JSON response: response was not a JSON object")
        return fallback

    return parsed


def build_run_summary(state: object) -> dict:
    """Build a unified execution summary from an IncidentState.

    Works for both single-run and eval modes. Accepts any object with the
    expected attributes so callers don't need to import IncidentState.
    """
    final = getattr(state, "final_output", None)
    node_timings = dict(getattr(state, "node_timings", {}))
    total_latency = node_timings.pop("total", None)

    prompt_tokens = getattr(state, "total_prompt_tokens", 0) or 0
    completion_tokens = getattr(state, "total_completion_tokens", 0) or 0
    total_tokens = prompt_tokens + completion_tokens if (prompt_tokens or completion_tokens) else 0

    return {
        "incident_type": getattr(final, "incident_type", None) if final else None,
        "confidence": round(getattr(final, "confidence", 0.0), 2) if final else None,
        "total_latency_seconds": total_latency,
        "retrieval_attempts": getattr(state, "retrieval_attempts", 0),
        "llm_calls": getattr(state, "total_llm_calls", 0),
        "token_usage": {
            "prompt_tokens": prompt_tokens or "unknown",
            "completion_tokens": completion_tokens or "unknown",
            "total_tokens": total_tokens or "unknown",
        },
        "node_timings": node_timings,
    }

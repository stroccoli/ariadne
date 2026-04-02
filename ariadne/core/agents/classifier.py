"""Incident classification agent.

Design choice:
- Keep prompt templates local to the agent because prompt versioning is part of the agent contract.

Tradeoff:
- This duplicates some prompting structure across agents, but keeps each module easy to read.

Production caveat:
- A single call with raw logs can misclassify noisy incidents without stronger guardrails or evaluation.
"""

from __future__ import annotations

import logging

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore[misc]
        """No-op fallback when langsmith is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn
from pydantic import ValidationError

from ariadne.core.config import get_llm_client
from ariadne.core.integrations.llm.base import LLMResponse
from ariadne.core.models import ALLOWED_PROMPT_MODES, ClassificationOutput
from ariadne.core.state import IncidentState
from ariadne.core.utils.logs import truncate_logs
from ariadne.core.utils.output import parse_json_response


logger = logging.getLogger(__name__)

DETAILED_PROMPT = """You are a strict incident classifier.

Classify the provided logs into exactly one incident type from this allowed set:
- timeout
- dependency_failure
- database_issue
- memory_issue
- unknown

Return exactly one JSON object with this shape:
{{
  "incident_type": "<one allowed type>",
  "confidence": <number between 0 and 1>
}}

Rules:
1. Use only evidence explicitly present in the logs.
2. Do not invent causes, systems, or context that are not stated in the logs.
3. If the evidence is weak, ambiguous, conflicting, or does not clearly match a known type, use "unknown".
4. "incident_type" must be exactly one of:
   "timeout", "dependency_failure", "database_issue", "memory_issue", "unknown"
5. "confidence" must be a numeric value from 0 to 1.
6. Output JSON only. No markdown. No prose. No extra keys.

Classification guidance:
- Use "timeout" when the dominant signal is requests or operations exceeding time limits, and there is no stronger explicit evidence for another category.
- Use "dependency_failure" when the logs indicate a downstream service, external API, or dependency is unavailable, failing, refusing connections, or returning errors.
- Use "database_issue" when the logs explicitly indicate database-related failures such as query errors, deadlocks, schema errors, database connection failures, or connection pool exhaustion tied to the database.
- Use "memory_issue" when the logs indicate out-of-memory conditions, heap exhaustion, memory pressure, memory leaks, or restarts caused by memory exhaustion.
- Use "unknown" when none of the above is clearly supported.

Tie-breaking rules:
- Prefer "database_issue" over "dependency_failure" if the failing dependency is explicitly a database.
- Prefer "memory_issue" if memory exhaustion is explicit, even if secondary timeouts also appear.
- Prefer "unknown" over a weak guess.

Confidence guidance:
- Use higher confidence only when the logs contain direct, repeated, and consistent evidence.
- Use medium confidence when the pattern is plausible but not explicit.
- Use low confidence when evidence is sparse or mixed.

Logs:
{logs}"""

COMPACT_PROMPT = """Classify these logs into one incident type: timeout, dependency_failure, database_issue, memory_issue, or unknown.

Return JSON only:
{{"incident_type": "timeout|dependency_failure|database_issue|memory_issue|unknown", "confidence": 0.0}}

Rules:
- Use only explicit evidence from the logs.
- Prefer unknown over guessing.
- Return exactly one JSON object.
- No markdown, comments, or extra text.

Logs:
{logs}"""

def _build_prompt(logs: str, mode: str) -> str:
  prompt_template = DETAILED_PROMPT if mode == "detailed" else COMPACT_PROMPT
  if mode not in ALLOWED_PROMPT_MODES:
    logger.warning("Unknown classification mode '%s', defaulting to detailed", mode)
    prompt_template = DETAILED_PROMPT

  return prompt_template.format(logs=truncate_logs(logs))


def _fallback_response() -> ClassificationOutput:
  return ClassificationOutput(incident_type="unknown", confidence=0.0)


@traceable(run_type="chain", name="classifier")
def classify(logs: str, mode: str = "detailed") -> tuple[ClassificationOutput, dict]:
  prompt = _build_prompt(logs, mode)
  logger.debug("Classifier prompt: %s", prompt)
  fallback = _fallback_response()
  token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}

  try:
    llm_response: LLMResponse = get_llm_client().generate(prompt, json_output=True)
  except Exception as error:
    logger.error("Classifier LLM call failed: %s", error)
    return fallback, token_stats

  raw_response = llm_response.text
  token_stats["llm_calls"] = 1
  token_stats["prompt_tokens"] = llm_response.prompt_tokens or 0
  token_stats["completion_tokens"] = llm_response.completion_tokens or 0

  logger.debug("Classifier raw response: %s", raw_response)

  parsed = parse_json_response(raw_response, fallback.model_dump())
  merged = {**fallback.model_dump(), **parsed}

  try:
    result = ClassificationOutput.model_validate(merged)
  except ValidationError as error:
    logger.warning("Classifier output validation failed: %s", error)
    result = fallback

  logger.debug("Classifier parsed output: %s", result.model_dump())
  return result, token_stats


def run_classifier(state: IncidentState) -> IncidentState:
    """Run classification on the given state and return the updated state."""
    result, token_stats = classify(state.logs, mode=state.mode)
    state.incident_type = result.incident_type
    state.classification_confidence = result.confidence
    state.total_prompt_tokens += token_stats["prompt_tokens"]
    state.total_completion_tokens += token_stats["completion_tokens"]
    state.total_llm_calls += token_stats["llm_calls"]
    return state

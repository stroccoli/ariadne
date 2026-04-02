"""Incident analysis agent.

Design choice:
- Keep analysis focused on logs plus a lightweight context string so the interface stays stable for future RAG work.

Tradeoff:
- A plain string context is easy to compose, but loses structure that could improve downstream reasoning.

Production caveat:
- This approach can break when context grows large or contains conflicting hints with no ranking.
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
from ariadne.core.models import ALLOWED_PROMPT_MODES, AnalysisOutput
from ariadne.core.state import IncidentState
from ariadne.core.utils.logs import truncate_logs
from ariadne.core.utils.output import parse_json_response


logger = logging.getLogger(__name__)

DETAILED_PROMPT = """You are an incident analysis assistant for engineering responders.

Your job is to infer the most likely cause of the incident from the evidence provided.

You will receive:
- Logs: raw operational log lines
- Context: additional incident context, which may include a predicted incident type, service metadata, human notes, or retrieval results

Inputs

LOGS
{logs}

CONTEXT
{context}

Task

Return exactly one JSON object with this shape:
{{
    "root_cause": "string",
    "recommended_actions": ["string", "string"],
    "confidence": 0.0
}}

Required behavior

1. Base the analysis only on the provided logs and context.
2. Treat any predicted incident type in the context as a hint, not as ground truth.
3. Do not invent infrastructure details, services, dependencies, deployments, regions, ownership, or failure mechanisms that are not directly supported by the evidence.
4. Prefer a narrow, evidence-backed explanation over a broad or generic one.
5. If the evidence is incomplete, mixed, or contradictory, say so explicitly in root_cause and lower the confidence.
6. Do not output any fields other than root_cause, recommended_actions, and confidence.

Reasoning rules

- First identify the concrete signals visible in the logs and context:
  repeated errors, affected component names, timing patterns, retries, saturation signals, restarts, dependency failures, or resource exhaustion.
- Then infer the single most likely cause that best explains those signals.
- If multiple causes are possible and the evidence does not clearly distinguish them, state the most supported hypothesis while making the uncertainty explicit.
- Do not rely on generic incident language such as "system issue", "service degradation", or "investigate further" unless tied to a specific observed signal.

Output rules

- root_cause must be one concise sentence.
- root_cause must name the likely failure mechanism and anchor it to observed evidence.
- recommended_actions must be a short ordered list of concrete next steps.
- Each recommended action must either:
  a) validate the hypothesis, or
  b) mitigate the likely issue
- Recommended actions must be specific to the evidence. Avoid generic advice like "check logs", "monitor the system", or "investigate the issue".
- confidence must be a number between 0 and 1.

Confidence guidance

- Use higher confidence only when the logs and context show a consistent, repeated, and specific pattern.
- Use medium confidence when the hypothesis fits the evidence but competing explanations remain plausible.
- Use low confidence when the evidence is sparse, indirect, or conflicting.

Final instruction

Return only the JSON object. No markdown. No explanation outside the object."""

COMPACT_PROMPT = """Analyze these incident logs using the provided context.

Return JSON only:
{{"root_cause": "string", "recommended_actions": ["string", "string"], "confidence": 0.0}}

Rules:
- Use only the evidence provided.
- Treat predicted incident type as a hint, not fact.
- root_cause must be one concise sentence.
- If the evidence is mixed, say the cause is uncertain.
- recommended_actions must be concrete next steps, not generic advice.
- confidence is required and must always be present.
- confidence must be a number between 0 and 1.
- If the evidence is weak or mixed, return a low confidence such as 0.2 to 0.5 instead of omitting the field.
- Do not return null, an empty value, or a missing confidence field.
- Before answering, verify the JSON includes exactly these three keys: root_cause, recommended_actions, confidence.
- Return exactly one JSON object with no markdown, comments, or extra text.

LOGS
{logs}

CONTEXT
{context}"""

CONFIDENCE_REPAIR_PROMPT = """Estimate confidence for the following incident analysis.

Return JSON only:
{{"confidence": 0.0}}

Rules:
- confidence must be a number between 0 and 1.
- Use higher confidence only when the logs and context strongly support the root cause.
- Use lower confidence when the evidence is mixed, sparse, or indirect.
- Return exactly one JSON object with no markdown, comments, or extra text.

LOGS
{logs}

CONTEXT
{context}

ANALYSIS
Root cause: {root_cause}
Recommended actions: {recommended_actions}
"""


def _build_prompt(logs: str, context: str, mode: str) -> str:
    prompt_template = DETAILED_PROMPT if mode == "detailed" else COMPACT_PROMPT
    if mode not in ALLOWED_PROMPT_MODES:
        logger.warning("Unknown analysis mode '%s', defaulting to detailed", mode)
        prompt_template = DETAILED_PROMPT

    return prompt_template.format(logs=truncate_logs(logs), context=context)


def _fallback_response() -> AnalysisOutput:
    return AnalysisOutput(
        root_cause="Unable to determine the root cause from the available evidence.",
        recommended_actions=[],
        confidence=0.0,
    )


def _repair_missing_confidence(
    logs: str,
    context: str,
    parsed: dict,
) -> tuple[float, dict]:
    """Return (confidence, token_stats) after a repair LLM call."""
    logger.warning("Analyzer response omitted confidence; requesting confidence repair")
    prompt = CONFIDENCE_REPAIR_PROMPT.format(
        logs=logs,
        context=context,
        root_cause=parsed.get("root_cause", ""),
        recommended_actions=parsed.get("recommended_actions", []),
    )
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}

    try:
        llm_response: LLMResponse = get_llm_client().generate(prompt, json_output=True)
    except Exception as error:
        logger.warning("Analyzer confidence repair failed during LLM call: %s", error)
        return 0.0, token_stats

    token_stats["llm_calls"] = 1
    token_stats["prompt_tokens"] = llm_response.prompt_tokens or 0
    token_stats["completion_tokens"] = llm_response.completion_tokens or 0

    logger.debug("Analyzer confidence repair raw response: %s", llm_response.text)
    repaired = parse_json_response(llm_response.text, {"confidence": 0.0})
    return float(repaired.get("confidence", 0.0)), token_stats


@traceable(run_type="chain", name="analyzer")
def analyze(logs: str, context: str, mode: str = "detailed") -> tuple[AnalysisOutput, dict]:
    prompt = _build_prompt(logs, context, mode)
    logger.debug("Analyzer prompt: %s", prompt)
    fallback = _fallback_response()
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}

    try:
        llm_response: LLMResponse = get_llm_client().generate(prompt, json_output=True)
    except Exception as error:
        logger.error("Analyzer LLM call failed: %s", error)
        return fallback, token_stats

    raw_response = llm_response.text
    token_stats["llm_calls"] = 1
    token_stats["prompt_tokens"] = llm_response.prompt_tokens or 0
    token_stats["completion_tokens"] = llm_response.completion_tokens or 0

    logger.debug("Analyzer raw response: %s", raw_response)

    parsed = parse_json_response(raw_response, fallback.model_dump())
    if "confidence" not in parsed and parsed.get("root_cause"):
        repaired_confidence, repair_stats = _repair_missing_confidence(logs, context, parsed)
        parsed["confidence"] = repaired_confidence
        token_stats["llm_calls"] += repair_stats["llm_calls"]
        token_stats["prompt_tokens"] += repair_stats["prompt_tokens"]
        token_stats["completion_tokens"] += repair_stats["completion_tokens"]
    merged = {**fallback.model_dump(), **parsed}

    try:
        result = AnalysisOutput.model_validate(merged)
    except ValidationError as error:
        logger.warning("Analyzer output validation failed: %s", error)
        result = fallback

    logger.debug("Analyzer parsed output: %s", result.model_dump())
    return result, token_stats


def _build_analysis_context(state: IncidentState) -> str:
    """Build the context string the analyzer prompt expects from current state."""
    if state.context:
        retrieved = "\n\n".join(
            f"[Context {i}] {doc}" for i, doc in enumerate(state.context, start=1)
        )
    else:
        retrieved = "No additional context available beyond the current logs."

    parts = [f"Retrieved context: {retrieved}"]
    if state.incident_type is not None:
        parts.append(f"Predicted incident type: {state.incident_type}")
    if state.classification_confidence is not None:
        parts.append(f"Classification confidence: {state.classification_confidence}")
    return "\n".join(parts)


def run_analyzer(state: IncidentState) -> IncidentState:
    """Run analysis on the given state and return the updated state."""
    context = _build_analysis_context(state)
    result, token_stats = analyze(state.logs, context, mode=state.mode)
    state.analysis = result
    state.total_prompt_tokens += token_stats["prompt_tokens"]
    state.total_completion_tokens += token_stats["completion_tokens"]
    state.total_llm_calls += token_stats["llm_calls"]
    return state

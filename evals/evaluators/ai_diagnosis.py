"""AI-synthesis meta-evaluator: interprets all metric scores via LLM.

Produces a human-readable ``comment`` (no numeric score) summarising the
pipeline's performance on a sample and suggesting concrete improvements.

The LLM used is controlled by EVAL_LLM_PROVIDER (same as ragas_metrics):
    ollama (default) → EVAL_OLLAMA_MODEL via /api/generate
    openai           → OPENAI_MODEL via chat completions
"""
from __future__ import annotations

import os

import requests
from langsmith.schemas import Example, Run

from evals.evaluators.rubric_evals import _get_sample_from_example
from evals.rubric_scoring import score_action, score_root_cause


def _call_eval_llm(prompt: str) -> str:
    """Synchronous LLM call using the eval provider (openai or ollama)."""
    eval_provider = os.getenv("EVAL_LLM_PROVIDER", "ollama").lower()
    if eval_provider == "openai":
        import openai
        client = openai.OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()
    else:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("EVAL_OLLAMA_MODEL", "deepseek-r1:8b").strip()
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


def eval_ai_diagnosis(run: Run, example: Example) -> dict:
    """LLM-synthesized interpretation of all metrics with concrete actions to take.

    Computes rubric scores inline so no dependency on other evaluator execution
    order. Returns the LLM response as a ``comment`` (no numeric score).
    """
    outputs = run.outputs or {}
    meta = example.metadata or {}
    sample_id = meta.get("sample_id", "unknown")

    expected_type = meta.get("expected_incident_type", "unknown")
    predicted_type = outputs.get("incident_type", "unknown")
    confidence = outputs.get("confidence")
    prompt_tokens = outputs.get("prompt_tokens", 0) or 0
    completion_tokens = outputs.get("completion_tokens", 0) or 0

    root_cause_score: float | None = None
    action_score: float | None = None
    sample = _get_sample_from_example(example)
    if sample is not None:
        root_cause_score = score_root_cause(outputs.get("response", ""), sample.root_cause_rubric)
        action_score = score_action(outputs.get("recommended_actions") or [], sample.action_rubric)

    rc_str = f"{root_cause_score:.2f}" if root_cause_score is not None else "N/A"
    act_str = f"{action_score:.2f}" if action_score is not None else "N/A"
    type_match = predicted_type == expected_type

    diagnosis_prompt = (
        "You are an AI evaluation assistant reviewing an incident analysis pipeline run.\n\n"
        f"Sample: {sample_id}\n"
        f"Expected incident type: {expected_type}\n"
        f"Predicted incident type: {predicted_type} (correct: {type_match})\n"
        f"Model confidence: {confidence}\n"
        f"Root cause quality score: {rc_str}/1.0\n"
        f"Action quality score: {act_str}/1.0\n"
        f"Token usage: {prompt_tokens} prompt + {completion_tokens} completion\n\n"
        "Respond with:\n"
        "1. A 1-2 sentence interpretation of the model's performance on this sample.\n"
        "2. 2-3 specific, actionable improvements (prompt, retrieval, or rubric changes).\n"
        "Be concise and technical. Do not repeat the numbers above verbatim."
    )

    try:
        comment = _call_eval_llm(diagnosis_prompt)
    except Exception as exc:  # noqa: BLE001
        comment = f"[eval_ai_diagnosis failed: {exc}]"

    return {"key": "ai_diagnosis", "score": None, "comment": comment}

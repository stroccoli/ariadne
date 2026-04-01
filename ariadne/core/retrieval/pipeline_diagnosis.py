"""LLM-driven pipeline diagnosis.

Reads a PipelineHealthReport and asks the configured LLM to produce a
structured human-readable diagnosis: status, summary, root causes with
metric evidence, operational impact, and recommended actions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ariadne.core.integrations.llm.base import LLMClient
from ariadne.core.retrieval.pipeline_report import PipelineHealthReport
from ariadne.core.utils.output import parse_json_response


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status levels
# ---------------------------------------------------------------------------

CRITICAL = "CRITICAL"
WARNING = "WARNING"
HEALTHY = "HEALTHY"

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RootCause:
    """A single diagnosed root cause with the metric evidence that supports it."""

    cause: str
    evidence: str  # e.g. "total_chunks_indexed=0, collection_vector_count=0"

    def to_dict(self) -> dict[str, str]:
        return {"cause": self.cause, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RootCause":
        return cls(cause=d.get("cause", ""), evidence=d.get("evidence", ""))


@dataclass
class PipelineDiagnosis:
    """Full LLM diagnosis of one pipeline run."""

    status: str = HEALTHY                       # CRITICAL | WARNING | HEALTHY
    summary: str = ""
    root_causes: list[RootCause] = field(default_factory=list)
    impact: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    reasoning: str = ""
    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "impact": self.impact,
            "recommended_actions": self.recommended_actions,
            "reasoning": self.reasoning,
            "analyzed_at": self.analyzed_at,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineDiagnosis":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            status=data.get("status", HEALTHY),
            summary=data.get("summary", ""),
            root_causes=[RootCause.from_dict(rc) for rc in data.get("root_causes", [])],
            impact=data.get("impact", []),
            recommended_actions=data.get("recommended_actions", []),
            reasoning=data.get("reasoning", ""),
            analyzed_at=data.get("analyzed_at", ""),
        )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a senior SRE analyzing an automated data pipeline health report.
Your job is to diagnose the state of the pipeline and explain what is wrong
(or confirm it is healthy), in terms a developer can act on immediately.

─────────────────────────────────────────
METRIC GLOSSARY
─────────────────────────────────────────
Performance:
  preprocess_duration_seconds       – wall time for the clean/dedup stage
  chunk_duration_seconds            – wall time for the chunking stage
  index_duration_seconds            – wall time to embed + upsert into Qdrant
  index_throughput_docs_per_sec     – docs upserted per second during indexing
  embedding_batches_total           – number of batches sent to the embedding model
  total_embedding_tokens_estimated  – total tokens estimated across all batches
  total_upsert_retries              – number of batch retries due to transient errors

Data quality:
  total_raw_docs         – documents collected before any filtering
  total_clean_docs       – documents after dedup and quality filters
  total_chunks           – chunks produced by the chunking stage
  total_chunks_indexed   – chunks successfully upserted into the vector index
                           ⚠ CRITICAL if 0 when total_chunks > 0
  extraction_error_rate  – fraction of raw docs rejected (error or dedup)
                           ⚠ WARNING if > 0.10
  duplicate_ratio        – fraction removed by exact-hash deduplication
  semantic_duplicate_ratio – fraction removed by near-duplicate Jaccard dedup
  null_vector_count      – vectors whose L2 norm ≈ 0 (embedding failed silently)
                           ⚠ WARNING if > 0
  malformed_payload_count – chunks with empty/missing document field
                           ⚠ WARNING if > 0
  upsert_error_count     – batches that permanently failed after all retries
                           ⚠ CRITICAL if > 0
  partial_failure        – true if any batch failed permanently
                           ⚠ CRITICAL if true

Vector health:
  collection_vector_count – actual vector count in the Qdrant collection
                           ⚠ CRITICAL if 0 when total_chunks > 0
  embedding_dim          – dimensionality of the stored vectors (0 = unknown)
                           ⚠ WARNING if 0
  index_fill_ratio       – collection_vector_count / total_chunks
                           ⚠ WARNING if < 0.95 and total_chunks > 0
  vector_norm_mean       – mean L2 norm of sampled vectors (should be close to 1 for
                           normalized embeddings, close to constant for unnormalized)
  vector_norm_std        – std dev of L2 norms (high std may indicate embedding drift)
  near_zero_vector_count – vectors with ||v|| < 0.01
                           ⚠ WARNING if > 0
  norm_drift_from_previous – |current_mean - prev_mean| / prev_mean
                           ⚠ WARNING if > 0.10 (potential embedding model change)

Corpus distribution:
  chunks_by_source       – breakdown of chunks per data source
  chunks_by_severity     – breakdown by incident severity label
  unique_services        – number of distinct services in the corpus
  avg_chunks_per_doc     – average chunks produced per document

embedding_model:
  "unknown"              – ⚠ WARNING: the embedding model tag was not recorded;
                           model mismatch between index and query time cannot be
                           detected

─────────────────────────────────────────
STATUS RULES (apply in order, first match wins)
─────────────────────────────────────────
CRITICAL – any of:
  • total_chunks_indexed = 0 AND total_chunks > 0
  • collection_vector_count = 0 AND total_chunks > 0
  • partial_failure = true
  • upsert_error_count > 0

WARNING – any of:
  • extraction_error_rate > 0.10
  • index_fill_ratio < 0.95 AND total_chunks > 0
  • near_zero_vector_count > 0
  • null_vector_count > 0
  • malformed_payload_count > 0
  • norm_drift_from_previous > 0.10
  • embedding_dim = 0
  • embedding_model = "unknown"

HEALTHY – none of the above apply.

─────────────────────────────────────────
PIPELINE REPORT (JSON)
─────────────────────────────────────────
{report_json}

─────────────────────────────────────────
INSTRUCTIONS
─────────────────────────────────────────
Analyze the report above using the glossary and status rules.

Return exactly one JSON object with this shape (no markdown, no prose outside JSON):

{{
  "status": "CRITICAL" | "WARNING" | "HEALTHY",
  "summary": "<one sentence — what is the overall state and the single most important finding>",
  "root_causes": [
    {{
      "cause": "<concise name of the root cause>",
      "evidence": "<which exact metric(s) and value(s) expose this — e.g. total_chunks_indexed=0, collection_vector_count=0>"
    }}
  ],
  "impact": [
    "<what will break or degrade at runtime as a result of each root cause>"
  ],
  "recommended_actions": [
    "<concrete, actionable step to fix the issue — specific enough to execute without guessing>"
  ],
  "reasoning": "<step-by-step explanation of how you read the metrics, which thresholds were triggered, and why you assigned the given status — cite specific metric=value pairs>"
}}

Rules:
1. root_causes must be empty ([]) for HEALTHY status.
2. Each root_cause.evidence must quote at least one metric name and its value from the report.
3. recommended_actions must be actionable immediately (no vague advice).
4. reasoning must justify the status by walking through the triggered thresholds.
5. Output JSON only. No markdown fences. No extra keys.
"""


# ---------------------------------------------------------------------------
# Deterministic status check (guardrail against LLM hallucination)
# ---------------------------------------------------------------------------


def _compute_deterministic_status(report: "PipelineHealthReport") -> str:
    """Apply status rules from the prompt deterministically, ignoring the LLM.

    This is used as a post-LLM guardrail: the LLM provides the narrative, but
    the final *status* is always authoritative based on the actual metric values.
    """
    dq = report.data_quality
    vh = report.vector_health

    # CRITICAL rules (first match wins)
    if dq.total_chunks > 0 and dq.total_chunks_indexed == 0:
        return CRITICAL
    if dq.total_chunks > 0 and vh.collection_vector_count == 0:
        return CRITICAL
    if dq.partial_failure:
        return CRITICAL
    if dq.upsert_error_count > 0:
        return CRITICAL

    # WARNING rules
    if dq.extraction_error_rate > 0.10:
        return WARNING
    if dq.total_chunks > 0 and vh.index_fill_ratio < 0.95:
        return WARNING
    if vh.near_zero_vector_count > 0:
        return WARNING
    if dq.null_vector_count > 0:
        return WARNING
    if dq.malformed_payload_count > 0:
        return WARNING
    if vh.norm_drift_from_previous is not None and vh.norm_drift_from_previous > 0.10:
        return WARNING
    if vh.embedding_dim == 0:
        return WARNING
    if report.embedding_model == "unknown":
        return WARNING

    return HEALTHY


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def diagnose_pipeline(
    report: PipelineHealthReport,
    llm_client: LLMClient,
) -> PipelineDiagnosis:
    """Call the LLM to produce a structured diagnosis of *report*.

    Raises ``ValueError`` if the LLM returns an unparseable response after
    the parse fallback is exhausted.
    """
    report_json = json.dumps(report.to_dict(), indent=2)
    prompt = _PROMPT_TEMPLATE.format(report_json=report_json)

    logger.info("Sending pipeline report to LLM for diagnosis...")
    response = llm_client.generate(prompt, json_output=True)

    fallback: dict[str, Any] = {
        "status": CRITICAL,
        "summary": "LLM returned an unparseable response — manual review required.",
        "root_causes": [],
        "impact": ["Cannot determine impact automatically."],
        "recommended_actions": ["Inspect pipeline_report.json manually."],
        "reasoning": f"Raw LLM output: {response.text[:500]}",
    }

    parsed = parse_json_response(response.text, fallback=fallback)

    # Normalise root_causes: accept both list[str] and list[dict]
    raw_rcs = parsed.get("root_causes", [])
    root_causes: list[RootCause] = []
    for rc in raw_rcs:
        if isinstance(rc, str):
            root_causes.append(RootCause(cause=rc, evidence=""))
        elif isinstance(rc, dict):
            root_causes.append(RootCause.from_dict(rc))

    diagnosis = PipelineDiagnosis(
        status=parsed.get("status", CRITICAL),
        summary=parsed.get("summary", ""),
        root_causes=root_causes,
        impact=parsed.get("impact", []),
        recommended_actions=parsed.get("recommended_actions", []),
        reasoning=parsed.get("reasoning", ""),
    )

    # --- Guardrail: override LLM status with deterministic rule evaluation ---
    # LLMs can hallucinate metric values from the prompt; the authoritative
    # status is always computed from the actual report object.
    authoritative_status = _compute_deterministic_status(report)
    if diagnosis.status != authoritative_status:
        llm_status = diagnosis.status  # capture before overwriting
        logger.warning(
            "LLM status '%s' overridden to '%s' based on deterministic rule check "
            "(LLM may have hallucinated metric values).",
            llm_status,
            authoritative_status,
        )
        diagnosis.status = authoritative_status
        # Clear LLM narrative that was based on fabricated metric values
        diagnosis.root_causes = []
        diagnosis.impact = []
        diagnosis.recommended_actions = []
        # Annotate the LLM reasoning so it's clear it was overridden
        diagnosis.reasoning = (
            f"[OVERRIDE] LLM assigned '{llm_status}' but deterministic rule "
            f"evaluation produced '{authoritative_status}'. LLM original reasoning: "
            + diagnosis.reasoning
        )

        # Build minimal deterministic narrative for the new status
        dq = report.data_quality
        vh = report.vector_health
        if authoritative_status == HEALTHY:
            diagnosis.summary = "Pipeline is healthy — all metric thresholds are within normal bounds."
        elif authoritative_status == WARNING:
            warnings: list[str] = []
            if report.embedding_model == "unknown":
                warnings.append("embedding_model=unknown")
                diagnosis.root_causes.append(RootCause(
                    cause="Embedding model not recorded",
                    evidence="embedding_model=unknown",
                ))
                diagnosis.impact.append(
                    "Model mismatch between indexing and query time cannot be detected."
                )
                diagnosis.recommended_actions.append(
                    "Set the EMBEDDING_PROVIDER environment variable before running the pipeline."
                )
            if vh.embedding_dim == 0:
                warnings.append("embedding_dim=0")
                diagnosis.root_causes.append(RootCause(
                    cause="Embedding dimension not recorded",
                    evidence="embedding_dim=0",
                ))
            if dq.total_chunks > 0 and vh.index_fill_ratio < 0.95:
                warnings.append(f"index_fill_ratio={vh.index_fill_ratio:.2f}")
                diagnosis.root_causes.append(RootCause(
                    cause="Incomplete indexing",
                    evidence=f"index_fill_ratio={vh.index_fill_ratio:.2f}, total_chunks={dq.total_chunks}",
                ))
            if dq.extraction_error_rate > 0.10:
                warnings.append(f"extraction_error_rate={dq.extraction_error_rate:.2%}")
                diagnosis.root_causes.append(RootCause(
                    cause="High extraction error rate",
                    evidence=f"extraction_error_rate={dq.extraction_error_rate:.2%}",
                ))
            diagnosis.summary = (
                f"Pipeline completed with warnings: {', '.join(warnings)}. "
                "No indexing failures detected."
            )
        elif authoritative_status == CRITICAL:
            crit_reasons: list[str] = []
            if dq.total_chunks > 0 and dq.total_chunks_indexed == 0:
                crit_reasons.append(f"total_chunks_indexed=0 (total_chunks={dq.total_chunks})")
                diagnosis.root_causes.append(RootCause(
                    cause="Indexing failure: no chunks indexed",
                    evidence=f"total_chunks_indexed=0, total_chunks={dq.total_chunks}",
                ))
            if dq.total_chunks > 0 and vh.collection_vector_count == 0:
                crit_reasons.append("collection_vector_count=0")
                diagnosis.root_causes.append(RootCause(
                    cause="Empty vector collection",
                    evidence=f"collection_vector_count=0, total_chunks={dq.total_chunks}",
                ))
            if dq.partial_failure:
                crit_reasons.append("partial_failure=true")
                diagnosis.root_causes.append(RootCause(
                    cause="Partial upsert failure",
                    evidence=f"partial_failure=true, upsert_error_count={dq.upsert_error_count}",
                ))
            diagnosis.summary = f"Pipeline CRITICAL: {'; '.join(crit_reasons)}."

    logger.info("Diagnosis complete — status: %s", diagnosis.status)
    return diagnosis

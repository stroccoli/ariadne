"use client";

import { useState } from "react";
import type { AnalyzeResponse } from "@/lib/api";
import {
  INCIDENT_TYPE_LABELS,
  getConfidenceLevel,
  getConfidenceLabel,
} from "@/lib/constants";

interface AnalysisResultProps {
  result: AnalyzeResponse;
  onReset: () => void;
}

function formatTime(nodeTimings: Record<string, number>): string {
  const total = Object.values(nodeTimings).reduce((sum, t) => sum + t, 0);
  if (total < 1) return `${Math.round(total * 1000)}ms`;
  return `${total.toFixed(1)}s`;
}

export default function AnalysisResult({
  result,
  onReset,
}: AnalysisResultProps) {
  const [copied, setCopied] = useState(false);
  const level = getConfidenceLevel(result.confidence);
  const pct = Math.round(result.confidence * 100);

  const confidenceColorClass =
    level === "high"
      ? "bg-confidence-high"
      : level === "mid"
        ? "bg-confidence-mid"
        : "bg-confidence-low";

  const confidenceTextClass =
    level === "high"
      ? "text-confidence-high"
      : level === "mid"
        ? "text-confidence-mid"
        : "text-confidence-low";

  async function handleCopy() {
    const text = [
      `Incident Type: ${INCIDENT_TYPE_LABELS[result.incident_type]}`,
      `Confidence: ${pct}% (${getConfidenceLabel(result.confidence)})`,
      "",
      `Root Cause:`,
      result.root_cause,
      "",
      `Recommended Actions:`,
      ...result.recommended_actions.map((a, i) => `${i + 1}. ${a}`),
      "",
      `---`,
      `Retrieval attempts: ${result.metadata.retrieval_attempts}`,
      `LLM calls: ${result.metadata.llm_calls}`,
      `Processing time: ${formatTime(result.metadata.node_timings)}`,
      `Tokens used: ${result.metadata.usage.total_tokens}`,
    ].join("\n");

    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  }

  return (
    <div className="animate-fade-in space-y-6">
      {/* Incident type badge + Confidence */}
      <div className="flex flex-wrap items-center gap-4">
        <span className="inline-flex items-center rounded-full bg-accent/15 px-3 py-1 text-sm font-semibold text-accent">
          {INCIDENT_TYPE_LABELS[result.incident_type]}
        </span>

        <div className="flex items-center gap-3">
          {/* Confidence bar */}
          <div className="h-2 w-32 overflow-hidden rounded-full bg-border">
            <div
              className={`h-full rounded-full animate-grow-width ${confidenceColorClass}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className={`text-sm font-semibold ${confidenceTextClass}`}>
            {pct}%
          </span>
          <span className="text-xs text-secondary">
            {getConfidenceLabel(result.confidence)}
          </span>
        </div>
      </div>

      {/* Root cause card */}
      <div className="rounded-lg border-l-[3px] border-l-gold bg-surface p-5">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-secondary">
          Root Cause
        </h3>
        <p className="text-sm leading-relaxed text-primary">
          {result.root_cause}
        </p>
      </div>

      {/* Recommended actions */}
      {result.recommended_actions.length > 0 && (
        <div>
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-secondary">
            Recommended Actions
          </h3>
          <ol className="space-y-2">
            {result.recommended_actions.map((action, i) => (
              <li key={i} className="flex items-start gap-3">
                {/* Chevron icon */}
                <svg
                  className="mt-0.5 h-4 w-4 flex-shrink-0 text-gold"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                <span className="text-sm text-primary">{action}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Metadata footer */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-border pt-4 text-xs text-secondary">
        <span>{result.metadata.llm_calls} LLM calls</span>
        <span className="text-muted">·</span>
        <span>{formatTime(result.metadata.node_timings)}</span>
        <span className="text-muted">·</span>
        <span>
          {result.metadata.usage.total_tokens.toLocaleString()} tokens
        </span>
        {result.metadata.retrieval_attempts > 1 && (
          <>
            <span className="text-muted">·</span>
            <span className="inline-flex items-center rounded bg-gold-muted px-1.5 py-0.5 text-xs text-gold">
              Retried
            </span>
          </>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-3">
        <button
          onClick={onReset}
          className="rounded-lg bg-gold px-5 py-2.5 text-sm font-semibold text-bg-primary transition-colors hover:bg-gold-light"
        >
          New Analysis
        </button>
        <button
          onClick={handleCopy}
          className="rounded-lg border border-gold/40 px-4 py-2.5 text-sm font-medium text-gold transition-colors hover:bg-gold/10"
        >
          {copied ? (
            <span className="flex items-center gap-1.5">
              <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              Copied
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <svg
                className="h-4 w-4"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <rect x="6" y="6" width="10" height="10" rx="1.5" />
                <path d="M4 14V4.5A.5.5 0 014.5 4H14" />
              </svg>
              Copy result
            </span>
          )}
        </button>
      </div>
    </div>
  );
}

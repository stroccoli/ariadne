"use client";

import { useState } from "react";
import type { PromptMode } from "@/lib/api";
import { SAMPLE_LOGS, MAX_LOG_LENGTH } from "@/lib/constants";

interface LogInputProps {
  onSubmit: (logs: string, mode: PromptMode) => void;
  isLoading: boolean;
}

export default function LogInput({ onSubmit, isLoading }: LogInputProps) {
  const [logs, setLogs] = useState("");
  const [mode, setMode] = useState<PromptMode>("detailed");
  const [validationError, setValidationError] = useState<string | null>(null);

  function handleSubmit() {
    const trimmed = logs.trim();
    if (!trimmed) {
      setValidationError("Paste some logs to analyze");
      return;
    }
    if (trimmed.length > MAX_LOG_LENGTH) {
      setValidationError(
        `Logs exceed maximum length (${MAX_LOG_LENGTH.toLocaleString()} characters)`,
      );
      return;
    }
    setValidationError(null);
    onSubmit(trimmed, mode);
  }

  function handleLoadExample() {
    setLogs(SAMPLE_LOGS);
    setValidationError(null);
  }

  function handleLogsChange(value: string) {
    setLogs(value);
    if (validationError) setValidationError(null);
  }

  return (
    <div className="space-y-4">
      {/* Textarea — terminal-style */}
      <div className="relative">
        <textarea
          value={logs}
          onChange={(e) => handleLogsChange(e.target.value)}
          placeholder={`2026-03-23T14:02:11.034Z ERROR [payments-service] Failed to acquire connection from pool…
2026-03-23T14:02:11.301Z ERROR [payments-service] POST /api/v1/charges failed: PoolTimeoutError
2026-03-23T14:02:12.150Z WARN  [gateway] Upstream returned 503 — retry 1/3
2026-03-23T14:02:13.220Z WARN  [gateway] Circuit breaker OPEN — failure rate 78%`}
          disabled={isLoading}
          className="w-full min-h-[200px] resize-y rounded-lg border border-border bg-surface-input p-4 font-mono text-sm text-primary placeholder:text-muted focus:border-gold focus:ring-1 focus:ring-gold/30 disabled:opacity-50 transition-colors"
          spellCheck={false}
        />
        {/* Character count */}
        {logs.length > 0 && (
          <span
            className={`absolute bottom-3 right-3 text-xs ${
              logs.length > MAX_LOG_LENGTH ? "text-confidence-low" : "text-muted"
            }`}
          >
            {logs.length.toLocaleString()} / {MAX_LOG_LENGTH.toLocaleString()}
          </span>
        )}
      </div>

      {/* Validation error */}
      {validationError && (
        <p className="text-sm text-confidence-low">{validationError}</p>
      )}

      {/* Mode toggle */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-medium text-secondary">Mode</span>
        <div className="inline-flex rounded-lg border border-border bg-surface p-0.5">
          {(["detailed", "compact"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              disabled={isLoading}
              className={`rounded-md px-3 py-1 text-xs font-medium capitalize transition-colors ${
                mode === m
                  ? "bg-gold text-bg-primary"
                  : "text-secondary hover:text-primary"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleSubmit}
          disabled={isLoading || !logs.trim()}
          className="rounded-lg bg-gold px-6 py-2.5 text-sm font-semibold text-bg-primary transition-colors hover:bg-gold-light disabled:cursor-not-allowed disabled:opacity-40"
        >
          Analyze
        </button>
        <button
          onClick={handleLoadExample}
          disabled={isLoading}
          className="rounded-lg border border-gold/40 px-4 py-2.5 text-sm font-medium text-gold transition-colors hover:bg-gold/10 disabled:opacity-40"
        >
          Load example
        </button>
      </div>
    </div>
  );
}

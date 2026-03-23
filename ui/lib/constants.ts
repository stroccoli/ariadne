import type { IncidentType } from "./api";

// ---------------------------------------------------------------------------
// Sample logs — realistic database connection pool exhaustion scenario
// ---------------------------------------------------------------------------

export const SAMPLE_LOGS = `2026-03-23T14:02:11.034Z ERROR [payments-service] Failed to acquire connection from pool within 5000ms — pool exhausted (active: 20/20, idle: 0, pending: 34)
2026-03-23T14:02:11.035Z WARN  [payments-service] Connection pool saturation detected — pending requests exceed threshold (34 > 10)
2026-03-23T14:02:11.301Z ERROR [payments-service] POST /api/v1/charges failed: could not execute query — PoolTimeoutError after 5002ms
2026-03-23T14:02:11.832Z ERROR [payments-service] Transaction rollback for order_id=ORD-9821 — connection acquisition timeout
2026-03-23T14:02:12.150Z WARN  [gateway] Upstream payments-service returned 503 for request req_abc123 — retry 1/3
2026-03-23T14:02:12.455Z ERROR [payments-service] Health check /ready failed: unable to ping database within 2000ms
2026-03-23T14:02:13.001Z ERROR [payments-service] Failed to acquire connection from pool within 5000ms — pool exhausted (active: 20/20, idle: 0, pending: 41)
2026-03-23T14:02:13.220Z WARN  [gateway] Circuit breaker OPEN for payments-service — failure rate 78% exceeds threshold 50%
2026-03-23T14:02:14.007Z ERROR [order-service] Dependent call to payments-service timed out after 10000ms for order_id=ORD-9834
2026-03-23T14:02:14.890Z WARN  [alertmanager] FIRING: PaymentsServiceErrorRate > 25% for 2m — severity=critical`;

// ---------------------------------------------------------------------------
// Incident type labels & colors
// ---------------------------------------------------------------------------

export const INCIDENT_TYPE_LABELS: Record<IncidentType, string> = {
  timeout: "Timeout",
  dependency_failure: "Dependency Failure",
  database_issue: "Database Issue",
  memory_issue: "Memory Issue",
  unknown: "Unknown",
};

// ---------------------------------------------------------------------------
// Confidence thresholds
// ---------------------------------------------------------------------------

export const CONFIDENCE_THRESHOLDS = {
  high: 0.75,
  mid: 0.5,
} as const;

export function getConfidenceLevel(confidence: number): "high" | "mid" | "low" {
  if (confidence >= CONFIDENCE_THRESHOLDS.high) return "high";
  if (confidence >= CONFIDENCE_THRESHOLDS.mid) return "mid";
  return "low";
}

export function getConfidenceLabel(confidence: number): string {
  const level = getConfidenceLevel(confidence);
  if (level === "high") return "High";
  if (level === "mid") return "Medium";
  return "Low";
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export const MAX_LOG_LENGTH = 50_000;

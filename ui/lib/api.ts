// ---------------------------------------------------------------------------
// Ariadne API client — aligned with the real Pydantic models
// ---------------------------------------------------------------------------

export type IncidentType =
  | "timeout"
  | "dependency_failure"
  | "database_issue"
  | "memory_issue"
  | "unknown";

export type PromptMode = "detailed" | "compact";

// -- Request --

export interface AnalyzeRequest {
  logs: string;
  mode: PromptMode;
}

// -- Response --

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface AnalysisMetadata {
  retrieval_attempts: number;
  llm_calls: number;
  node_timings: Record<string, number>;
  usage: TokenUsage;
}

export interface AnalyzeResponse {
  incident_type: IncidentType;
  root_cause: string;
  confidence: number;
  recommended_actions: string[];
  metadata: AnalysisMetadata;
}

// -- Errors --

export type ApiErrorType = "network" | "server" | "unexpected";

export class ApiError extends Error {
  type: ApiErrorType;
  status?: number;

  constructor(type: ApiErrorType, message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.type = type;
    this.status = status;
  }
}

// -- Helpers --

function getBaseUrl(): string {
  // Empty string means same-origin (production: UI and API served from the same Fly.io host).
  // In development, set NEXT_PUBLIC_API_URL=http://localhost:8000 in .env.local.
  const url = process.env.NEXT_PUBLIC_API_URL ?? "";
  return url.replace(/\/+$/, "");
}

async function parseJsonSafe<T>(response: Response): Promise<T> {
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(
      "unexpected",
      `Expected JSON response but received ${response.headers.get("content-type") ?? "unknown content type"}`,
      response.status,
    );
  }
}

// -- Public API --

export async function analyzeIncident(
  logs: string,
  mode: PromptMode = "detailed",
): Promise<AnalyzeResponse> {
  const url = `${getBaseUrl()}/analyze`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // X-API-Key is required by the backend on POST /analyze.
        // In development, set NEXT_PUBLIC_API_KEY in ui/.env.local.
        // In production (Fly.io), it is baked in at Docker build time via --build-arg.
        "X-API-Key": process.env.NEXT_PUBLIC_API_KEY ?? "",
      },
      body: JSON.stringify({ logs, mode } satisfies AnalyzeRequest),
    });
  } catch {
    throw new ApiError(
      "network",
      "Could not reach the analysis service. Check your connection and try again.",
    );
  }

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    let detail = `Server error (${response.status})`;
    try {
      const parsed = JSON.parse(body) as { detail?: string };
      if (parsed.detail) detail = parsed.detail;
    } catch {
      // body wasn't JSON — use default detail
    }
    throw new ApiError("server", detail, response.status);
  }

  return parseJsonSafe<AnalyzeResponse>(response);
}

export async function checkHealth(): Promise<{ status: string }> {
  const url = `${getBaseUrl()}/health`;

  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new ApiError(
      "network",
      "Could not reach the analysis service.",
    );
  }

  if (!response.ok) {
    throw new ApiError(
      "server",
      response.status === 503
        ? "Ariadne is starting up. This usually takes a few seconds."
        : `Health check failed (${response.status})`,
      response.status,
    );
  }

  return parseJsonSafe<{ status: string }>(response);
}

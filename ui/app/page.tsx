"use client";

import { useState } from "react";
import Logo from "@/components/Logo";
import LogInput from "@/components/LogInput";
import LoadingState from "@/components/LoadingState";
import AnalysisResult from "@/components/AnalysisResult";
import ErrorMessage from "@/components/ErrorMessage";
import {
  analyzeIncident,
  ApiError,
  type AnalyzeResponse,
  type PromptMode,
} from "@/lib/api";

type AppState =
  | { view: "input" }
  | { view: "loading" }
  | { view: "result"; data: AnalyzeResponse }
  | { view: "error"; error: ApiError };

export default function Home() {
  const [state, setState] = useState<AppState>({ view: "input" });

  async function handleAnalyze(logs: string, mode: PromptMode) {
    setState({ view: "loading" });
    try {
      const data = await analyzeIncident(logs, mode);
      setState({ view: "result", data });
    } catch (err) {
      if (err instanceof ApiError) {
        setState({ view: "error", error: err });
      } else {
        setState({
          view: "error",
          error: new ApiError(
            "unexpected",
            "Something went wrong. Please try again.",
          ),
        });
      }
    }
  }

  function handleReset() {
    setState({ view: "input" });
  }

  return (
    <main className="mx-auto max-w-2xl px-4 py-8 sm:py-12">
      {/* Header */}
      <header className="mb-10">
        <Logo />
      </header>

      {/* Input state */}
      {state.view === "input" && (
        <div className="animate-fade-in">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-primary sm:text-4xl">
            From raw logs to root cause
          </h1>
          <p className="mb-8 text-base text-secondary">
            Paste your incident logs. Ariadne finds the thread.
          </p>
          <LogInput onSubmit={handleAnalyze} isLoading={false} />
        </div>
      )}

      {/* Loading state */}
      {state.view === "loading" && <LoadingState />}

      {/* Result state */}
      {state.view === "result" && (
        <AnalysisResult result={state.data} onReset={handleReset} />
      )}

      {/* Error state */}
      {state.view === "error" && (
        <ErrorMessage error={state.error} onRetry={handleReset} />
      )}

      {/* Footer */}
      <footer className="mt-16 border-t border-border pt-6 text-center text-xs text-muted">
        Ariadne — AI Incident Analyzer ·{" "}
        <a
          href="https://github.com/sctroccoli/ariadne"
          target="_blank"
          rel="noopener noreferrer"
          className="text-secondary hover:text-gold transition-colors"
        >
          GitHub
        </a>
      </footer>
    </main>
  );
}

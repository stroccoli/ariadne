import type { ApiError } from "@/lib/api";

interface ErrorMessageProps {
  error: ApiError;
  onRetry: () => void;
}

function getErrorContent(error: ApiError): {
  title: string;
  message: string;
  showRetry: boolean;
} {
  if (error.type === "network") {
    return {
      title: "Connection lost",
      message:
        "Could not reach the analysis service. Check your connection and try again.",
      showRetry: true,
    };
  }

  if (error.type === "server") {
    if (error.status === 503) {
      return {
        title: "Service warming up",
        message:
          "Ariadne is starting up. This usually takes a few seconds.",
        showRetry: true,
      };
    }
    return {
      title: "Analysis failed",
      message:
        error.message ||
        "The analysis pipeline encountered an error. Try again or simplify your logs.",
      showRetry: true,
    };
  }

  return {
    title: "Unexpected error",
    message: error.message || "Something went wrong. Please try again.",
    showRetry: true,
  };
}

export default function ErrorMessage({ error, onRetry }: ErrorMessageProps) {
  const { title, message, showRetry } = getErrorContent(error);

  return (
    <div className="animate-fade-in rounded-lg border border-confidence-low/30 bg-confidence-low/5 p-6 text-center">
      {/* Error icon */}
      <svg
        className="mx-auto mb-3 h-8 w-8 text-confidence-low"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>

      <h3 className="mb-1 text-lg font-semibold text-primary">{title}</h3>
      <p className="mb-4 text-sm text-secondary">{message}</p>

      {showRetry && (
        <button
          onClick={onRetry}
          className="rounded-lg border border-gold/40 px-4 py-2 text-sm font-medium text-gold transition-colors hover:bg-gold/10"
        >
          Try again
        </button>
      )}
    </div>
  );
}

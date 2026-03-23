"use client";

import { useEffect, useState } from "react";

const PIPELINE_STEPS = [
  { label: "Classifying incident…", delay: 0 },
  { label: "Retrieving context…", delay: 1500 },
  { label: "Analyzing root cause…", delay: 3000 },
  { label: "Building report…", delay: 5000 },
];

export default function LoadingState() {
  const [visibleCount, setVisibleCount] = useState(1);

  useEffect(() => {
    const timers = PIPELINE_STEPS.slice(1).map((step, i) =>
      setTimeout(() => setVisibleCount(i + 2), step.delay),
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="animate-fade-in py-8">
      <div className="relative mx-auto max-w-sm">
        {/* Vertical golden thread line */}
        <div className="absolute left-[11px] top-2 bottom-2 w-0.5 bg-gold/20">
          <div
            className="animate-pulse-gold w-full bg-gold"
            style={{
              height: `${((visibleCount - 1) / (PIPELINE_STEPS.length - 1)) * 100}%`,
              transition: "height 0.6s ease-out",
            }}
          />
        </div>

        <div className="space-y-6">
          {PIPELINE_STEPS.map((step, i) => {
            const isVisible = i < visibleCount;
            const isCurrent = i === visibleCount - 1;
            const isCompleted = i < visibleCount - 1;

            return (
              <div
                key={step.label}
                className={`flex items-center gap-4 transition-opacity duration-300 ${
                  isVisible ? "opacity-100" : "opacity-0"
                }`}
              >
                {/* Step indicator */}
                <div className="relative z-10 flex h-6 w-6 flex-shrink-0 items-center justify-center">
                  {isCompleted ? (
                    <svg
                      className="h-5 w-5 text-gold"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ) : isCurrent ? (
                    <div className="h-4 w-4 rounded-full border-2 border-gold border-t-transparent animate-spin-slow" />
                  ) : (
                    <div className="h-2 w-2 rounded-full bg-border" />
                  )}
                </div>

                {/* Step label */}
                <span
                  className={`text-sm font-medium ${
                    isCurrent
                      ? "text-primary"
                      : isCompleted
                        ? "text-secondary"
                        : "text-muted"
                  }`}
                >
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
